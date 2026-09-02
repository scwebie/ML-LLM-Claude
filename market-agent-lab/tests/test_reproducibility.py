"""Tests for V0.3 Stage 13: model registry reproducibility provenance.

Covers models/reproducibility.py's standalone hashing utilities, that
models.train.train_all_targets populates a TrainedModels' provenance
fields, that models.registry.register_model persists them into
model_registry (git commit, target-definition hash, random seed, data
fingerprint, artifact hash), and that re-running the same training call
with the same data and seed produces an identical data fingerprint and
(with LightGBM's deterministic mode) identical predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from database.db import fresh_connection
from database.schema import init_schema
from models import reproducibility
from models.registry import (
    ModelPeriods,
    load_model,
    register_model,
    verify_artifact_reproducibility,
)
from models.train import compute_excess_return_targets, prepare_training_frame, train_all_targets


def _synthetic_market(symbol: str, n: int, seed: int, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(
        {
            "symbol": symbol, "timestamp": dates, "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close, "adjusted_close": close, "volume": 1_000_000,
        }
    )


def _training_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    market = pd.concat([_synthetic_market(f"SYN_{i}", n, seed=seed + i) for i in range(3)], ignore_index=True)
    bench = _synthetic_market("SYN_BENCH", n, seed=seed + 99, drift=0.0)
    targets = compute_excess_return_targets(market, bench)

    feature_rows = []
    for symbol in market["symbol"].unique():
        sym_dates = market[market["symbol"] == symbol]["timestamp"]
        for ts in sym_dates:
            feature_rows.append({"symbol": symbol, "timestamp": ts, "f1": rng.normal(), "f2": rng.normal()})
    features = pd.DataFrame(feature_rows)

    df = prepare_training_frame(features, targets)
    return df, ["f1", "f2"]


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["timestamp"] < df["timestamp"].quantile(0.7)]
    val_df = df[df["timestamp"] >= df["timestamp"].quantile(0.7)]
    return train_df, val_df


# --- reproducibility.py standalone utilities -----------------------------------------------


def test_get_git_commit_returns_a_hex_string_or_none():
    commit = reproducibility.get_git_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_hash_source_is_deterministic_for_the_same_function():
    h1 = reproducibility.hash_source(compute_excess_return_targets)
    h2 = reproducibility.hash_source(compute_excess_return_targets)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


def test_hash_source_differs_for_different_functions():
    h_targets = reproducibility.hash_source(compute_excess_return_targets)
    h_prepare = reproducibility.hash_source(prepare_training_frame)
    assert h_targets != h_prepare


def test_compute_data_fingerprint_is_deterministic_for_identical_data():
    df, feature_cols = _training_frame()
    train_df, val_df = _split(df)
    fp1 = reproducibility.compute_data_fingerprint(train_df, val_df, feature_cols, ["excess_return_5d"])
    fp2 = reproducibility.compute_data_fingerprint(train_df, val_df, feature_cols, ["excess_return_5d"])
    assert fp1 == fp2


def test_compute_data_fingerprint_changes_when_data_changes():
    df, feature_cols = _training_frame(seed=0)
    other_df, _ = _training_frame(seed=1)
    train_df, val_df = _split(df)
    other_train_df, other_val_df = _split(other_df)
    fp = reproducibility.compute_data_fingerprint(train_df, val_df, feature_cols, ["excess_return_5d"])
    other_fp = reproducibility.compute_data_fingerprint(other_train_df, other_val_df, feature_cols, ["excess_return_5d"])
    assert fp != other_fp


def test_compute_artifact_hash_is_deterministic_and_detects_a_changed_file(tmp_path):
    (tmp_path / "excess_return_5d.txt").write_text("tree A content")
    (tmp_path / "excess_return_20d.txt").write_text("tree B content")
    h1 = reproducibility.compute_artifact_hash(tmp_path)
    h2 = reproducibility.compute_artifact_hash(tmp_path)
    assert h1 == h2

    (tmp_path / "excess_return_5d.txt").write_text("tree A content -- TAMPERED")
    h3 = reproducibility.compute_artifact_hash(tmp_path)
    assert h3 != h1


def test_extract_seed_finds_the_seed_from_a_hyperparameters_dict():
    hyperparameters = {"regression": {"objective": "regression", "seed": 42}, "classification": {"objective": "binary", "seed": 42}}
    assert reproducibility.extract_seed(hyperparameters) == 42


def test_extract_seed_returns_none_when_absent():
    assert reproducibility.extract_seed({"regression": {"objective": "regression"}}) is None


# --- train_all_targets populates provenance fields ------------------------------------------


def test_train_all_targets_populates_reproducibility_fields():
    df, feature_cols = _training_frame()
    train_df, val_df = _split(df)
    trained = train_all_targets(train_df, val_df, feature_cols)
    assert trained.data_fingerprint is not None
    assert trained.target_definition_hash == reproducibility.hash_source(compute_excess_return_targets)
    assert trained.random_seed == 42


def test_retraining_identical_data_and_seed_produces_identical_fingerprint_and_predictions():
    """The core Stage 13 requirement: re-running the same experiment with
    the same data and seed must produce an identical data fingerprint and
    (with LightGBM's deterministic mode) numerically identical predictions
    -- not merely 'close', since DEFAULT_HYPERPARAMETERS now sets
    deterministic=True/force_row_wise=True specifically to guarantee
    this."""
    df, feature_cols = _training_frame()
    train_df, val_df = _split(df)

    trained_a = train_all_targets(train_df, val_df, feature_cols)
    trained_b = train_all_targets(train_df, val_df, feature_cols)

    assert trained_a.data_fingerprint == trained_b.data_fingerprint
    assert trained_a.random_seed == trained_b.random_seed

    val_rows = val_df.dropna(subset=["excess_return_5d", *feature_cols])
    preds_a = trained_a.boosters["excess_return_5d"].predict(val_rows[feature_cols])
    preds_b = trained_b.boosters["excess_return_5d"].predict(val_rows[feature_cols])
    assert np.array_equal(preds_a, preds_b)


# --- register_model / load_model / verify_artifact_reproducibility --------------------------


def _periods(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> ModelPeriods:
    return ModelPeriods(
        training_start=train_df["timestamp"].min(), training_end=train_df["timestamp"].max(),
        validation_start=val_df["timestamp"].min(), validation_end=val_df["timestamp"].max(),
    )


def test_register_model_persists_all_reproducibility_fields():
    df, feature_cols = _training_frame()
    train_df, val_df = _split(df)
    trained = train_all_targets(train_df, val_df, feature_cols)

    with fresh_connection(":memory:") as con:
        init_schema(con)
        version = register_model(con, trained, "fv1", _periods(df, train_df, val_df), metrics={}, role="CHALLENGER")
        _, record = load_model(con, version)

        assert record["target_definition_hash"] == trained.target_definition_hash
        assert record["random_seed"] == 42
        assert record["data_fingerprint"] == trained.data_fingerprint
        assert record["artifact_hash"] is not None
        # git_commit may legitimately be None outside a git checkout, but
        # must never raise or be silently dropped from the record.
        assert "git_commit" in record


def test_verify_artifact_reproducibility_passes_for_an_untouched_artifact():
    df, feature_cols = _training_frame()
    train_df, val_df = _split(df)
    trained = train_all_targets(train_df, val_df, feature_cols)

    with fresh_connection(":memory:") as con:
        init_schema(con)
        version = register_model(con, trained, "fv1", _periods(df, train_df, val_df), metrics={}, role="CHALLENGER")
        result = verify_artifact_reproducibility(con, version)
        assert result["matches"] is True
        assert result["recorded_artifact_hash"] == result["current_artifact_hash"]


def test_verify_artifact_reproducibility_detects_a_tampered_artifact():
    df, feature_cols = _training_frame()
    train_df, val_df = _split(df)
    trained = train_all_targets(train_df, val_df, feature_cols)

    with fresh_connection(":memory:") as con:
        init_schema(con)
        version = register_model(con, trained, "fv1", _periods(df, train_df, val_df), metrics={}, role="CHALLENGER")
        _, record = load_model(con, version)
        artifact_dir = record["artifact_path"]

        from pathlib import Path

        booster_file = next(Path(artifact_dir).glob("*.txt"))
        booster_file.write_text(booster_file.read_text() + "\n# tampered")

        result = verify_artifact_reproducibility(con, version)
        assert result["matches"] is False


def test_verify_artifact_reproducibility_raises_for_unknown_model_version():
    with fresh_connection(":memory:") as con:
        init_schema(con)
        with pytest.raises(KeyError):
            verify_artifact_reproducibility(con, "does_not_exist")
