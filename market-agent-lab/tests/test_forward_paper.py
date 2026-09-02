"""Tests for backtesting/forward_paper.py (V0.3 Stage 10): forward-paper
evaluation of an already-frozen champion, with no training or model
selection performed here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.forward_paper import (
    evaluate_on_forward_paper,
    load_frozen_champion_for_forward_paper,
)
from database import repository_v2 as repo_v2
from database.db import fresh_connection
from database.schema import init_schema
from models.registry import ModelPeriods, promote_to_champion, register_model
from models.train import get_feature_columns, train_all_targets


def _synthetic_frame(n_days=700, seed=1) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "symbol": "SYN_X", "timestamp": dates, "f1": rng.normal(0, 1, n_days), "f2": rng.normal(0, 1, n_days),
            "excess_return_5d": rng.normal(0, 0.02, n_days), "excess_return_20d": rng.normal(0, 0.04, n_days),
            "positive_5d": rng.integers(0, 2, n_days).astype(float), "positive_20d": rng.integers(0, 2, n_days).astype(float),
        }
    )


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        init_schema(c)
        yield c


def _register_a_champion(con, df, feature_cols) -> str:
    trained = train_all_targets(df.iloc[:500], df.iloc[500:600], feature_cols)
    periods = ModelPeriods(
        training_start=df["timestamp"].iloc[0], training_end=df["timestamp"].iloc[499],
        validation_start=df["timestamp"].iloc[500], validation_end=df["timestamp"].iloc[599],
    )
    metrics = {"excess_return_20d": {"information_coefficient": 0.6, "sharpe_ratio": 1.2}}
    version = register_model(con, trained, "fv1", periods, metrics, role="CHALLENGER")
    promote_to_champion(con, version)
    return version


def test_load_frozen_champion_for_forward_paper_raises_when_none_exists(con):
    with pytest.raises(ValueError, match="no champion"):
        load_frozen_champion_for_forward_paper(con)


def test_load_frozen_champion_for_forward_paper_returns_the_champion_version(con):
    df = _synthetic_frame()
    feature_cols = get_feature_columns(df)
    version = _register_a_champion(con, df, feature_cols)
    assert load_frozen_champion_for_forward_paper(con) == version


def test_evaluate_on_forward_paper_rejects_empty_frame(con):
    df = _synthetic_frame()
    feature_cols = get_feature_columns(df)
    version = _register_a_champion(con, df, feature_cols)
    with pytest.raises(ValueError):
        evaluate_on_forward_paper(con, df.iloc[0:0], feature_cols, version, "test")


def test_evaluate_on_forward_paper_logs_exactly_one_row_per_call_and_trains_nothing(con):
    df = _synthetic_frame()
    feature_cols = get_feature_columns(df)
    version = _register_a_champion(con, df, feature_cols)
    forward_df = df.iloc[600:]

    assert len(repo_v2.get_forward_paper_access_log(con)) == 0
    registry_row_count_before = len(con.execute("SELECT * FROM model_registry").fetchdf())

    result = evaluate_on_forward_paper(con, forward_df, feature_cols, version, "forward paper test")

    log = repo_v2.get_forward_paper_access_log(con)
    assert len(log) == 1
    assert log.iloc[0]["model_version"] == version
    assert log.iloc[0]["n_rows"] == result.n_rows == len(forward_df)

    # No new model was registered -- this function trains and selects nothing.
    registry_row_count_after = len(con.execute("SELECT * FROM model_registry").fetchdf())
    assert registry_row_count_after == registry_row_count_before

    evaluate_on_forward_paper(con, forward_df, feature_cols, version, "forward paper test")
    assert len(repo_v2.get_forward_paper_access_log(con)) == 2  # every call logs, no dedup


def test_evaluate_on_forward_paper_predictions_use_the_exact_frozen_model(con):
    """The predictions must come from the SAME booster that was
    registered as champion -- re-running evaluate_on_forward_paper twice
    on the same data must produce identical predictions (no retraining
    introducing any nondeterminism beyond the model's own fixed seed)."""
    df = _synthetic_frame()
    feature_cols = get_feature_columns(df)
    version = _register_a_champion(con, df, feature_cols)
    forward_df = df.iloc[600:]

    first = evaluate_on_forward_paper(con, forward_df, feature_cols, version, "run 1")
    second = evaluate_on_forward_paper(con, forward_df, feature_cols, version, "run 2")
    pd.testing.assert_series_equal(
        first.predictions["predicted_excess_return_20d"].reset_index(drop=True),
        second.predictions["predicted_excess_return_20d"].reset_index(drop=True),
    )
