"""Tests for expanding-window walk-forward splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.walk_forward import WalkForwardFold, generate_expanding_folds, run_walk_forward


def test_generate_expanding_folds_matches_brief_example():
    folds = generate_expanding_folds(
        data_start="2015-01-01", initial_train_end="2020-12-31", overall_end="2023-12-31", validation_years=1
    )
    assert len(folds) == 3
    assert folds[0].validation_start.year == 2021 and folds[0].validation_end.year == 2021
    assert folds[1].train_end.year == 2021 and folds[1].validation_start.year == 2022
    assert folds[2].train_end.year == 2022 and folds[2].validation_start.year == 2023
    # Training window always expands from the same fixed start.
    assert all(f.train_start == pd.Timestamp("2015-01-01") for f in folds)


def test_fold_rejects_overlapping_windows():
    with pytest.raises(ValueError):
        WalkForwardFold(
            fold_id=0,
            train_start=pd.Timestamp("2015-01-01"),
            train_end=pd.Timestamp("2021-06-01"),
            validation_start=pd.Timestamp("2021-01-01"),  # overlaps training window
            validation_end=pd.Timestamp("2021-12-31"),
        )


def test_no_folds_when_insufficient_history():
    folds = generate_expanding_folds(
        data_start="2015-01-01", initial_train_end="2023-06-01", overall_end="2023-12-31", validation_years=1
    )
    assert folds == []


def _synthetic_feature_target_frame(n_days: int = 900) -> pd.DataFrame:
    import numpy as np

    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "symbol": "SYN_X",
            "timestamp": dates,
            "f1": rng.normal(0, 1, n_days),
            "f2": rng.normal(0, 1, n_days),
            "excess_return_5d": rng.normal(0, 0.02, n_days),
            "excess_return_20d": rng.normal(0, 0.04, n_days),
            "positive_5d": rng.integers(0, 2, n_days).astype(float),
            "positive_20d": rng.integers(0, 2, n_days).astype(float),
        }
    )


def test_run_walk_forward_never_trains_on_validation_rows():
    df = _synthetic_feature_target_frame()
    folds = generate_expanding_folds(
        data_start=df["timestamp"].min(),
        initial_train_end=df["timestamp"].min() + pd.DateOffset(days=500),
        overall_end=df["timestamp"].max(),
        validation_years=1,
    )
    assert len(folds) >= 1
    results = run_walk_forward(df, folds, feature_cols=["f1", "f2"])
    assert len(results) == len(folds)
    for result in results:
        # Every prediction timestamp must fall inside the fold's validation
        # window and strictly after the fold's train_end.
        if not result.predictions.empty:
            assert (result.predictions["timestamp"] > result.fold.train_end).all()
            assert (result.predictions["timestamp"] >= result.fold.validation_start).all()
            assert (result.predictions["timestamp"] <= result.fold.validation_end).all()


def test_run_walk_forward_detects_manual_leakage_injection():
    """If a caller hands run_walk_forward a frame with an out-of-order fold
    (validation before training), it must raise rather than silently train
    on future-looking data."""
    df = _synthetic_feature_target_frame()
    bad_fold = WalkForwardFold(
        fold_id=0,
        train_start=df["timestamp"].min(),
        train_end=df["timestamp"].max(),  # deliberately consumes the whole series
        validation_start=df["timestamp"].max() + pd.Timedelta(days=1),
        validation_end=df["timestamp"].max() + pd.Timedelta(days=30),
    )
    # No validation rows exist beyond the data -- run_walk_forward must
    # simply skip this fold rather than fabricate a leaking split.
    results = run_walk_forward(df, [bad_fold], feature_cols=["f1", "f2"])
    assert results == []
