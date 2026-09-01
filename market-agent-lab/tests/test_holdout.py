"""Tests for the final untouched holdout period (Stage 12).

Covers: the development/holdout split (including an adversarial row whose
target window reaches into the holdout and must be purged from
development), the audit-log guarantee that every holdout access is
recorded, and the defensive check that no walk-forward fold may overlap
the holdout window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.holdout import (
    HoldoutConfig,
    assert_no_fold_touches_holdout,
    default_holdout_config,
    evaluate_on_holdout,
    split_development_and_holdout,
)
from backtesting.purged_walk_forward import (
    PurgedFold,
    build_trading_calendar,
    generate_purged_folds,
)
from database.db import fresh_connection
from database.schema import init_schema
from models.train import get_feature_columns, train_all_targets


def _synthetic_feature_target_frame(n_days: int = 900) -> pd.DataFrame:
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


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        init_schema(c)
        yield c


# --- HoldoutConfig -------------------------------------------------------------------------


def test_holdout_config_rejects_start_on_or_after_end():
    with pytest.raises(ValueError):
        HoldoutConfig(start_date=pd.Timestamp("2024-01-01"), end_date=pd.Timestamp("2024-01-01"))


def test_default_holdout_config_reads_settings():
    holdout = default_holdout_config()
    assert holdout.start_date < holdout.end_date


# --- split_development_and_holdout -----------------------------------------------------------


def test_holdout_df_is_exactly_the_configured_window():
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    _development_df, holdout_df = split_development_and_holdout(df, holdout)
    assert (holdout_df["timestamp"] >= holdout.start_date).all()
    assert (holdout_df["timestamp"] <= holdout.end_date).all()
    assert len(holdout_df) == 100  # calendar[700..799] inclusive, one symbol


def test_development_df_never_overlaps_holdout_window():
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    development_df, _holdout_df = split_development_and_holdout(df, holdout)
    assert not ((development_df["timestamp"] >= holdout.start_date) & (development_df["timestamp"] <= holdout.end_date)).any()


def test_adversarial_row_whose_target_window_reaches_holdout_is_purged_from_development():
    """The same vulnerability Stage 11 fixes for walk-forward folds applies
    to the holdout boundary: a row timestamped strictly before
    holdout.start_date but whose 20-day target window reaches into it must
    not end up in development_df."""
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])

    poisoned_ts = calendar[700 - 10]  # target window reaches idx 710, inside the holdout
    poisoned_row_mask = df["timestamp"] == poisoned_ts
    assert poisoned_row_mask.sum() == 1

    development_df, _holdout_df = split_development_and_holdout(df, holdout)
    assert not (development_df["timestamp"] == poisoned_ts).any()


def test_row_well_before_holdout_with_no_target_overlap_remains_in_development():
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    safe_ts = calendar[500]
    development_df, _holdout_df = split_development_and_holdout(df, holdout)
    assert (development_df["timestamp"] == safe_ts).any()


# --- assert_no_fold_touches_holdout ------------------------------------------------------------


def test_assert_no_fold_touches_holdout_passes_for_disjoint_folds():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=900))
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    folds = [
        PurgedFold(fold_id=0, train_start=calendar[0], validation_start=calendar[400], validation_end=calendar[419], window_mode="expanding"),
        PurgedFold(fold_id=1, train_start=calendar[0], validation_start=calendar[420], validation_end=calendar[439], window_mode="expanding"),
    ]
    assert_no_fold_touches_holdout(folds, holdout)  # must not raise


def test_assert_no_fold_touches_holdout_raises_when_a_fold_overlaps():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=900))
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    folds = [
        PurgedFold(fold_id=0, train_start=calendar[0], validation_start=calendar[750], validation_end=calendar[760], window_mode="expanding"),
    ]
    with pytest.raises(ValueError, match="overlaps the holdout period"):
        assert_no_fold_touches_holdout(folds, holdout)


def test_generate_purged_folds_from_development_calendar_never_touches_holdout():
    """Integration check for the intended usage: the holdout is the FINAL
    period (no development data after it), so folds generated from
    development_df's own calendar naturally stop before the holdout and
    pass the defensive assertion. (A holdout carved out of the *middle* of
    the series, with development data continuing after it, is not this
    project's intended usage -- see the two mid-series-holdout tests above
    for split-correctness coverage of that case instead.)"""
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[800], end_date=calendar[899])  # trailing 100 days = the tail
    development_df, _holdout_df = split_development_and_holdout(df, holdout)

    dev_calendar = build_trading_calendar(development_df["timestamp"])
    folds = generate_purged_folds(dev_calendar, initial_train_days=400, validation_days=100, window_mode="expanding")
    assert len(folds) >= 1
    assert_no_fold_touches_holdout(folds, holdout)  # must not raise


# --- evaluate_on_holdout -----------------------------------------------------------------------


def test_evaluate_on_holdout_rejects_empty_frame(con):
    df = _synthetic_feature_target_frame()
    trained = train_all_targets(df.iloc[:400], df.iloc[400:500], get_feature_columns(df))
    with pytest.raises(ValueError):
        evaluate_on_holdout(con, trained, df.iloc[0:0], get_feature_columns(df), "m1", "final report")


def test_evaluate_on_holdout_logs_exactly_one_row_per_call(con):
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    development_df, holdout_df = split_development_and_holdout(df, holdout)
    feature_cols = ["f1", "f2"]
    trained = train_all_targets(development_df.iloc[:-100], development_df.iloc[-100:], feature_cols)

    from database import repository_v2 as repo_v2

    assert len(repo_v2.get_holdout_access_log(con)) == 0
    result = evaluate_on_holdout(con, trained, holdout_df, feature_cols, "model_v1", "final V0.2 report")
    log = repo_v2.get_holdout_access_log(con)
    assert len(log) == 1
    assert log.iloc[0]["model_version"] == "model_v1"
    assert log.iloc[0]["n_rows"] == result.n_rows == len(holdout_df)

    evaluate_on_holdout(con, trained, holdout_df, feature_cols, "model_v1", "final V0.2 report")
    assert len(repo_v2.get_holdout_access_log(con)) == 2  # every call logs, no dedup


def test_evaluate_on_holdout_predictions_stay_inside_holdout_window(con):
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    development_df, holdout_df = split_development_and_holdout(df, holdout)
    feature_cols = ["f1", "f2"]
    trained = train_all_targets(development_df.iloc[:-100], development_df.iloc[-100:], feature_cols)

    result = evaluate_on_holdout(con, trained, holdout_df, feature_cols, "model_v1", "final V0.2 report")
    if not result.predictions.empty:
        assert (result.predictions["timestamp"] >= holdout.start_date).all()
        assert (result.predictions["timestamp"] <= holdout.end_date).all()
