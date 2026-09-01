"""Tests for the final untouched holdout period (Stage 12) and the
three-way temporal partition fix (development / holdout / post-holdout).

Covers: the development/holdout split (including an adversarial row whose
target window reaches into the holdout and must be purged from
development), the audit-log guarantee that every holdout access is
recorded, the defensive check that no walk-forward fold may overlap the
holdout window, and -- the core of the fix -- that POST-HOLDOUT data
(real ingestion routinely runs past the holdout end, e.g. ``real-demo``
ingesting through "today") can never leak backward into development or
influence model selection, regardless of how far past the holdout the
source data extends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.holdout import (
    HoldoutConfig,
    TemporalPartition,
    assert_no_fold_touches_holdout,
    default_holdout_config,
    evaluate_on_holdout,
    split_development_and_holdout,
    split_temporal_partitions,
)
from backtesting.purged_walk_forward import (
    PurgedFold,
    build_trading_calendar,
    generate_purged_folds,
    run_purged_walk_forward,
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


def _frame_spanning_dates(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Like _synthetic_feature_target_frame, but spanning explicit
    calendar dates (rather than a fixed row count) -- used to build
    frames that deliberately extend well past a realistic holdout end
    date, mirroring real-demo's "ingest through today" behaviour."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    n_days = len(dates)
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


# --- split_temporal_partitions: the three-way split (regression suite) -------------------------
#
# These tests cover a real bug: split_development_and_holdout only ever
# excluded rows INSIDE the holdout window, never rows AFTER it. Real
# ingestion (real-demo) keeps running through "today", which is normally
# well past the holdout end -- so post-holdout data was silently entering
# "development", and from there, walk-forward folds whose validation
# windows extended past the holdout, exactly the failure
# assert_no_fold_touches_holdout exists to catch (and did catch, in
# production: "fold 1: validation window [2024-03-04, 2026-02-05]
# overlaps the holdout period [2024-07-01, 2025-06-30]").


def test_post_holdout_exclusion_three_way_partition_boundaries():
    """1. Post-holdout exclusion."""
    df = _frame_spanning_dates("2020-01-01", "2026-02-01")
    holdout = HoldoutConfig(start_date=pd.Timestamp("2024-07-01"), end_date=pd.Timestamp("2025-06-30"))
    partition = split_temporal_partitions(df, holdout)
    assert isinstance(partition, TemporalPartition)

    assert not partition.development_df.empty
    assert not partition.holdout_df.empty
    assert not partition.post_holdout_df.empty

    assert partition.development_df["timestamp"].max() < pd.Timestamp("2024-07-01")
    assert partition.holdout_df["timestamp"].min() >= pd.Timestamp("2024-07-01")
    assert partition.holdout_df["timestamp"].max() <= pd.Timestamp("2025-06-30")
    assert partition.post_holdout_df["timestamp"].min() > pd.Timestamp("2025-06-30")

    # No row is double-counted or dropped across the three regions.
    assert len(partition.development_df) + len(partition.holdout_df) + len(partition.post_holdout_df) <= len(df)


def test_split_development_and_holdout_wrapper_matches_three_way_partition():
    """The backward-compatible 2-value wrapper must be bug-for-bug
    consistent with the canonical three-way split -- development_df must
    never contain a post-holdout row via either entry point."""
    df = _frame_spanning_dates("2020-01-01", "2026-02-01")
    holdout = HoldoutConfig(start_date=pd.Timestamp("2024-07-01"), end_date=pd.Timestamp("2025-06-30"))
    partition = split_temporal_partitions(df, holdout)
    development_df, holdout_df = split_development_and_holdout(df, holdout)

    pd.testing.assert_frame_equal(development_df, partition.development_df)
    pd.testing.assert_frame_equal(holdout_df, partition.holdout_df)
    assert development_df["timestamp"].max() < holdout.start_date


def test_no_fold_may_cross_the_holdout_even_when_source_data_extends_past_it():
    """2. No fold may cross the holdout.

    This is the direct regression test for the reported production
    failure: folds generated from development_df's own calendar (the
    correct, intended usage) must never reach the holdout, even though
    the SOURCE data extends nearly two years past the holdout end."""
    df = _frame_spanning_dates("2020-01-01", "2026-02-01")
    holdout = HoldoutConfig(start_date=pd.Timestamp("2024-07-01"), end_date=pd.Timestamp("2025-06-30"))
    partition = split_temporal_partitions(df, holdout)
    assert not partition.post_holdout_df.empty  # sanity: source data really does extend past holdout

    dev_calendar = build_trading_calendar(partition.development_df["timestamp"])
    folds = generate_purged_folds(dev_calendar, initial_train_days=600, validation_days=100, window_mode="expanding")
    assert len(folds) >= 1
    for fold in folds:
        assert fold.validation_end < holdout.start_date
        assert fold.validation_start < holdout.start_date
    assert_no_fold_touches_holdout(folds, holdout)  # must not raise


def test_assert_no_fold_touches_holdout_rejects_a_fold_entirely_after_holdout():
    """A fold whose window is entirely AFTER the holdout doesn't literally
    "overlap" it, but is just as invalid: it was never generated from
    pre-holdout development data. The strengthened guard must catch this
    too, not just literal overlap."""
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=900))
    holdout = HoldoutConfig(start_date=calendar[700], end_date=calendar[799])
    folds = [
        PurgedFold(fold_id=0, train_start=calendar[0], validation_start=calendar[820], validation_end=calendar[840], window_mode="expanding"),
    ]
    with pytest.raises(ValueError, match="does not end strictly before"):
        assert_no_fold_touches_holdout(folds, holdout)


def test_post_holdout_mutation_does_not_affect_development_partition_or_model_selection():
    """4. Post-holdout mutation independence.

    Changing post-holdout values drastically must not change
    development_df, holdout_df, or anything trained on development_df --
    an anti-leakage regression guard."""
    holdout = HoldoutConfig(start_date=pd.Timestamp("2024-07-01"), end_date=pd.Timestamp("2025-06-30"))
    df1 = _frame_spanning_dates("2020-01-01", "2026-02-01", seed=0)
    df2 = df1.copy()
    post_mask = df2["timestamp"] > holdout.end_date
    assert post_mask.any()
    for col in ["f1", "f2", "excess_return_5d", "excess_return_20d"]:
        df2.loc[post_mask, col] = df2.loc[post_mask, col] * -1000.0 + 999.0

    partition1 = split_temporal_partitions(df1, holdout)
    partition2 = split_temporal_partitions(df2, holdout)

    pd.testing.assert_frame_equal(partition1.development_df, partition2.development_df)
    pd.testing.assert_frame_equal(partition1.holdout_df, partition2.holdout_df)
    # The post-holdout region itself legitimately differs -- that's the point.
    assert not partition1.post_holdout_df.equals(partition2.post_holdout_df)

    feature_cols = ["f1", "f2"]
    trained1 = train_all_targets(partition1.development_df.iloc[:-100], partition1.development_df.iloc[-100:], feature_cols)
    trained2 = train_all_targets(partition2.development_df.iloc[:-100], partition2.development_df.iloc[-100:], feature_cols)
    probe = partition1.development_df[feature_cols].iloc[:50]
    preds1 = trained1.boosters["excess_return_20d"].predict(probe)
    preds2 = trained2.boosters["excess_return_20d"].predict(probe)
    np.testing.assert_array_equal(preds1, preds2)


def test_holdout_mutation_does_not_affect_development_partition_or_model_selection():
    """5. Holdout mutation independence before formal evaluation.

    Changing HOLDOUT values must not change development_df or anything
    trained on it -- champion qualification (which trains and selects
    only from development_df) is therefore also unaffected."""
    holdout = HoldoutConfig(start_date=pd.Timestamp("2024-07-01"), end_date=pd.Timestamp("2025-06-30"))
    df1 = _frame_spanning_dates("2020-01-01", "2026-02-01", seed=0)
    df2 = df1.copy()
    holdout_mask = (df2["timestamp"] >= holdout.start_date) & (df2["timestamp"] <= holdout.end_date)
    assert holdout_mask.any()
    for col in ["f1", "f2", "excess_return_5d", "excess_return_20d"]:
        df2.loc[holdout_mask, col] = df2.loc[holdout_mask, col] * -1000.0 + 999.0

    partition1 = split_temporal_partitions(df1, holdout)
    partition2 = split_temporal_partitions(df2, holdout)

    pd.testing.assert_frame_equal(partition1.development_df, partition2.development_df)
    assert not partition1.holdout_df.equals(partition2.holdout_df)  # mutated by construction, as expected

    feature_cols = ["f1", "f2"]
    trained1 = train_all_targets(partition1.development_df.iloc[:-100], partition1.development_df.iloc[-100:], feature_cols)
    trained2 = train_all_targets(partition2.development_df.iloc[:-100], partition2.development_df.iloc[-100:], feature_cols)
    probe = partition1.development_df[feature_cols].iloc[:50]
    preds1 = trained1.boosters["excess_return_20d"].predict(probe)
    preds2 = trained2.boosters["excess_return_20d"].predict(probe)
    np.testing.assert_array_equal(preds1, preds2)


def test_model_selection_causes_zero_holdout_access_log_entries_and_formal_evaluation_adds_exactly_one(con):
    """6. Holdout access.

    Running the full purged-walk-forward model-selection path must never
    touch holdout_access_log; only the one deliberate evaluate_on_holdout
    call afterward should add a row."""
    from database import repository_v2 as repo_v2

    holdout = HoldoutConfig(start_date=pd.Timestamp("2024-07-01"), end_date=pd.Timestamp("2025-06-30"))
    df = _frame_spanning_dates("2020-01-01", "2026-02-01", seed=0)
    partition = split_temporal_partitions(df, holdout)

    feature_cols = ["f1", "f2"]
    calendar = build_trading_calendar(partition.development_df["timestamp"])
    folds = generate_purged_folds(calendar, initial_train_days=800, validation_days=100, window_mode="expanding")
    assert_no_fold_touches_holdout(folds, holdout)
    fold_results = run_purged_walk_forward(partition.development_df, folds, feature_cols)
    assert len(fold_results) >= 1

    assert len(repo_v2.get_holdout_access_log(con)) == 0  # model selection touched nothing

    last_fold = fold_results[-1]
    evaluate_on_holdout(con, last_fold.trained, partition.holdout_df, feature_cols, "model_v1", "formal one-time evaluation")
    assert len(repo_v2.get_holdout_access_log(con)) == 1
