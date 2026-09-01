"""Tests for the purged + embargoed + nested walk-forward evaluator
(Stage 11). The "adversarial" tests deliberately construct rows whose
target-realization window reaches into an evaluation period -- exactly
the kind of row a naive 'timestamp < validation_start' split would
silently leak -- and assert the purge/embargo machinery excludes them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    PurgedFold,
    build_outer_train_eligibility,
    build_trading_calendar,
    compute_purge_embargo_mask,
    default_hyperparameter_grid,
    generate_purged_folds,
    run_nested_purged_walk_forward,
    run_purged_walk_forward,
)


@pytest.fixture
def calendar():
    return pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=300))


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


# --- compute_purge_embargo_mask ---------------------------------------------------------


def test_row_well_before_eval_window_with_no_target_overlap_is_eligible(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[50]])
    mask = compute_purge_embargo_mask(row_ts, calendar, eval_start, eval_end, horizon_days=20, embargo_days=5)
    assert bool(mask.iloc[0]) is True


def test_row_whose_target_window_reaches_into_eval_start_is_purged(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[90]])  # target end = idx 110, inside [100, 104]... reaches past eval_start
    mask = compute_purge_embargo_mask(row_ts, calendar, eval_start, eval_end, horizon_days=20, embargo_days=5)
    assert bool(mask.iloc[0]) is False


def test_row_exactly_at_purge_boundary_is_purged_not_off_by_one(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[100 - MAX_TARGET_HORIZON_DAYS]])  # target end == eval_start exactly
    mask = compute_purge_embargo_mask(
        row_ts, calendar, eval_start, eval_end, horizon_days=MAX_TARGET_HORIZON_DAYS, embargo_days=5
    )
    assert bool(mask.iloc[0]) is False


def test_row_one_trading_day_further_back_than_purge_boundary_is_eligible(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[100 - MAX_TARGET_HORIZON_DAYS - 1]])
    mask = compute_purge_embargo_mask(
        row_ts, calendar, eval_start, eval_end, horizon_days=MAX_TARGET_HORIZON_DAYS, embargo_days=5
    )
    assert bool(mask.iloc[0]) is True


def test_rows_inside_eval_window_itself_are_excluded(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[102]])
    mask = compute_purge_embargo_mask(row_ts, calendar, eval_start, eval_end)
    assert bool(mask.iloc[0]) is False


def test_row_within_embargo_window_after_eval_end_is_excluded(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[104 + DEFAULT_EMBARGO_DAYS]])
    mask = compute_purge_embargo_mask(row_ts, calendar, eval_start, eval_end, embargo_days=DEFAULT_EMBARGO_DAYS)
    assert bool(mask.iloc[0]) is False


def test_row_beyond_embargo_window_after_eval_end_is_eligible(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series([calendar[104 + DEFAULT_EMBARGO_DAYS + 1]])
    mask = compute_purge_embargo_mask(row_ts, calendar, eval_start, eval_end, embargo_days=DEFAULT_EMBARGO_DAYS)
    assert bool(mask.iloc[0]) is True


def test_multiple_rows_masked_independently_and_index_is_preserved(calendar):
    eval_start, eval_end = calendar[100], calendar[104]
    row_ts = pd.Series(
        [calendar[50], calendar[95], calendar[102], calendar[106], calendar[200]], index=[10, 11, 12, 13, 14]
    )
    mask = compute_purge_embargo_mask(row_ts, calendar, eval_start, eval_end, horizon_days=20, embargo_days=5)
    assert list(mask.index) == [10, 11, 12, 13, 14]
    assert list(mask) == [True, False, False, False, True]


# --- fold generation ---------------------------------------------------------------------


def test_generate_purged_folds_expanding_train_start_always_zero(calendar):
    folds = generate_purged_folds(calendar, initial_train_days=100, validation_days=20, window_mode="expanding")
    assert len(folds) >= 2
    assert all(f.train_start == calendar[0] for f in folds)
    assert folds[0].validation_start == calendar[100]
    assert folds[0].validation_end == calendar[119]
    assert folds[1].validation_start == calendar[120]


def test_generate_purged_folds_rolling_train_start_advances(calendar):
    folds = generate_purged_folds(
        calendar, initial_train_days=100, validation_days=20, window_mode="rolling", rolling_train_days=50
    )
    assert folds[0].train_start == calendar[100 - 50]
    assert folds[1].train_start == calendar[120 - 50]


def test_generate_purged_folds_rolling_requires_rolling_train_days(calendar):
    with pytest.raises(ValueError):
        generate_purged_folds(calendar, initial_train_days=100, validation_days=20, window_mode="rolling")


def test_generate_purged_folds_rejects_unknown_window_mode(calendar):
    with pytest.raises(ValueError):
        generate_purged_folds(calendar, initial_train_days=100, validation_days=20, window_mode="bogus")


def test_generate_purged_folds_stops_before_running_past_calendar_end(calendar):
    folds = generate_purged_folds(calendar, initial_train_days=290, validation_days=20, window_mode="expanding")
    assert folds == []


# --- adversarial future-data-injection leakage tests --------------------------------------


def test_adversarial_row_would_leak_under_naive_split_but_is_excluded_by_purge():
    """Demonstrates the exact vulnerability purge+embargo fixes: a row
    timestamped strictly before validation_start -- which a naive
    'timestamp < validation_start' split (as in a plain expanding-window
    walk-forward with no purging) would happily include as training data
    -- has its 20-day target window reach into the validation period.
    Purge must exclude it."""
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    val_start_idx, val_end_idx = 700, 719
    validation_start, validation_end = calendar[val_start_idx], calendar[val_end_idx]

    poisoned_ts = calendar[val_start_idx - 10]  # target window reaches idx 700+10, inside [700, 719]
    poisoned_row_mask = df["timestamp"] == poisoned_ts
    assert poisoned_row_mask.sum() == 1, "test setup: exactly one row at the poisoned timestamp"

    naive_included = (df["timestamp"] == poisoned_ts) & (df["timestamp"] < validation_start)
    assert naive_included.any(), "test setup: a naive split would include this row"

    fold = PurgedFold(
        fold_id=0, train_start=calendar[0], validation_start=validation_start,
        validation_end=validation_end, window_mode="expanding",
    )
    candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar)
    train_mask = candidate_mask & eligible_mask

    assert candidate_mask[poisoned_row_mask].all()  # naive filter alone would have kept it
    assert not eligible_mask[poisoned_row_mask].any()  # purge marks it ineligible
    assert not train_mask[poisoned_row_mask].any()  # final training set excludes it


def test_adversarial_row_just_after_validation_end_would_leak_without_embargo():
    """A row timestamped one trading day after validation_end has no
    target-window overlap with the validation period at all (so purge
    alone would keep it), but embargo must still exclude it as a buffer
    against residual serial correlation."""
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    val_start_idx, val_end_idx = 300, 319
    validation_start, validation_end = calendar[val_start_idx], calendar[val_end_idx]

    poisoned_ts = calendar[val_end_idx + 1]
    poisoned_row_mask = df["timestamp"] == poisoned_ts

    mask = compute_purge_embargo_mask(
        df["timestamp"], calendar, validation_start, validation_end,
        horizon_days=MAX_TARGET_HORIZON_DAYS, embargo_days=DEFAULT_EMBARGO_DAYS,
    )
    assert not mask[poisoned_row_mask.values].any()


def test_purge_measurably_shrinks_training_set_relative_to_naive_candidate_window():
    """On a realistic run, purge+embargo must actually remove rows -- if it
    never does, the guard is a no-op and the earlier fold would be
    indistinguishable from V0.1's plain walk-forward."""
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    fold = PurgedFold(
        fold_id=0, train_start=calendar[0], validation_start=calendar[700],
        validation_end=calendar[719], window_mode="expanding",
    )
    candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar)
    assert candidate_mask.sum() > (candidate_mask & eligible_mask).sum()


# --- end-to-end run_purged_walk_forward ---------------------------------------------------


def test_run_purged_walk_forward_leakage_guard_holds_and_produces_results():
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    folds = generate_purged_folds(calendar, initial_train_days=500, validation_days=100, window_mode="expanding")
    assert len(folds) >= 1
    results = run_purged_walk_forward(df, folds, feature_cols=["f1", "f2"])
    assert len(results) == len(folds)
    for result in results:
        assert result.n_train_rows > 0
        assert result.n_purged_or_embargoed >= 0
        if not result.predictions.empty:
            assert (result.predictions["timestamp"] >= result.fold.validation_start).all()
            assert (result.predictions["timestamp"] <= result.fold.validation_end).all()


def test_run_purged_walk_forward_rolling_window_also_respects_leakage_guard():
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    folds = generate_purged_folds(
        calendar, initial_train_days=400, validation_days=100, window_mode="rolling", rolling_train_days=200
    )
    assert len(folds) >= 1
    results = run_purged_walk_forward(df, folds, feature_cols=["f1", "f2"])
    assert len(results) == len(folds)
    for result in results:
        assert result.fold.train_start > calendar[0]  # rolling window, not expanding from the very start


# --- nested inner-CV hyperparameter selection ----------------------------------------------


def test_default_hyperparameter_grid_covers_both_target_kinds():
    grid = default_hyperparameter_grid()
    assert len(grid) >= 2
    for candidate in grid:
        assert set(candidate.keys()) == {"regression", "classification"}


def test_run_nested_purged_walk_forward_never_lets_inner_folds_touch_outer_validation():
    df = _synthetic_feature_target_frame()
    calendar = build_trading_calendar(df["timestamp"])
    outer_folds = generate_purged_folds(calendar, initial_train_days=600, validation_days=100, window_mode="expanding")
    assert len(outer_folds) >= 1

    small_grid = default_hyperparameter_grid(num_leaves_options=(7, 15), learning_rate_options=(0.05,))
    results = run_nested_purged_walk_forward(
        df, outer_folds[:1], hyperparameter_candidates=small_grid, feature_cols=["f1", "f2"],
        n_inner_folds=2, inner_validation_days=60,
    )
    assert len(results) == 1
    result = results[0]
    assert result.selected_hyperparameters in small_grid
    assert len(result.inner_cv_scores) == len(small_grid)
    assert result.n_inner_folds_used >= 1
    # Every prediction in the final, single evaluation must fall inside the
    # (untouched-until-now) outer validation window.
    if not result.predictions.empty:
        assert (result.predictions["timestamp"] >= result.fold.validation_start).all()
        assert (result.predictions["timestamp"] <= result.fold.validation_end).all()
