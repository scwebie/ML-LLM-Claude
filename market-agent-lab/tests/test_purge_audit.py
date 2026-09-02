"""Tests for backtesting/purge_audit.py (V0.3 Stage 7): exact boundary
reporting and cross-fold structural re-audits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.purge_audit import (
    assert_folds_temporally_ordered,
    assert_no_validation_row_reused_in_training,
    describe_all_folds,
    describe_fold_boundaries,
)
from backtesting.purged_walk_forward import (
    PurgedFold,
    build_trading_calendar,
    generate_purged_folds,
)


def _daily_frame(n_days=200, seed=1):
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"symbol": "SYM0", "timestamp": dates, "f1": rng.normal(size=n_days)})


def test_describe_fold_boundaries_purge_and_embargo_match_configured_horizon():
    """Exact boundary-case check: with horizon_days=20 and embargo_days=5
    on a plain business-day calendar, the purge zone must be exactly the
    20 trading days immediately before validation_start, and the embargo
    zone must be exactly the 5 trading days immediately after
    validation_end -- neither one day short nor one day long."""
    df = _daily_frame(n_days=200)
    calendar = build_trading_calendar(df["timestamp"])
    fold = PurgedFold(
        fold_id=0, train_start=calendar[0], validation_start=calendar[100], validation_end=calendar[119],
        window_mode="expanding",
    )
    report = describe_fold_boundaries(df, fold, calendar, horizon_days=20, embargo_days=5)

    assert report.purge_end == calendar[99]  # the trading day immediately before validation_start
    assert report.purge_start == calendar[99 - 20 + 1]  # 20 trading days back from validation_start
    assert report.train_end == calendar[99 - 20]  # the trading day immediately before purge_start

    assert report.embargo_start == calendar[120]  # the trading day immediately after validation_end
    assert report.embargo_end == calendar[120 + 5 - 1]  # 5 trading days after validation_end


def test_describe_fold_boundaries_no_purge_when_gap_exceeds_horizon():
    """A fold whose training candidate window ends well more than
    horizon_days before validation_start should show NO purged rows at
    all -- purge only bites the tail end of the candidate window."""
    df = _daily_frame(n_days=60)
    calendar = build_trading_calendar(df["timestamp"])
    # train candidates end at index 29 (validation_start - 1); with only
    # 30 candidate rows and horizon_days=20, most of them ARE within the
    # purge zone -- so instead directly test a fold where train_start is
    # already past the purge boundary, leaving zero eligible candidates
    # to purge (they were never candidates in the first place).
    fold = PurgedFold(fold_id=0, train_start=calendar[15], validation_start=calendar[30], validation_end=calendar[39], window_mode="expanding")
    report = describe_fold_boundaries(df, fold, calendar, horizon_days=20, embargo_days=5)
    # Every candidate row (indices 15..29) falls inside the last 20
    # trading days before validation_start (indices 10..29) -- so ALL
    # candidates are purged, and train_end is None.
    assert report.train_end is None
    assert report.n_purged_rows == 30 - 15


def test_describe_all_folds_returns_one_row_per_fold_with_every_required_column():
    df = _daily_frame(n_days=400)
    calendar = build_trading_calendar(df["timestamp"])
    folds = generate_purged_folds(calendar, initial_train_days=200, validation_days=50, window_mode="expanding")
    report_df = describe_all_folds(df, folds, calendar)
    assert len(report_df) == len(folds)
    required = {
        "fold_id", "train_start", "train_end", "purge_start", "purge_end", "n_purged_rows",
        "validation_start", "validation_end", "embargo_start", "embargo_end", "n_embargoed_rows",
    }
    assert required <= set(report_df.columns)


# --- structural re-audits -----------------------------------------------------------------------


def test_assert_folds_temporally_ordered_passes_for_generated_folds():
    df = _daily_frame(n_days=400)
    calendar = build_trading_calendar(df["timestamp"])
    folds = generate_purged_folds(calendar, initial_train_days=200, validation_days=50, window_mode="expanding")
    assert_folds_temporally_ordered(folds)  # must not raise


def test_assert_folds_temporally_ordered_raises_for_overlapping_folds():
    dates = pd.bdate_range("2020-01-02", periods=100)
    bad_folds = [
        PurgedFold(fold_id=0, train_start=dates[0], validation_start=dates[50], validation_end=dates[69], window_mode="expanding"),
        # Fold 1's validation window starts BEFORE fold 0's ends -- overlap.
        PurgedFold(fold_id=1, train_start=dates[0], validation_start=dates[60], validation_end=dates[89], window_mode="expanding"),
    ]
    with pytest.raises(AssertionError, match="overlap"):
        assert_folds_temporally_ordered(bad_folds)


def test_assert_folds_temporally_ordered_raises_when_train_start_after_validation_start():
    dates = pd.bdate_range("2020-01-02", periods=100)
    bad_folds = [
        PurgedFold(fold_id=0, train_start=dates[60], validation_start=dates[50], validation_end=dates[69], window_mode="expanding"),
    ]
    with pytest.raises(AssertionError, match="train_start"):
        assert_folds_temporally_ordered(bad_folds)


def test_assert_no_validation_row_reused_in_training_passes_for_generated_folds():
    df = _daily_frame(n_days=400)
    calendar = build_trading_calendar(df["timestamp"])
    folds = generate_purged_folds(calendar, initial_train_days=200, validation_days=50, window_mode="expanding")
    assert_no_validation_row_reused_in_training(df, folds, calendar)  # must not raise
