"""V0.3 Stage 7: purging/overlap re-audit.

The purge/embargo mechanism itself (``backtesting/purged_walk_forward.py``)
is unchanged -- it already has extensive boundary-case coverage in
``tests/test_purged_walk_forward.py`` (off-by-one tests at the exact
purge and embargo edges). This module adds what V0.3 additionally asks
for: an explicit, printable per-fold boundary report (train/purge/
validation/embargo date ranges) computed directly from the SAME masks
``run_purged_walk_forward`` actually trains on (not re-derived
analytically, so this can never silently drift out of sync with the real
eligibility logic), plus two structural re-audits across a full fold
list: that folds stay temporally ordered, and that no fold's own
validation rows are ever present in its own training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    PurgedFold,
    _trading_day_offset,
    build_outer_train_eligibility,
)


@dataclass
class FoldBoundaryReport:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp | None  # last date actually eligible for training, after purge+embargo
    purge_start: pd.Timestamp | None
    purge_end: pd.Timestamp | None
    n_purged_rows: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    n_embargoed_rows: int


def describe_fold_boundaries(
    df: pd.DataFrame,
    fold: PurgedFold,
    calendar: pd.DatetimeIndex,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> FoldBoundaryReport:
    """Computed directly from ``build_outer_train_eligibility``'s own
    masks -- the same function ``run_purged_walk_forward`` calls -- so
    this report can never describe a different boundary than what was
    actually trained on."""
    candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar, horizon_days, embargo_days)
    train_mask = candidate_mask & eligible_mask
    purged_mask = candidate_mask & ~eligible_mask
    ts = df["timestamp"]

    train_end = ts[train_mask].max() if train_mask.any() else None
    purge_start = ts[purged_mask].min() if purged_mask.any() else None
    purge_end = ts[purged_mask].max() if purged_mask.any() else None

    embargo_start = _trading_day_offset(fold.validation_end, calendar, 1)
    embargo_end = _trading_day_offset(fold.validation_end, calendar, embargo_days)
    embargo_mask = (ts > fold.validation_end) & (ts <= embargo_end)

    return FoldBoundaryReport(
        fold_id=fold.fold_id, train_start=fold.train_start, train_end=train_end,
        purge_start=purge_start, purge_end=purge_end, n_purged_rows=int(purged_mask.sum()),
        validation_start=fold.validation_start, validation_end=fold.validation_end,
        embargo_start=embargo_start, embargo_end=embargo_end, n_embargoed_rows=int(embargo_mask.sum()),
    )


def describe_all_folds(
    df: pd.DataFrame,
    folds: list[PurgedFold],
    calendar: pd.DatetimeIndex,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> pd.DataFrame:
    """One printable row per fold with every boundary V0.3 Stage 7
    requires: train start/end, purge start/end, validation start/end,
    embargo start/end."""
    rows = [describe_fold_boundaries(df, f, calendar, horizon_days, embargo_days).__dict__ for f in folds]
    return pd.DataFrame(rows)


def assert_folds_temporally_ordered(folds: list[PurgedFold]) -> None:
    """Every fold's train_start must not exceed its own validation_start,
    and successive folds' validation windows must be strictly increasing
    and non-overlapping -- a walk-forward evaluator that ever produced
    folds out of chronological order would silently validate on data
    older than a later fold's training window."""
    for fold in folds:
        assert fold.train_start <= fold.validation_start, (
            f"fold {fold.fold_id}: train_start ({fold.train_start}) is after its own validation_start "
            f"({fold.validation_start})"
        )
        assert fold.validation_start <= fold.validation_end, (
            f"fold {fold.fold_id}: validation_start ({fold.validation_start}) is after validation_end "
            f"({fold.validation_end})"
        )
    for prev, cur in zip(folds, folds[1:], strict=False):
        assert cur.validation_start > prev.validation_end, (
            f"folds {prev.fold_id} and {cur.fold_id}: validation windows overlap or are out of order "
            f"([{prev.validation_start}, {prev.validation_end}] then [{cur.validation_start}, {cur.validation_end}])"
        )


def assert_no_validation_row_reused_in_training(
    df: pd.DataFrame,
    folds: list[PurgedFold],
    calendar: pd.DatetimeIndex,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> None:
    """For every fold, no row inside that fold's OWN validation window may
    ever also be marked eligible for that fold's OWN training set."""
    for fold in folds:
        candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar, horizon_days, embargo_days)
        train_mask = candidate_mask & eligible_mask
        val_mask = (df["timestamp"] >= fold.validation_start) & (df["timestamp"] <= fold.validation_end)
        overlap = train_mask & val_mask
        assert not overlap.any(), (
            f"fold {fold.fold_id}: {int(overlap.sum())} row(s) inside its own validation window "
            f"[{fold.validation_start}, {fold.validation_end}] were also marked eligible for its own training set"
        )
