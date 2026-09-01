"""Final untouched holdout period (Stage 12, V0.2).

A configurable date range (``core/config.py::Settings.holdout_start_date``/
``holdout_end_date``, fixed once, in advance, and never moved after seeing
results) that is never used for model selection, hyperparameter tuning,
feature engineering iteration, or walk-forward fold boundaries.

:func:`split_development_and_holdout` partitions a feature+target frame
into ``(development_df, holdout_df)``. Every walk-forward run, ablation,
and robustness check in this project (Stages 11, 13, 14) operates ONLY on
``development_df``. :func:`evaluate_on_holdout` is the single, deliberately
narrow entry point that may ever touch ``holdout_df``: it trains nothing
itself, it only scores an already-selected, already-trained model, and it
writes an audit-trail row (``core/schemas_v2.py::HoldoutAccessLog``, table
``holdout_access_log``) to the database on every single call -- there is
no silent-mode flag. A reviewer can query that table and confirm the
holdout was accessed exactly as many times as models were formally
evaluated, never more.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import pandas as pd

from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    TARGET_TO_PRED_COL,
    PurgedFold,
    build_trading_calendar,
    compute_purge_embargo_mask,
)
from core.config import settings
from core.schemas_v2 import HoldoutAccessLog
from database import repository_v2 as repo_v2
from models.evaluate import evaluate_classification, evaluate_regression
from models.predict import predict_batch
from models.train import TARGET_KIND, TrainedModels


@dataclass(frozen=True)
class HoldoutConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError(f"holdout start_date ({self.start_date}) must be before end_date ({self.end_date})")


def default_holdout_config() -> HoldoutConfig:
    """The project's fixed holdout window, read from ``Settings`` (which in
    turn reads ``HOLDOUT_START_DATE``/``HOLDOUT_END_DATE`` env vars, or a
    documented default)."""
    return HoldoutConfig(
        start_date=pd.Timestamp(settings.holdout_start_date), end_date=pd.Timestamp(settings.holdout_end_date)
    )


def split_development_and_holdout(
    df: pd.DataFrame,
    holdout: HoldoutConfig | None = None,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition ``df`` into ``(development_df, holdout_df)``.

    ``development_df`` is purged+embargoed against the holdout window
    exactly like an outer walk-forward fold's training window is purged
    against its validation window (see
    ``backtesting/purged_walk_forward.py``): a development row whose
    target-realization window reaches into the holdout period is
    excluded, and rows within ``embargo_days`` trading days of either
    edge of the holdout window are dropped from development too.

    ``holdout_df`` is untouched -- exactly the rows in
    ``[holdout.start_date, holdout.end_date]``, with no purge/embargo
    applied (there is nothing "downstream" of it to protect).
    """
    holdout = holdout or default_holdout_config()
    calendar = build_trading_calendar(df["timestamp"])

    holdout_mask = (df["timestamp"] >= holdout.start_date) & (df["timestamp"] <= holdout.end_date)
    holdout_df = df.loc[holdout_mask].sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    eligible = compute_purge_embargo_mask(
        df["timestamp"], calendar, holdout.start_date, holdout.end_date, horizon_days, embargo_days
    )
    development_mask = (~holdout_mask) & eligible
    development_df = df.loc[development_mask].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return development_df, holdout_df


def assert_no_fold_touches_holdout(folds: list[PurgedFold], holdout: HoldoutConfig | None = None) -> None:
    """Defensive wiring check: raises if any walk-forward fold's
    validation window overlaps the holdout period at all. Intended to be
    called wherever outer folds are generated for real-data evaluation
    (e.g. the ``evaluate-real`` CLI command, Stage 15), so a future
    configuration mistake that widens the development window into the
    holdout can never pass silently."""
    holdout = holdout or default_holdout_config()
    for fold in folds:
        overlap = fold.validation_start <= holdout.end_date and fold.validation_end >= holdout.start_date
        if overlap:
            raise ValueError(
                f"fold {fold.fold_id}: validation window [{fold.validation_start}, {fold.validation_end}] "
                f"overlaps the holdout period [{holdout.start_date}, {holdout.end_date}] -- "
                "the holdout must never be used for model selection"
            )


@dataclass
class HoldoutEvaluationResult:
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    n_rows: int
    log_entry: HoldoutAccessLog


def evaluate_on_holdout(
    con: duckdb.DuckDBPyConnection,
    trained: TrainedModels,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
    model_version: str,
    purpose: str,
    feature_version: str = "fv1",
) -> HoldoutEvaluationResult:
    """The ONLY function in this project permitted to score a model on
    holdout-period rows. Trains nothing -- ``trained`` must already be a
    fully fit model (selected and validated entirely on
    ``development_df``). Every call is logged, unconditionally, to
    ``holdout_access_log``."""
    if holdout_df.empty:
        raise ValueError("evaluate_on_holdout called with an empty holdout_df -- nothing to evaluate")

    preds = predict_batch(trained.boosters, holdout_df, feature_cols, model_version, feature_version)
    pred_df = pd.DataFrame([p.model_dump() for p in preds])

    metrics: dict[str, dict[str, float]] = {}
    if not pred_df.empty:
        merged = pred_df.merge(holdout_df[["symbol", "timestamp", *TARGET_KIND.keys()]], on=["symbol", "timestamp"])
        for target_col, kind in TARGET_KIND.items():
            sub = merged.dropna(subset=[target_col])
            if sub.empty:
                continue
            pred_col = TARGET_TO_PRED_COL[target_col]
            if kind == "regression":
                metrics[target_col] = evaluate_regression(sub[target_col], sub[pred_col])
            else:
                metrics[target_col] = evaluate_classification(sub[target_col], sub[pred_col])

    log_entry = HoldoutAccessLog(
        accessed_at=datetime.now(UTC),
        purpose=purpose,
        model_version=model_version,
        holdout_start=holdout_df["timestamp"].min(),
        holdout_end=holdout_df["timestamp"].max(),
        n_rows=len(holdout_df),
        symbols=sorted(holdout_df["symbol"].unique().tolist()),
    )
    repo_v2.insert_holdout_access_log(con, log_entry)

    return HoldoutEvaluationResult(metrics=metrics, predictions=pred_df, n_rows=len(holdout_df), log_entry=log_entry)
