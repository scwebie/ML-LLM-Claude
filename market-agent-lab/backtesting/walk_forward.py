"""Expanding-window walk-forward validation.

Version 0.1 NEVER uses a random train/test split for the primary
financial evaluation -- see the project brief. Every fold expands the
training window forward in time and validates on a subsequent,
non-overlapping period, e.g.:

    fold 0: train 2015-2020, validate 2021
    fold 1: train 2015-2021, validate 2022
    fold 2: train 2015-2022, validate 2023

Dates are fully configurable via :func:`generate_expanding_folds`.
:func:`run_walk_forward` enforces, as a hard assertion (not just a
convention), that every fold's training window ends strictly before its
validation window begins.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from models.evaluate import evaluate_classification, evaluate_regression
from models.predict import predict_batch
from models.train import TARGET_KIND, TrainedModels, get_feature_columns, train_all_targets


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.train_end >= self.validation_start:
            raise ValueError(
                f"fold {self.fold_id}: train_end ({self.train_end}) must be strictly "
                f"before validation_start ({self.validation_start}) -- overlapping "
                "windows would leak validation-period information into training."
            )


def generate_expanding_folds(
    data_start: str | pd.Timestamp,
    initial_train_end: str | pd.Timestamp,
    overall_end: str | pd.Timestamp,
    validation_years: int = 1,
    step_years: int | None = None,
) -> list[WalkForwardFold]:
    """Generate expanding-window folds. All dates are inclusive."""
    step_years = step_years or validation_years
    data_start = pd.Timestamp(data_start)
    overall_end = pd.Timestamp(overall_end)
    train_end = pd.Timestamp(initial_train_end)

    folds: list[WalkForwardFold] = []
    fold_id = 0
    while True:
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.DateOffset(years=validation_years) - pd.Timedelta(days=1)
        if validation_end > overall_end:
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=data_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
        fold_id += 1
        train_end = train_end + pd.DateOffset(years=step_years)
    return folds


@dataclass
class FoldResult:
    fold: WalkForwardFold
    trained: TrainedModels
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame


def run_walk_forward(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    feature_cols: list[str] | None = None,
    model_version_prefix: str = "wf",
    feature_version: str = "fv1",
) -> list[FoldResult]:
    """Train + evaluate one model per fold.

    ``df`` must already be the joined feature+target frame (see
    ``models.train.prepare_training_frame``), containing ``symbol``,
    ``timestamp``, every feature column, and the four target columns.
    """
    feature_cols = feature_cols or get_feature_columns(df)
    results: list[FoldResult] = []

    for fold in folds:
        train_mask = (df["timestamp"] >= fold.train_start) & (df["timestamp"] <= fold.train_end)
        val_mask = (df["timestamp"] >= fold.validation_start) & (df["timestamp"] <= fold.validation_end)
        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]

        # Hard runtime leakage guard, independent of the dataclass check
        # above (which only validates the *configured* boundaries): assert
        # no timestamp actually used for training falls on/after any
        # timestamp actually used for validation.
        if len(train_df) and len(val_df):
            assert train_df["timestamp"].max() < val_df["timestamp"].min(), (
                f"fold {fold.fold_id}: leakage detected -- a training row's timestamp "
                "is not strictly before the earliest validation timestamp"
            )

        if train_df.empty or val_df.empty:
            continue

        trained = train_all_targets(train_df, val_df, feature_cols)
        model_version = f"{model_version_prefix}_fold{fold.fold_id}"
        preds = predict_batch(trained.boosters, val_df, feature_cols, model_version, feature_version)
        pred_df = pd.DataFrame([p.model_dump() for p in preds])

        metrics: dict[str, dict[str, float]] = {}
        if not pred_df.empty:
            merged = pred_df.merge(
                val_df[["symbol", "timestamp", *TARGET_KIND.keys()]], on=["symbol", "timestamp"]
            )
            for target_col, kind in TARGET_KIND.items():
                sub = merged.dropna(subset=[target_col])
                if sub.empty:
                    continue
                pred_col = {
                    "excess_return_5d": "predicted_excess_return_5d",
                    "excess_return_20d": "predicted_excess_return_20d",
                    "positive_5d": "probability_positive_5d",
                    "positive_20d": "probability_positive_20d",
                }[target_col]
                if kind == "regression":
                    metrics[target_col] = evaluate_regression(sub[target_col], sub[pred_col])
                else:
                    metrics[target_col] = evaluate_classification(sub[target_col], sub[pred_col])

        results.append(FoldResult(fold=fold, trained=trained, metrics=metrics, predictions=pred_df))

    return results
