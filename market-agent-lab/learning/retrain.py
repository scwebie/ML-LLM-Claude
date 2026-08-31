"""Controlled retraining pipeline (Phase 10).

Workflow: predictions -> wait for horizon to complete -> label outcomes
(``learning/outcomes.py``) -> append to the training set -> train a
CHALLENGER model -> walk-forward evaluate it -> hand off to
``learning/champion_challenger.py`` for the promote/reject decision.

There is no online/uncontrolled self-modification anywhere in this
pipeline: a challenger is just another row in the model registry until a
human-configured promotion rule (see ``champion_challenger.py``)
explicitly promotes it.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from models.evaluate import evaluate_classification, evaluate_regression
from models.predict import predict_batch
from models.registry import ModelPeriods, register_model
from models.train import TARGET_KIND, get_feature_columns, prepare_training_frame, train_all_targets


def train_challenger(
    con: duckdb.DuckDBPyConnection,
    feature_matrix: pd.DataFrame,
    targets: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    feature_version: str,
    feature_cols: list[str] | None = None,
) -> tuple[str, dict]:
    """Train a new CHALLENGER model on data up to ``validation_end`` and register it."""
    df = prepare_training_frame(feature_matrix, targets)
    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= validation_end)]
    if train_df.empty or val_df.empty:
        raise ValueError("insufficient data to train a challenger for the requested periods")

    feature_cols = feature_cols or get_feature_columns(df)
    trained = train_all_targets(train_df, val_df, feature_cols)

    val_predictions = predict_batch(trained.boosters, val_df, feature_cols, "challenger_pending", feature_version)
    pred_df = pd.DataFrame([p.model_dump() for p in val_predictions])
    metrics: dict = {}
    if not pred_df.empty:
        merged = pred_df.merge(val_df[["symbol", "timestamp", *TARGET_KIND.keys()]], on=["symbol", "timestamp"])
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
            metrics[target_col] = (
                evaluate_regression(sub[target_col], sub[pred_col])
                if kind == "regression"
                else evaluate_classification(sub[target_col], sub[pred_col])
            )

    periods = ModelPeriods(
        training_start=train_df["timestamp"].min(),
        training_end=train_df["timestamp"].max(),
        validation_start=val_df["timestamp"].min(),
        validation_end=val_df["timestamp"].max(),
    )
    model_version = register_model(con, trained, feature_version, periods, metrics, role="CHALLENGER")
    return model_version, metrics
