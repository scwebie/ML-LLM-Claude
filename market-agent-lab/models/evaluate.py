"""Model evaluation metrics for the four alpha-model targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman rank correlation between predicted and realised values --
    the standard "IC" metric for cross-sectional alpha models."""
    if len(y_true) < 3:
        return float("nan")
    return float(pd.Series(y_pred).rank().corr(pd.Series(y_true).rank(), method="pearson"))


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "information_coefficient": information_coefficient(pd.Series(y_true), pd.Series(y_pred)),
    }


def evaluate_classification(y_true: pd.Series, y_pred_proba: pd.Series) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred_proba = np.clip(np.asarray(y_pred_proba, dtype=float), 1e-6, 1 - 1e-6)
    metrics: dict[str, float] = {}
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_pred_proba)) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    metrics["accuracy"] = float(accuracy_score(y_true, (y_pred_proba > 0.5).astype(float)))
    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba, labels=[0.0, 1.0]))
    metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
    return metrics


def calibration_curve(y_true: pd.Series, y_pred_proba: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into ``n_bins`` and compare mean predicted vs. realised rate per bin."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_proba}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "mean_predicted", "mean_realised", "count"])
    df["bin"] = pd.qcut(df["y_pred"], q=min(n_bins, df["y_pred"].nunique()), duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("y_pred", "mean"), mean_realised=("y_true", "mean"), count=("y_true", "size")
    )
    return grouped.reset_index()


def feature_importance(booster, feature_names: list[str], importance_type: str = "gain") -> pd.DataFrame:
    importances = booster.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
