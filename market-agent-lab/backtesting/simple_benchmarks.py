"""V0.3 Stage 11: benchmark LightGBM against simple baselines, on the
SAME purged folds and features (where applicable), development data only.

The purpose is not to find a "better" model -- it's to determine whether
LightGBM's added complexity earns its keep at all, by comparing its
out-of-sample rank IC against: ridge regression (continuous targets),
logistic regression (direction), a simple momentum factor, a simple
mean-reversion factor, and an equal-weight composite of the two factors.
The factor baselines are RULE-BASED (no fitting at all) -- a real
existing feature column used directly as the ranking signal -- exactly
what a simple, well-known, non-ML trading signal actually is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from backtesting.purged_walk_forward import (
    PurgedFold,
    build_outer_train_eligibility,
    build_trading_calendar,
)
from models.evaluate import information_coefficient


@dataclass
class BenchmarkFoldResult:
    fold_id: int
    model: str
    rank_ic: float
    n_obs: int


@dataclass
class BenchmarkComparisonReport:
    target_col: str
    per_fold: list[BenchmarkFoldResult] = field(default_factory=list)
    mean_ic_by_model: dict[str, float] = field(default_factory=dict)
    incremental_ic_over_best_baseline: float = float("nan")
    best_baseline: str = ""


def _fold_train_val(df: pd.DataFrame, fold: PurgedFold, calendar, horizon_days: int, embargo_days: int):
    candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar, horizon_days, embargo_days)
    train_mask = candidate_mask & eligible_mask
    val_mask = (df["timestamp"] >= fold.validation_start) & (df["timestamp"] <= fold.validation_end)
    return df.loc[train_mask], df.loc[val_mask]


def ridge_baseline_ic(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], target_col: str, alpha: float = 1.0) -> tuple[float, int]:
    train_clean = train_df.dropna(subset=[*feature_cols, target_col])
    val_clean = val_df.dropna(subset=[*feature_cols, target_col])
    if len(train_clean) < 20 or val_clean.empty:
        return float("nan"), 0
    model = Ridge(alpha=alpha)
    model.fit(train_clean[feature_cols], train_clean[target_col])
    preds = model.predict(val_clean[feature_cols])
    return information_coefficient(val_clean[target_col], pd.Series(preds, index=val_clean.index)), len(val_clean)


def logistic_baseline_ic(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[float, int]:
    """``target_col`` here is a binary 0/1 direction target
    (``positive_5d``/``positive_20d``); rank IC is computed against the
    predicted probability, same convention as the LightGBM classifier
    heads."""
    train_clean = train_df.dropna(subset=[*feature_cols, target_col])
    val_clean = val_df.dropna(subset=[*feature_cols, target_col])
    if len(train_clean) < 20 or val_clean.empty or train_clean[target_col].nunique() < 2:
        return float("nan"), 0
    model = LogisticRegression(max_iter=1000)
    model.fit(train_clean[feature_cols], train_clean[target_col])
    proba = model.predict_proba(val_clean[feature_cols])[:, 1]
    return information_coefficient(val_clean[target_col], pd.Series(proba, index=val_clean.index)), len(val_clean)


def factor_signal_ic(val_df: pd.DataFrame, factor_col: str, target_col: str, invert: bool = False) -> tuple[float, int]:
    """No fitting -- ranks directly by an existing raw feature column
    (e.g. ``raw_return_60d`` for momentum, ``raw_rsi_14`` for mean-
    reversion with ``invert=True`` since a HIGH RSI predicts a pullback,
    i.e. a NEGATIVE forward return)."""
    clean = val_df.dropna(subset=[factor_col, target_col])
    if clean.empty:
        return float("nan"), 0
    signal = -clean[factor_col] if invert else clean[factor_col]
    return information_coefficient(clean[target_col], signal), len(clean)


def equal_weight_composite_ic(val_df: pd.DataFrame, factor_cols: list[str], target_col: str, invert: list[bool] | None = None) -> tuple[float, int]:
    """Z-scores each factor cross-sectionally per date, then averages --
    the simplest possible multi-factor composite, no fitting."""
    invert = invert or [False] * len(factor_cols)
    clean = val_df.dropna(subset=[*factor_cols, target_col])
    if clean.empty:
        return float("nan"), 0
    z_scores = []
    for col, inv in zip(factor_cols, invert, strict=True):
        by_date = clean.groupby("timestamp")[col]
        z = (clean[col] - by_date.transform("mean")) / by_date.transform("std").replace(0, np.nan)
        z_scores.append(-z if inv else z)
    composite = pd.concat(z_scores, axis=1).mean(axis=1)
    return information_coefficient(clean[target_col], composite), len(clean)


def run_simple_benchmarks(
    development_df: pd.DataFrame,
    folds: list[PurgedFold],
    feature_cols: list[str],
    target_col: str = "excess_return_20d",
    momentum_col: str = "raw_return_60d",
    reversion_col: str = "raw_rsi_14",
    horizon_days: int = 20,
    embargo_days: int = 5,
    lightgbm_per_fold_ic: list[float] | None = None,
) -> BenchmarkComparisonReport:
    """Runs every baseline on the SAME per-fold train/validation split
    purge/embargo produces for LightGBM (``build_outer_train_eligibility``).
    Pass ``lightgbm_per_fold_ic`` (the same per-fold rank ICs already
    computed by ``run_purged_walk_forward``/``PurgedFoldResult.metrics``)
    to include LightGBM in the comparison and get the incremental-IC
    verdict; omit it to only compare the baselines against each other."""
    calendar = build_trading_calendar(development_df["timestamp"])
    pos_target = target_col.replace("excess_return_", "positive_")
    has_factor_cols = momentum_col in development_df.columns and reversion_col in development_df.columns

    report = BenchmarkComparisonReport(target_col=target_col)
    model_ics: dict[str, list[float]] = {"ridge": [], "logistic": [], "momentum": [], "mean_reversion": [], "equal_weight_composite": []}

    for fold in folds:
        train_df, val_df = _fold_train_val(development_df, fold, calendar, horizon_days, embargo_days)

        ridge_ic, ridge_n = ridge_baseline_ic(train_df, val_df, feature_cols, target_col)
        report.per_fold.append(BenchmarkFoldResult(fold.fold_id, "ridge", ridge_ic, ridge_n))
        if ridge_ic == ridge_ic:
            model_ics["ridge"].append(ridge_ic)

        if pos_target in development_df.columns:
            logit_ic, logit_n = logistic_baseline_ic(train_df, val_df, feature_cols, pos_target)
            report.per_fold.append(BenchmarkFoldResult(fold.fold_id, "logistic", logit_ic, logit_n))
            if logit_ic == logit_ic:
                model_ics["logistic"].append(logit_ic)

        if has_factor_cols:
            mom_ic, mom_n = factor_signal_ic(val_df, momentum_col, target_col, invert=False)
            report.per_fold.append(BenchmarkFoldResult(fold.fold_id, "momentum", mom_ic, mom_n))
            if mom_ic == mom_ic:
                model_ics["momentum"].append(mom_ic)

            rev_ic, rev_n = factor_signal_ic(val_df, reversion_col, target_col, invert=True)
            report.per_fold.append(BenchmarkFoldResult(fold.fold_id, "mean_reversion", rev_ic, rev_n))
            if rev_ic == rev_ic:
                model_ics["mean_reversion"].append(rev_ic)

            combo_ic, combo_n = equal_weight_composite_ic(val_df, [momentum_col, reversion_col], target_col, invert=[False, True])
            report.per_fold.append(BenchmarkFoldResult(fold.fold_id, "equal_weight_composite", combo_ic, combo_n))
            if combo_ic == combo_ic:
                model_ics["equal_weight_composite"].append(combo_ic)

    if lightgbm_per_fold_ic:
        model_ics["lightgbm"] = [v for v in lightgbm_per_fold_ic if v == v]

    report.mean_ic_by_model = {name: float(np.mean(values)) for name, values in model_ics.items() if values}

    baseline_means = {k: v for k, v in report.mean_ic_by_model.items() if k != "lightgbm"}
    if baseline_means:
        report.best_baseline = max(baseline_means, key=baseline_means.get)
        if "lightgbm" in report.mean_ic_by_model:
            report.incremental_ic_over_best_baseline = report.mean_ic_by_model["lightgbm"] - baseline_means[report.best_baseline]

    return report
