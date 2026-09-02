"""V0.3 Stage 4: feature-family ablation, DEVELOPMENT DATA ONLY.

Extends V0.2's ``backtesting/robustness.py::run_feature_ablation`` (kept
exactly as-is; its ``classify_feature_family``/``group_features_by_family``
taxonomy is reused verbatim here, not duplicated) with the fuller set of
statistics V0.3 requires per family: not just a single mean-IC delta, but
per-fold IC, rank IC, calibration, a genuine chronological development
portfolio Sharpe (``backtesting.daily_portfolio``, not the overlapping-
target one), turnover, max drawdown, and a block-bootstrap uncertainty
estimate on the per-date IC series -- so a family is never declared
"useful" or "harmful" off a single fold's number alone.

Every fold here comes from ``run_purged_walk_forward`` over
``development_df`` (pre-holdout only) -- nothing in this module ever
touches the fixed holdout or post-holdout regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.daily_portfolio import (
    build_daily_rebalanced_portfolio_returns,
    sharpe_audit_report,
)
from backtesting.development_diagnostics import pearson_ic
from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    TARGET_TO_PRED_COL,
    PurgedFold,
    PurgedFoldResult,
    run_purged_walk_forward,
)
from backtesting.robustness import (
    block_bootstrap_ci,
    build_evaluation_frame,
    classify_feature_family,
    group_features_by_family,
)
from models.evaluate import information_coefficient

__all__ = ["AblationFamilyReport", "run_feature_ablation_v3", "classify_feature_family", "group_features_by_family"]


@dataclass
class AblationFamilyReport:
    family: str
    variant: str  # "baseline" | "remove_<family>" | "only_<family>"
    n_features: int
    n_folds: int
    mean_rank_ic: float
    per_fold_rank_ic: list[float] = field(default_factory=list)
    pearson_ic: float = float("nan")
    rank_ic_bootstrap_ci: dict = field(default_factory=dict)
    mean_brier: float = float("nan")
    sharpe_audit: dict = field(default_factory=dict)
    delta_vs_baseline_rank_ic: float = float("nan")


def _fold_rank_ics(fold_results: list[PurgedFoldResult], target_col: str) -> list[float]:
    values = [r.metrics.get(target_col, {}).get("information_coefficient", float("nan")) for r in fold_results]
    return [v for v in values if v == v]


def _build_report(
    variant: str, family: str, feature_cols: list[str], fold_results: list[PurgedFoldResult],
    development_df: pd.DataFrame, market_df: pd.DataFrame, target_col: str, pred_col: str,
) -> AblationFamilyReport:
    per_fold_ic = _fold_rank_ics(fold_results, target_col)
    mean_ic = float(np.mean(per_fold_ic)) if per_fold_ic else float("nan")

    eval_frame = build_evaluation_frame(fold_results, development_df, target_col) if fold_results else pd.DataFrame()
    p_ic = pearson_ic(eval_frame[target_col], eval_frame[pred_col]) if not eval_frame.empty else float("nan")

    ci = {}
    if not eval_frame.empty:
        per_date_ic = eval_frame.groupby("timestamp").apply(
            lambda g: information_coefficient(g[target_col], g[pred_col]) if len(g) >= 3 else np.nan,
            include_groups=False,
        ).dropna()
        if len(per_date_ic) >= 5:
            ci = block_bootstrap_ci(per_date_ic.to_numpy(), block_size=min(10, max(2, len(per_date_ic) // 3)))

    pos_target = target_col.replace("excess_return_", "positive_")
    brier_values = [r.metrics.get(pos_target, {}).get("brier_score", float("nan")) for r in fold_results]
    brier_values = [v for v in brier_values if v == v]
    mean_brier = float(np.mean(brier_values)) if brier_values else float("nan")

    sharpe_audit = {}
    if fold_results and not market_df.empty:
        last_fold = fold_results[-1]
        daily_portfolio = build_daily_rebalanced_portfolio_returns(last_fold.predictions, market_df, pred_col)
        sharpe_audit = sharpe_audit_report(daily_portfolio)

    return AblationFamilyReport(
        family=family, variant=variant, n_features=len(feature_cols), n_folds=len(fold_results),
        mean_rank_ic=mean_ic, per_fold_rank_ic=per_fold_ic, pearson_ic=p_ic, rank_ic_bootstrap_ci=ci,
        mean_brier=mean_brier, sharpe_audit=sharpe_audit,
    )


def run_feature_ablation_v3(
    development_df: pd.DataFrame,
    folds: list[PurgedFold],
    feature_cols: list[str],
    market_df: pd.DataFrame,
    families: dict[str, list[str]] | None = None,
    target_col: str = "excess_return_20d",
    include_family_only: bool = False,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    hyperparameters: dict | None = None,
) -> list[AblationFamilyReport]:
    """Baseline (every feature) vs. remove-one-family vs. (optionally)
    family-only, for every feature family present in ``feature_cols``.
    Never trains or evaluates on anything but ``development_df`` (the
    caller's pre-holdout region) and its purged ``folds``."""
    families = families or group_features_by_family(feature_cols)
    pred_col = TARGET_TO_PRED_COL[target_col]
    reports: list[AblationFamilyReport] = []

    baseline_folds = run_purged_walk_forward(
        development_df, folds, feature_cols, horizon_days, embargo_days, hyperparameters,
        model_version_prefix="ablation_v3_baseline",
    )
    baseline_report = _build_report(
        "baseline", "baseline", feature_cols, baseline_folds, development_df, market_df, target_col, pred_col
    )
    reports.append(baseline_report)

    for family, cols_to_remove in families.items():
        remaining = [c for c in feature_cols if c not in cols_to_remove]
        if not remaining:
            continue
        removed_folds = run_purged_walk_forward(
            development_df, folds, remaining, horizon_days, embargo_days, hyperparameters,
            model_version_prefix=f"ablation_v3_remove_{family}",
        )
        removed_report = _build_report(
            f"remove_{family}", family, remaining, removed_folds, development_df, market_df, target_col, pred_col
        )
        removed_report.delta_vs_baseline_rank_ic = baseline_report.mean_rank_ic - removed_report.mean_rank_ic
        reports.append(removed_report)

        if include_family_only and cols_to_remove:
            only_folds = run_purged_walk_forward(
                development_df, folds, cols_to_remove, horizon_days, embargo_days, hyperparameters,
                model_version_prefix=f"ablation_v3_only_{family}",
            )
            only_report = _build_report(
                f"only_{family}", family, cols_to_remove, only_folds, development_df, market_df, target_col, pred_col
            )
            reports.append(only_report)

    return reports
