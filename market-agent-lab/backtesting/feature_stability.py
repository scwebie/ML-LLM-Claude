"""V0.3 Stage 5: feature-importance stability across DEVELOPMENT folds.

Measures feature importance independently on each purged walk-forward
fold (never touching the holdout) using two independent methods --
LightGBM's native "gain" importance (``models.evaluate.feature_importance``,
unchanged) and permutation importance (shuffle one column at a time in
that fold's own validation set, measure the drop in rank IC) -- and
reports where they agree or disagree: top features per fold, cross-fold
rank correlation, features whose importance changes sign or swings
dramatically, features that only ever mattered in one fold, and
feature-family-level stability. Nothing here removes a feature
automatically; it only reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.purged_walk_forward import PurgedFoldResult
from backtesting.robustness import classify_feature_family
from models.evaluate import feature_importance, information_coefficient


def permutation_importance_for_fold(
    trained, val_df: pd.DataFrame, feature_cols: list[str], target_col: str, n_repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Shuffles each feature column ``n_repeats`` times in ``val_df`` (that
    fold's own out-of-sample rows) and measures the drop in rank IC
    relative to the unshuffled baseline -- a feature the model actually
    relies on shows a positive drop; a feature it ignores (or that is
    actively counterproductive) shows ~zero or negative importance."""
    booster = trained.boosters.get(target_col)
    if booster is None or val_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    clean = val_df.dropna(subset=[*feature_cols, target_col])
    if len(clean) < 10:
        return pd.DataFrame(columns=["feature", "importance"])

    x = clean[feature_cols].astype(float).reset_index(drop=True)
    y = clean[target_col].astype(float).reset_index(drop=True)
    baseline_score = information_coefficient(y, pd.Series(booster.predict(x)))

    rng = np.random.default_rng(seed)
    rows = []
    for col in feature_cols:
        drops = []
        for _ in range(n_repeats):
            x_shuffled = x.copy()
            x_shuffled[col] = rng.permutation(x_shuffled[col].to_numpy())
            shuffled_score = information_coefficient(y, pd.Series(booster.predict(x_shuffled)))
            drop = baseline_score - shuffled_score
            drops.append(drop if drop == drop else 0.0)
        rows.append({"feature": col, "importance": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def _mean_pairwise_spearman(importance_series: list[pd.Series]) -> float:
    if len(importance_series) < 2:
        return float("nan")
    combined = pd.concat(importance_series, axis=1, keys=range(len(importance_series))).fillna(0.0)
    corr = combined.corr(method="spearman")
    n = len(corr.columns)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    values = [corr.iloc[i, j] for i, j in pairs]
    return float(np.mean(values)) if values else float("nan")


@dataclass
class FeatureStabilityReport:
    n_folds: int
    top_features_native_per_fold: list[list[str]] = field(default_factory=list)
    top_features_permutation_per_fold: list[list[str]] = field(default_factory=list)
    native_rank_correlation: float = float("nan")
    permutation_rank_correlation: float = float("nan")
    sign_flip_features: list[str] = field(default_factory=list)
    one_period_only_features: list[str] = field(default_factory=list)
    family_stability: dict = field(default_factory=dict)
    stability_score: float = float("nan")


def compute_feature_stability(
    fold_results: list[PurgedFoldResult],
    development_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "excess_return_20d",
    top_k: int = 10,
    n_permutation_repeats: int = 5,
) -> FeatureStabilityReport:
    if len(fold_results) < 2:
        return FeatureStabilityReport(n_folds=len(fold_results))

    native_series: list[pd.Series] = []
    permutation_series: list[pd.Series] = []
    top_native: list[list[str]] = []
    top_permutation: list[list[str]] = []

    for result in fold_results:
        booster = result.trained.boosters.get(target_col)
        if booster is None:
            continue
        native_df = feature_importance(booster, result.trained.feature_names)
        native_series.append(native_df.set_index("feature")["importance"])
        top_native.append(list(native_df.head(top_k)["feature"]))

        val_mask = (development_df["timestamp"] >= result.fold.validation_start) & (
            development_df["timestamp"] <= result.fold.validation_end
        )
        val_df = development_df.loc[val_mask]
        perm_df = permutation_importance_for_fold(
            result.trained, val_df, feature_cols, target_col, n_repeats=n_permutation_repeats
        )
        if not perm_df.empty:
            permutation_series.append(perm_df.set_index("feature")["importance"])
            top_permutation.append(list(perm_df.head(top_k)["feature"]))

    native_corr = _mean_pairwise_spearman(native_series)
    permutation_corr = _mean_pairwise_spearman(permutation_series)

    # Sign-flip: a feature whose PERMUTATION importance is positive
    # (genuinely helped) in at least one fold and negative (actively hurt
    # -- shuffling it improved the score) in at least one other.
    sign_flip = []
    if permutation_series:
        perm_matrix = pd.concat(permutation_series, axis=1).fillna(0.0)
        has_pos = (perm_matrix > 0).any(axis=1)
        has_neg = (perm_matrix < 0).any(axis=1)
        sign_flip = sorted(perm_matrix.index[has_pos & has_neg].tolist())

    # One-period-only: a feature that appears in top_k native importance
    # in EXACTLY one fold and no others.
    one_period_only = []
    if len(top_native) >= 2:
        appearance_count: dict[str, int] = {}
        for feats in top_native:
            for f in set(feats):
                appearance_count[f] = appearance_count.get(f, 0) + 1
        one_period_only = sorted([f for f, count in appearance_count.items() if count == 1])

    # Family-level stability: per fold, sum native importance within each
    # family; then the cross-fold rank correlation of family totals.
    family_stability: dict = {}
    if native_series:
        family_totals = []
        for series in native_series:
            by_family: dict[str, float] = {}
            for feature, value in series.items():
                fam = classify_feature_family(feature)
                by_family[fam] = by_family.get(fam, 0.0) + float(value)
            family_totals.append(pd.Series(by_family))
        family_corr = _mean_pairwise_spearman(family_totals)
        mean_family_importance = pd.concat(family_totals, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)
        family_stability = {
            "mean_pairwise_spearman": family_corr,
            "mean_importance_by_family": mean_family_importance.to_dict(),
        }

    scores = [v for v in (native_corr, permutation_corr) if v == v]
    stability_score = float(np.mean(scores)) if scores else float("nan")

    return FeatureStabilityReport(
        n_folds=len(fold_results),
        top_features_native_per_fold=top_native,
        top_features_permutation_per_fold=top_permutation,
        native_rank_correlation=native_corr,
        permutation_rank_correlation=permutation_corr,
        sign_flip_features=sign_flip,
        one_period_only_features=one_period_only,
        family_stability=family_stability,
        stability_score=stability_score,
    )
