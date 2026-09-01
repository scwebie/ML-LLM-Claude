"""Extended drift monitoring: KS test and Wasserstein distance (Stage 14,
V0.2), alongside V0.1's PSI (``learning/drift.py``, left completely
untouched -- V0.1's demo pipeline keeps using PSI exactly as before).

PSI is a useful coarse, binned summary but is sensitive to bin-edge
choice and blind to distribution shape within a bin. This module adds
two complementary, bin-free statistics:

* Kolmogorov-Smirnov two-sample test -- the maximum distance between the
  two empirical CDFs, with a p-value for "these two samples could plausibly
  come from the same distribution."
* Wasserstein (earth-mover's) distance -- the minimum "work" to transform
  one distribution into the other; unlike KS, it is sensitive to HOW FAR
  apart the distributions are, not just whether they differ at all.

Same governing rule as ``learning/drift.py``: these are deterministic
statistics used only to *trigger* a retraining/challenger cycle -- never
to modify risk limits or bypass the champion/challenger promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy import stats

from learning.drift import population_stability_index


def ks_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Returns ``(ks_statistic, p_value)``. A small p-value is evidence
    the two samples come from different distributions."""
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) < 2 or len(cur) < 2:
        return 0.0, 1.0
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def wasserstein(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return 0.0
    return float(stats.wasserstein_distance(ref, cur))


@dataclass(frozen=True)
class DriftStatistic:
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    wasserstein_distance: float
    flagged: bool
    reasons: list[str] = field(default_factory=list)


def detect_feature_drift_full(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = 0.25,
    ks_p_value_threshold: float = 0.01,
    wasserstein_threshold: float | None = 2.0,
) -> list[DriftStatistic]:
    """Runs PSI, KS, and Wasserstein distance for every feature present in
    both frames and flags a feature as drifted if ANY of:
      * PSI exceeds ``psi_threshold`` (V0.1's existing rule of thumb), or
      * the KS test rejects equal distributions at ``ks_p_value_threshold``, or
      * (when ``wasserstein_threshold`` is set) the Wasserstein distance
        exceeds ``wasserstein_threshold`` times the reference period's own
        standard deviation -- i.e. the distribution has moved by more than
        that many reference-period "typical spreads."
    Returns one :class:`DriftStatistic` per checked feature (flagged or
    not), so a caller can report the full picture, not just the alarms."""
    results = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col]
        cur = current_df[col]
        psi = population_stability_index(ref, cur)
        ks_stat, ks_p = ks_test(ref, cur)
        wass = wasserstein(ref, cur)

        reasons = []
        if psi > psi_threshold:
            reasons.append(f"PSI {psi:.3f} > {psi_threshold}")
        if ks_p < ks_p_value_threshold:
            reasons.append(f"KS p-value {ks_p:.4f} < {ks_p_value_threshold} (statistic={ks_stat:.3f})")
        if wasserstein_threshold is not None:
            ref_std = ref.dropna().std()
            if ref_std and wass > wasserstein_threshold * ref_std:
                reasons.append(f"Wasserstein distance {wass:.4f} > {wasserstein_threshold}x reference std ({ref_std:.4f})")

        results.append(
            DriftStatistic(
                feature=col, psi=psi, ks_statistic=ks_stat, ks_p_value=ks_p,
                wasserstein_distance=wass, flagged=bool(reasons), reasons=reasons,
            )
        )
    return results
