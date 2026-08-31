"""Drift detection: feature-distribution drift (PSI) and performance drift.

Deterministic statistics only -- used to *trigger* a retraining/challenger
cycle (Phase 10), never to modify risk limits or bypass the champion/
challenger promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Standard PSI between two numeric distributions using reference-derived bin edges."""
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) < bins or len(current) < 1:
        return 0.0

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_feature_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str], psi_threshold: float = 0.25
) -> dict[str, float]:
    """Returns {feature: psi} for every feature whose PSI exceeds ``psi_threshold``."""
    flagged = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[col], current_df[col])
        if psi > psi_threshold:
            flagged[col] = psi
    return flagged


@dataclass(frozen=True)
class PerformanceDriftResult:
    drifted: bool
    recent_information_coefficient: float
    reference_information_coefficient: float
    rationale: str


def detect_performance_drift(
    recent_ic: float, reference_ic: float, relative_drop_threshold: float = 0.5
) -> PerformanceDriftResult:
    """Flags drift when the recent (live/OOS) information coefficient has
    degraded by more than ``relative_drop_threshold`` relative to the
    model's validation-time IC."""
    if reference_ic == 0 or reference_ic != reference_ic:
        drifted = recent_ic < 0
        rationale = "reference IC was ~0; flagging only if recent IC has turned negative"
    else:
        relative_drop = (reference_ic - recent_ic) / abs(reference_ic)
        drifted = relative_drop > relative_drop_threshold
        rationale = f"IC dropped {relative_drop:.1%} relative to reference ({reference_ic:.4f} -> {recent_ic:.4f})"
    return PerformanceDriftResult(drifted, recent_ic, reference_ic, rationale)
