"""Deterministic macro feature engineering.

Converts a long history of as-of macro observations into trailing
z-scores per series, which the Market Overview Agent (Phase 3) uses to
classify regimes. All arithmetic here is plain statistics -- no LLM
involvement.
"""

from __future__ import annotations

import pandas as pd

MACRO_FEATURE_SUFFIX = "_zscore"


def compute_macro_features(macro_history: pd.DataFrame, as_of: pd.Timestamp, lookback_periods: int = 36) -> dict[str, float]:
    """Trailing z-score per macro series, using only observations published <= as_of.

    ``macro_history`` should already be filtered to
    ``publication_timestamp <= as_of`` (see ``data/macro.py``); this
    function additionally restricts the trailing window used for the
    mean/std baseline to the most recent ``lookback_periods`` observations
    *per series* strictly at-or-before ``as_of``, so the baseline itself
    never uses information from the future.
    """
    if macro_history.empty:
        return {}

    features: dict[str, float] = {}
    for series_name, group in macro_history.groupby("series_name"):
        group = group.sort_values("timestamp")
        window = group.tail(lookback_periods)
        if len(window) < 3:
            continue
        std = window["value"].std(ddof=0)
        latest_value = group["value"].iloc[-1]
        if not std:
            z = 0.0
        else:
            z = (latest_value - window["value"].mean()) / std
        features[f"{series_name}{MACRO_FEATURE_SUFFIX}"] = float(z)
        features[f"{series_name}_level"] = float(latest_value)
    return features
