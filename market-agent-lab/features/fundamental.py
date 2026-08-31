"""Deterministic fundamental feature engineering.

Like the technical engine, this module performs plain arithmetic on
already-fetched, as-of fundamentals (see ``data/fundamentals.py``). The
Fundamental Analysis Agent (Phase 3) consumes these numbers; it never
computes ratios itself.

Valuation and quality metrics are expressed both as raw values and as
cross-sectional z-scores against the rest of the universe *as of the same
date*, which gives the agent a sense of "cheap vs. peers" rather than an
absolute number with no context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNDAMENTAL_FEATURE_COLUMNS: list[str] = [
    "revenue_growth", "eps_growth", "gross_margin", "operating_margin",
    "fcf_margin", "roic", "debt_to_cash",
    "pe_ratio", "ev_to_ebitda", "price_to_book", "price_to_sales",
    "valuation_zscore", "profitability_zscore", "growth_zscore",
]

_VALUATION_COLS = ["pe_ratio", "ev_to_ebitda", "price_to_book", "price_to_sales"]
_PROFITABILITY_COLS = ["gross_margin", "operating_margin", "fcf_margin", "roic"]
_GROWTH_COLS = ["revenue_growth", "eps_growth"]


def _cross_sectional_zscore(universe: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Mean z-score across ``cols`` for every row of ``universe`` (higher = better)."""
    z = pd.DataFrame(index=universe.index)
    for col in cols:
        series = universe[col].astype(float)
        std = series.std(ddof=0)
        if not std or np.isnan(std):
            z[col] = 0.0
        else:
            z[col] = (series - series.mean()) / std
    return z.mean(axis=1)


def compute_fundamental_features(universe_asof: pd.DataFrame) -> pd.DataFrame:
    """Compute fundamental features for every symbol in an as-of universe snapshot.

    ``universe_asof`` is expected to be the output of
    ``data.fundamentals.get_fundamentals_asof`` for *all* symbols at one
    ``as_of`` timestamp, so cross-sectional statistics are computed fairly
    across the peer set available on that date.
    """
    if universe_asof.empty:
        return pd.DataFrame(columns=["symbol", *FUNDAMENTAL_FEATURE_COLUMNS])

    df = universe_asof.copy()
    df["debt_to_cash"] = df["debt"] / df["cash"].replace(0.0, np.nan)

    # Valuation metrics are "cheaper is better" -- invert sign so that a
    # higher valuation_zscore always means cheaper/more attractive.
    valuation_z = -_cross_sectional_zscore(df, _VALUATION_COLS)
    profitability_z = _cross_sectional_zscore(df, _PROFITABILITY_COLS)
    growth_z = _cross_sectional_zscore(df, _GROWTH_COLS)

    out = df[["symbol"]].copy()
    for col in [
        "revenue_growth", "eps_growth", "gross_margin", "operating_margin",
        "fcf_margin", "roic", "debt_to_cash",
        "pe_ratio", "ev_to_ebitda", "price_to_book", "price_to_sales",
    ]:
        out[col] = df[col]
    out["valuation_zscore"] = valuation_z
    out["profitability_zscore"] = profitability_z
    out["growth_zscore"] = growth_z
    return out.reset_index(drop=True)
