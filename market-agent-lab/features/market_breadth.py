"""Market breadth features (Phase 12).

Computed cross-sectionally, one row per date, from that date's
point-in-time universe's technical features -- never from "today's"
universe applied to a historical date.
"""

from __future__ import annotations

import pandas as pd


def compute_market_breadth(cross_section: pd.DataFrame, dollar_volume_col: str = "dollar_volume") -> dict[str, float]:
    """``cross_section``: one row per symbol for a single date, with at
    least ``dist_sma_20``, ``dist_sma_50``, ``dist_sma_200``, ``return_1d``,
    ``dist_52w_high``, ``dist_52w_low`` columns from
    ``features/technical.py``. Missing columns are simply omitted from the
    result rather than fabricated.
    """
    breadth: dict[str, float] = {}
    n = len(cross_section)
    if n == 0:
        return breadth

    for col, key in (("dist_sma_20", "pct_above_sma20"), ("dist_sma_50", "pct_above_sma50"), ("dist_sma_200", "pct_above_sma200")):
        if col in cross_section.columns:
            valid = cross_section[col].dropna()
            if len(valid):
                breadth[key] = float((valid > 0).mean())

    if "return_1d" in cross_section.columns:
        r = cross_section["return_1d"].dropna()
        if len(r):
            breadth["advance_decline_proxy"] = float((r > 0).mean() - (r < 0).mean())
            breadth["cross_sectional_return_dispersion"] = float(r.std(ddof=0))
            breadth["median_stock_return"] = float(r.median())
            breadth["equal_weight_return"] = float(r.mean())

            if dollar_volume_col in cross_section.columns:
                weights = cross_section.loc[r.index, dollar_volume_col].clip(lower=0)
                total_weight = weights.sum()
                if total_weight > 0:
                    cap_weight_return = float((r * weights / total_weight).sum())
                    breadth["dollar_volume_weighted_return"] = cap_weight_return
                    breadth["equal_vs_volume_weighted_spread"] = breadth["equal_weight_return"] - cap_weight_return

    if "dist_52w_high" in cross_section.columns:
        d = cross_section["dist_52w_high"].dropna()
        if len(d):
            breadth["pct_new_52w_highs"] = float((d >= -0.001).mean())
    if "dist_52w_low" in cross_section.columns:
        d = cross_section["dist_52w_low"].dropna()
        if len(d):
            breadth["pct_new_52w_lows"] = float((d <= 0.001).mean())

    return breadth


def compute_market_breadth_series(panel: pd.DataFrame, dollar_volume_col: str = "dollar_volume") -> pd.DataFrame:
    """Apply :func:`compute_market_breadth` per ``timestamp`` group; panel
    must already be restricted, per date, to that date's point-in-time
    universe."""
    rows = []
    for ts, group in panel.groupby("timestamp", sort=False):
        breadth = compute_market_breadth(group, dollar_volume_col)
        breadth["timestamp"] = ts
        rows.append(breadth)
    return pd.DataFrame(rows)
