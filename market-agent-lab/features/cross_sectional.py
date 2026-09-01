"""Cross-sectional feature ranks (Phase 11 of the brief).

For each date, ranks are computed strictly across the symbols present in
the point-in-time investable universe for that date (see
``data/universe.py``) -- never against today's universe applied
retroactively, which would leak future index-membership information into
a historical percentile.
"""

from __future__ import annotations

import pandas as pd

# name -> (source_column, ascending). ascending=True means "higher raw
# value -> higher percentile"; False inverts it (e.g. lower volatility is
# the 'better'/higher percentile end for a volatility-percentile feature).
DEFAULT_METRIC_SPEC: dict[str, tuple[str, bool]] = {
    "momentum_percentile": ("return_60d", True),
    "value_percentile": ("valuation_zscore", True),
    "quality_percentile": ("profitability_zscore", True),
    "growth_percentile": ("growth_zscore", True),
    "volatility_percentile": ("realised_vol_20d", False),
    "liquidity_percentile": ("dollar_volume", True),
    "earnings_quality_percentile": ("fundamental_earnings_quality_score", True),
}


def compute_percentile_ranks(
    cross_section: pd.DataFrame, metric_spec: dict[str, tuple[str, bool]] | None = None
) -> pd.DataFrame:
    """``cross_section``: one row per symbol, for a SINGLE date, restricted
    to that date's point-in-time universe. Returns ``symbol`` plus one
    ``{name}`` percentile-rank column [0, 1] per metric whose source
    column is present (metrics whose source column is missing are simply
    skipped, not fabricated as 0.5)."""
    metric_spec = metric_spec or DEFAULT_METRIC_SPEC
    out = pd.DataFrame({"symbol": cross_section["symbol"].to_numpy()})
    for name, (source_col, ascending) in metric_spec.items():
        if source_col not in cross_section.columns:
            continue
        out[name] = cross_section[source_col].rank(pct=True, ascending=ascending)
    return out


def compute_percentile_ranks_multi_date(
    panel: pd.DataFrame, metric_spec: dict[str, tuple[str, bool]] | None = None
) -> pd.DataFrame:
    """Apply :func:`compute_percentile_ranks` independently per
    ``timestamp`` group -- the panel must already be restricted, per date,
    to that date's point-in-time universe (callers filter before calling)."""
    parts = []
    for ts, group in panel.groupby("timestamp", sort=False):
        ranks = compute_percentile_ranks(group, metric_spec)
        ranks.insert(0, "timestamp", ts)
        parts.append(ranks)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "symbol"])
    return pd.concat(parts, ignore_index=True)
