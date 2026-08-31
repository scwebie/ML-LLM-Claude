"""Shared transaction-cost assumptions for the backtester.

The primary ML-driven strategy is simulated event-by-event through the
real Paper Execution Engine (``execution/fills.py``, spreads + slippage +
commissions), so its costs fall out of that simulation directly. The
comparison benchmarks (buy-and-hold, equal-weight, momentum) are simpler
vectorised daily-rebalance strategies; this module gives them a
consistent, comparable cost drag per unit of turnover so that "beat
buy-and-hold after costs" is a fair comparison rather than the benchmarks
running frictionlessly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from execution.fills import FillConfig

BENCHMARK_FILL_CONFIG = FillConfig()


@dataclass(frozen=True)
class SimpleCostModel:
    cost_per_unit_turnover: float = (
        BENCHMARK_FILL_CONFIG.half_spread_bps / 10_000.0 + BENCHMARK_FILL_CONFIG.commission_per_share * 0
    ) * 2  # round-trip spread cost; commissions are per-share and not meaningful for a notional-based vectorised model


def apply_cost_drag(gross_returns: pd.Series, turnover_series: pd.Series, cost_model: SimpleCostModel | None = None) -> pd.Series:
    """Subtract a simple turnover-proportional cost drag from daily gross returns."""
    cost_model = cost_model or SimpleCostModel()
    turnover_series = turnover_series.reindex(gross_returns.index).fillna(0.0)
    return gross_returns - turnover_series * cost_model.cost_per_unit_turnover
