"""V0.3 Stage 12: transaction-cost, execution-delay, and turnover-cadence
stress tests, DEVELOPMENT PERIOD ONLY.

Every combination is run on the SAME genuinely chronological daily
portfolio construction from V0.3 Stage 2
(``backtesting.daily_portfolio.build_daily_rebalanced_portfolio_returns``)
-- never the overlapping-target series. Cost/delay/cadence values are
FIXED, standard grid points given directly (0/5/10/20/50bps; next-close/
next-open/+1-day; daily/weekly/threshold), never tuned to make any one
combination look better.
"""

from __future__ import annotations

import pandas as pd

from backtesting.daily_portfolio import (
    build_daily_rebalanced_portfolio_returns,
    sharpe_audit_report,
)
from backtesting.metrics import cagr as compute_cagr

DEFAULT_COST_BPS_GRID: tuple[float, ...] = (0, 5, 10, 20, 50)

# (label, extra_delay_days, price_col)
DEFAULT_DELAY_VARIANTS: tuple[tuple[str, int, str], ...] = (
    ("next_close", 0, "adjusted_close"),
    ("next_open", 0, "open"),
    ("plus_one_trading_day_delay", 1, "adjusted_close"),
)

# (label, cadence, rebalance_threshold)
DEFAULT_REBALANCE_VARIANTS: tuple[tuple[str, str, float | None], ...] = (
    ("daily", "daily", None),
    ("weekly", "weekly", None),
    ("threshold_based_30pct", "daily", 0.30),
)


def _resample_to_weekly_signal_dates(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the first available prediction date in each ISO
    (year, week) -- one rebalance signal per week, same convention the
    daily builder already uses to hold a book between signal dates."""
    if predictions_df.empty:
        return predictions_df
    dates = sorted(predictions_df["timestamp"].unique())
    seen: set[tuple[int, int]] = set()
    keep: list = []
    for d in dates:
        iso = pd.Timestamp(d).isocalendar()
        key = (int(iso.year), int(iso.week))
        if key not in seen:
            seen.add(key)
            keep.append(d)
    return predictions_df[predictions_df["timestamp"].isin(keep)]


def run_cost_delay_turnover_stress(
    predictions_df: pd.DataFrame,
    market_df: pd.DataFrame,
    pred_col: str,
    cost_bps_grid: tuple[float, ...] = DEFAULT_COST_BPS_GRID,
    delay_variants: tuple[tuple[str, int, str], ...] = DEFAULT_DELAY_VARIANTS,
    rebalance_variants: tuple[tuple[str, str, float | None], ...] = DEFAULT_REBALANCE_VARIANTS,
) -> pd.DataFrame:
    """Full grid: rebalance cadence x execution delay x transaction cost.
    One row per combination with net Sharpe, CAGR, mean turnover, and max
    drawdown -- all computed from a genuinely chronological daily
    portfolio-return series, so these numbers are directly comparable to
    the corrected Sharpe from V0.3 Stage 2, not the earlier overlapping-
    target one."""
    rows: list[dict] = []
    for rebalance_label, cadence, threshold in rebalance_variants:
        preds = _resample_to_weekly_signal_dates(predictions_df) if cadence == "weekly" else predictions_df
        for delay_label, extra_delay_days, price_col in delay_variants:
            portfolio = build_daily_rebalanced_portfolio_returns(
                preds, market_df, pred_col, price_col=price_col,
                extra_delay_days=extra_delay_days, rebalance_threshold=threshold,
            )
            for cost_bps in cost_bps_grid:
                audit = sharpe_audit_report(portfolio, cost_bps=cost_bps)
                if not portfolio.empty:
                    cost = portfolio["turnover"] * (cost_bps / 10_000.0)
                    equity = (1.0 + portfolio["gross_return"] - cost).cumprod()
                    cagr_value = compute_cagr(equity) if len(equity) > 1 else float("nan")
                else:
                    cagr_value = float("nan")
                rows.append(
                    {
                        "rebalance": rebalance_label, "execution_delay": delay_label, "cost_bps": cost_bps,
                        "net_sharpe": audit["net_sharpe"], "cagr": cagr_value,
                        "mean_turnover": audit["mean_turnover"], "max_drawdown": audit["max_drawdown"],
                        "n_observations": audit["n_observations"],
                    }
                )
    return pd.DataFrame(rows)
