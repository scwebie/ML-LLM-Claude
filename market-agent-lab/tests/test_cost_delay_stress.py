"""Tests for backtesting/cost_delay_stress.py (V0.3 Stage 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.cost_delay_stress import (
    DEFAULT_COST_BPS_GRID,
    DEFAULT_DELAY_VARIANTS,
    DEFAULT_REBALANCE_VARIANTS,
    _resample_to_weekly_signal_dates,
    run_cost_delay_turnover_stress,
)


def _universe(n_symbols=10, n_days=200, seed=1):
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    daily_rets = pd.DataFrame({s: rng.normal(0.0002, 0.012, n_days) for s in symbols}, index=dates)
    close = 100.0 * (1 + daily_rets).cumprod()
    open_ = close.shift(1).fillna(close.iloc[0])  # a simple, distinct open series
    rows = []
    for s in symbols:
        for d in dates:
            rows.append({"symbol": s, "timestamp": d, "adjusted_close": close.loc[d, s], "open": open_.loc[d, s]})
    market_df = pd.DataFrame(rows)

    pred_rows = []
    for s in symbols:
        for d in dates:
            pred_rows.append({"symbol": s, "timestamp": d, "pred": rng.normal()})
    predictions_df = pd.DataFrame(pred_rows)
    return predictions_df, market_df


def test_resample_to_weekly_signal_dates_keeps_one_date_per_iso_week():
    dates = pd.bdate_range("2021-01-04", periods=15)  # 3 ISO weeks
    df = pd.DataFrame({"symbol": "A", "timestamp": dates, "pred": range(15)})
    weekly = _resample_to_weekly_signal_dates(df)
    kept_dates = sorted(weekly["timestamp"].unique())
    assert len(kept_dates) == 3
    # First kept date of each week should be that week's earliest date present.
    assert kept_dates[0] == dates[0]


def test_run_cost_delay_turnover_stress_produces_full_grid():
    predictions_df, market_df = _universe()
    report = run_cost_delay_turnover_stress(predictions_df, market_df, "pred")
    expected_rows = len(DEFAULT_REBALANCE_VARIANTS) * len(DEFAULT_DELAY_VARIANTS) * len(DEFAULT_COST_BPS_GRID)
    assert len(report) == expected_rows
    required_cols = {"rebalance", "execution_delay", "cost_bps", "net_sharpe", "cagr", "mean_turnover", "max_drawdown", "n_observations"}
    assert required_cols <= set(report.columns)


def test_run_cost_delay_turnover_stress_net_sharpe_never_improves_with_higher_cost():
    """For a fixed rebalance/delay combination, net Sharpe at a higher
    cost assumption must never exceed net Sharpe at a lower one -- costs
    only ever subtract from returns when there is any turnover."""
    predictions_df, market_df = _universe()
    report = run_cost_delay_turnover_stress(
        predictions_df, market_df, "pred",
        cost_bps_grid=(0, 50), delay_variants=(("next_close", 0, "adjusted_close"),),
        rebalance_variants=(("daily", "daily", None),),
    )
    zero_cost = report[report["cost_bps"] == 0]["net_sharpe"].iloc[0]
    high_cost = report[report["cost_bps"] == 50]["net_sharpe"].iloc[0]
    assert high_cost <= zero_cost


def test_run_cost_delay_turnover_stress_weekly_rebalance_has_lower_mean_turnover_than_daily():
    predictions_df, market_df = _universe()
    report = run_cost_delay_turnover_stress(
        predictions_df, market_df, "pred",
        cost_bps_grid=(10,), delay_variants=(("next_close", 0, "adjusted_close"),),
        rebalance_variants=(("daily", "daily", None), ("weekly", "weekly", None)),
    )
    daily_turnover = report[report["rebalance"] == "daily"]["mean_turnover"].iloc[0]
    weekly_turnover = report[report["rebalance"] == "weekly"]["mean_turnover"].iloc[0]
    assert weekly_turnover < daily_turnover
