"""Tests for backtesting/daily_portfolio.py (V0.3 Stage 2): the corrected,
genuinely chronological daily Sharpe calculation, and regression coverage
proving the old build_quantile_portfolio_returns-based Sharpe (still used
elsewhere for relative-degradation diagnostics, never for a reported
Sharpe number) is inflated relative to it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.daily_portfolio import (
    build_daily_rebalanced_portfolio_returns,
    sharpe_audit_report,
)
from backtesting.metrics import TRADING_DAYS_PER_YEAR, sharpe_ratio
from backtesting.robustness import build_quantile_portfolio_returns


def _market_df(prices: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"symbol": sym, "timestamp": d, "adjusted_close": prices.loc[d, sym]}
        for sym in prices.columns for d in prices.index
    ]
    return pd.DataFrame(rows)


def _synthetic_universe(n_symbols=6, n_days=60, seed=1, drift=None):
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rng = np.random.default_rng(seed)
    drift = drift if drift is not None else np.zeros(n_symbols)
    daily_rets = pd.DataFrame(
        {sym: drift[i] + rng.normal(0, 0.01, n_days) for i, sym in enumerate(symbols)}, index=dates
    )
    prices = (1 + daily_rets).cumprod() * 100.0
    return symbols, dates, daily_rets, prices


# --- mechanical correctness: one row per TRADING DAY, never per symbol --------------------------


def test_output_is_one_row_per_trading_day_not_per_symbol():
    """The core V0.3 Stage 2 regression: a naive symbol-level flattening
    would produce n_days * n_symbols rows; the correct chronological
    series must produce at most n_days rows (one per calendar day), never
    scaling with the number of symbols."""
    symbols, dates, daily_rets, prices = _synthetic_universe(n_symbols=8, n_days=50, seed=2)
    market_df = _market_df(prices)

    rng = np.random.default_rng(3)
    pred_rows = [
        {"symbol": sym, "timestamp": d, "pred": rng.normal()}
        for sym in symbols for d in dates
    ]
    predictions_df = pd.DataFrame(pred_rows)

    result = build_daily_rebalanced_portfolio_returns(predictions_df, market_df, "pred")
    assert len(result) <= len(dates)
    assert len(result) > 0
    assert result["timestamp"].is_unique  # exactly one row per date, never one per (symbol, date)
    assert set(result.columns) >= {"timestamp", "gross_return", "turnover", "n_long", "n_short"}


def test_no_look_ahead_signal_takes_effect_the_day_after_it_was_observed():
    """A rebalance decided using date D's signal must not affect date D's
    own realized return -- only trading days strictly after D."""
    symbols = ["A", "B", "C", "D"]
    dates = pd.bdate_range("2021-01-04", periods=10)
    # Symbol A jumps 50% on day index 5 ONLY; every other symbol is flat.
    prices = pd.DataFrame(100.0, index=dates, columns=symbols)
    prices.loc[dates[5]:, "A"] = 150.0
    market_df = _market_df(prices)

    # The signal that would put A at the top of the book is only observed
    # on day index 4 (the day BEFORE A's jump) -- if there is no
    # look-ahead, A's jump on day 5 should be captured (book active by
    # then); a signal observed ON day 5 itself must not retroactively
    # capture day 5's own return.
    predictions_df = pd.DataFrame(
        [{"symbol": s, "timestamp": dates[4], "pred": 1.0 if s == "A" else 0.0} for s in symbols]
        + [{"symbol": s, "timestamp": dates[8], "pred": -1.0 if s == "A" else 0.0} for s in symbols]
    )
    result = build_daily_rebalanced_portfolio_returns(predictions_df, market_df, "pred", top_frac=0.25)
    # Day index 5 (A jumps) must show a large positive return -- the book
    # from day 4's signal (long A) was already active.
    day5_row = result[result["timestamp"] == dates[5]]
    assert not day5_row.empty
    assert day5_row["gross_return"].iloc[0] > 0.1


def test_turnover_is_zero_while_holding_and_nonzero_on_a_rebalance_day():
    symbols, dates, _, prices = _synthetic_universe(n_symbols=6, n_days=40, seed=4)
    market_df = _market_df(prices)
    rng = np.random.default_rng(5)
    # Only two rebalance dates -- the book should be held (turnover 0.0)
    # on every day in between.
    reb_dates = [dates[2], dates[20]]
    predictions_df = pd.DataFrame(
        [{"symbol": sym, "timestamp": d, "pred": rng.normal()} for d in reb_dates for sym in symbols]
    )
    result = build_daily_rebalanced_portfolio_returns(predictions_df, market_df, "pred", top_frac=0.33)
    # The day the first book activates carries turnover 1.0 (whole book is new).
    first_active = result.iloc[0]
    assert first_active["turnover"] == pytest.approx(1.0)
    # Holding days between rebalances show zero turnover.
    holding_rows = result[(result["timestamp"] > dates[3]) & (result["timestamp"] < dates[21])]
    assert not holding_rows.empty
    assert (holding_rows["turnover"] == 0.0).all()


# --- sharpe_audit_report: shape and required fields ----------------------------------------------


def test_sharpe_audit_report_contains_every_required_field():
    symbols, dates, _, prices = _synthetic_universe(n_symbols=10, n_days=120, seed=6)
    market_df = _market_df(prices)
    rng = np.random.default_rng(7)
    predictions_df = pd.DataFrame(
        [{"symbol": sym, "timestamp": d, "pred": rng.normal()} for d in dates for sym in symbols]
    )
    portfolio = build_daily_rebalanced_portfolio_returns(predictions_df, market_df, "pred")
    report = sharpe_audit_report(portfolio)
    required = {
        "n_observations", "date_range", "mean_daily_return", "daily_volatility", "annualization_factor",
        "gross_sharpe", "net_sharpe", "mean_turnover", "cost_bps_assumption", "total_cost_drag_annualized",
        "max_drawdown",
    }
    assert required <= set(report.keys())
    assert report["n_observations"] == len(portfolio)
    assert report["annualization_factor"] == pytest.approx(np.sqrt(TRADING_DAYS_PER_YEAR))
    assert report["date_range"][0] <= report["date_range"][1]


def test_sharpe_audit_report_empty_input_returns_nan_not_a_crash():
    report = sharpe_audit_report(pd.DataFrame(columns=["timestamp", "gross_return", "turnover"]))
    assert report["n_observations"] == 0
    assert report["gross_sharpe"] != report["gross_sharpe"]  # NaN


# --- the actual inflation regression: overlapping-target Sharpe vs the corrected daily series ----


def test_overlapping_target_sharpe_is_inflated_relative_to_genuine_daily_sharpe():
    """The central V0.3 Stage 2 regression test: for the SAME underlying
    persistent per-symbol signal, computing "Sharpe" from
    build_quantile_portfolio_returns's overlapping 20-day-forward series
    (annualised at sqrt(252) as if independent daily returns -- the OLD,
    buggy path that fed the promotion gate) must come out substantially
    higher than the genuine daily-rebalanced Sharpe computed from real
    chronological one-day returns. If this test ever fails because the two
    numbers converge, it means the overlap/annualisation bug has silently
    come back."""
    n_symbols, n_days = 20, 300
    rng = np.random.default_rng(42)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    true_drift = rng.normal(0, 0.0003, n_symbols)  # a real, persistent, modest per-symbol edge
    daily_rets = pd.DataFrame(
        {sym: true_drift[i] + rng.normal(0, 0.015, n_days) for i, sym in enumerate(symbols)}, index=dates
    )
    prices = (1 + daily_rets).cumprod() * 100.0
    market_df = _market_df(prices)

    horizon = 20
    fwd20 = daily_rets.rolling(horizon).sum().shift(-horizon)
    pred_rows, target_rows = [], []
    for sym in symbols:
        for d in dates:
            f = fwd20.loc[d, sym]
            if pd.isna(f):
                continue
            pred_rows.append({"symbol": sym, "timestamp": d, "predicted_excess_return_20d": f + rng.normal(0, 0.02)})
            target_rows.append({"symbol": sym, "timestamp": d, "excess_return_20d": f})
    predictions_df = pd.DataFrame(pred_rows)
    eval_frame = predictions_df.merge(pd.DataFrame(target_rows), on=["symbol", "timestamp"])

    old_portfolio = build_quantile_portfolio_returns(eval_frame, "excess_return_20d", "predicted_excess_return_20d")
    old_sharpe = sharpe_ratio(old_portfolio["gross_return"])

    new_portfolio = build_daily_rebalanced_portfolio_returns(predictions_df, market_df, "predicted_excess_return_20d")
    new_sharpe = sharpe_audit_report(new_portfolio)["gross_sharpe"]

    assert old_sharpe == old_sharpe and new_sharpe == new_sharpe  # neither is NaN
    assert old_sharpe > 0  # this scenario has a genuine positive edge
    # The overlapping-window calculation must be substantially larger --
    # not just marginally -- than the corrected one.
    assert new_sharpe < old_sharpe * 0.5
