"""Backtest performance metrics (Phase 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    n_periods = len(equity) - 1
    growth = equity.iloc[-1] / equity.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return float("nan")
    rf_period = risk_free_annual / periods_per_year
    excess = returns - rf_period
    return float(excess.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) < 2:
        return float("nan")
    rf_period = risk_free_annual / periods_per_year
    excess = returns - rf_period
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not downside_std:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    if not max_dd or max_dd == 0 or max_dd != max_dd:
        return float("nan")
    return float(cagr_value / abs(max_dd))


def hit_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return float("nan")
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def profit_factor(trade_pnls: list[float]) -> float:
    wins = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def avg_win(trade_pnls: list[float]) -> float:
    wins = [p for p in trade_pnls if p > 0]
    return float(np.mean(wins)) if wins else float("nan")


def avg_loss(trade_pnls: list[float]) -> float:
    losses = [p for p in trade_pnls if p < 0]
    return float(np.mean(losses)) if losses else float("nan")


def turnover(traded_notional: pd.Series, avg_equity: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised turnover = total traded notional / average equity / number of years covered."""
    if avg_equity <= 0 or traded_notional.empty:
        return float("nan")
    n_years = max(len(traded_notional) / periods_per_year, 1e-9)
    return float(traded_notional.sum() / avg_equity / n_years)


def beta_alpha(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> tuple[float, float]:
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan"), float("nan")
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    cov = np.cov(r, b)
    if cov[1, 1] == 0:
        return float("nan"), float("nan")
    beta = cov[0, 1] / cov[1, 1]
    alpha_daily = r.mean() - beta * b.mean()
    alpha_annual = alpha_daily * periods_per_year
    return float(beta), float(alpha_annual)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if active.std(ddof=1) == 0:
        return float("nan")
    return float(active.mean() / active.std(ddof=1) * np.sqrt(periods_per_year))


def exposure(gross_exposure_series: pd.Series) -> float:
    if gross_exposure_series.empty:
        return float("nan")
    return float(gross_exposure_series.mean())


def avg_holding_period_days(holding_periods: list[float]) -> float:
    return float(np.mean(holding_periods)) if holding_periods else float("nan")


def compute_all_metrics(
    equity: pd.Series,
    benchmark_equity: pd.Series | None,
    trade_pnls: list[float],
    traded_notional: pd.Series,
    gross_exposure_series: pd.Series,
    holding_periods: list[float],
) -> dict[str, float]:
    rets = daily_returns(equity)
    bench_rets = daily_returns(benchmark_equity) if benchmark_equity is not None else None
    mdd = max_drawdown(equity)
    cagr_value = cagr(equity)

    metrics = {
        "total_return": total_return(equity),
        "cagr": cagr_value,
        "annualized_volatility": annualized_volatility(rets),
        "sharpe_ratio": sharpe_ratio(rets),
        "sortino_ratio": sortino_ratio(rets),
        "max_drawdown": mdd,
        "calmar_ratio": calmar_ratio(cagr_value, mdd),
        "hit_rate": hit_rate(trade_pnls),
        "profit_factor": profit_factor(trade_pnls),
        "turnover": turnover(traded_notional, equity.mean()),
        "avg_win": avg_win(trade_pnls),
        "avg_loss": avg_loss(trade_pnls),
        "exposure": exposure(gross_exposure_series),
        "avg_holding_period_days": avg_holding_period_days(holding_periods),
    }
    if bench_rets is not None:
        beta, alpha = beta_alpha(rets, bench_rets)
        metrics["beta"] = beta
        metrics["alpha"] = alpha
        metrics["information_ratio"] = information_ratio(rets, bench_rets)
    else:
        metrics["beta"] = float("nan")
        metrics["alpha"] = float("nan")
        metrics["information_ratio"] = float("nan")
    return metrics
