"""A genuinely chronological, non-overlapping daily portfolio return
series and the Sharpe-ratio audit built from it (V0.3 Stage 2).

**The bug this module fixes.** Every prior Sharpe reported by this
pipeline -- including the one that fed the champion/challenger promotion
gate (``real_pipeline.py``) and the "raw gross sharpe" numbers in
``backtesting/robustness.py::cost_stress_test`` / the earlier
``EXPERIMENT_REPORT.md`` -- was computed by
``sharpe_ratio(build_quantile_portfolio_returns(eval_frame, ...)["gross_return"])``.
``build_quantile_portfolio_returns`` produces one row per *prediction
date*, but the value in that row is ``target_col`` (``excess_return_5d``
or ``excess_return_20d``) -- a MULTI-DAY FORWARD return, not a one-day
return. Sampled at a one-day step, consecutive rows share 4/5 or 19/20 of
the same underlying price move: the series is heavily autocorrelated
(overlapping), which artificially shrinks the sample standard deviation
relative to a true independent-return series, and ``sharpe_ratio``'s
``sqrt(252)`` annualisation additionally assumes 252 INDEPENDENT
one-day compounding periods a year -- true for a real daily return
series, wrong by a factor of ~``sqrt(20)`` (~4.47x) for a series of
20-day returns sampled daily. Both effects inflate the reported Sharpe;
together they are large enough to turn a real "moderate, if any" edge
into a headline number like 6.649 or 4.09.

**The fix.** :func:`build_daily_rebalanced_portfolio_returns` builds an
actual one-row-per-TRADING-DAY series: on each date a prediction exists
("rebalance date"), it ranks symbols into a top/bottom quantile long/short
book from the model's signal; that book takes effect starting the NEXT
trading day (no look-ahead: the day a signal is observed cannot also be
the day it earns a return) and is marked to market every single
subsequent trading day using ACTUAL realised one-day price returns
(``adjusted_close`` close-to-close) -- never the multi-day forward target
used only to rank symbols. :func:`sharpe_audit_report` then computes
Sharpe (and every other statistic V0.3 Stage 2 requires) from this
genuinely chronological series, annualised with the now-legitimate
``sqrt(252)``.

Do NOT feed ``build_quantile_portfolio_returns``'s output into
``sharpe_ratio`` for a reported Sharpe number -- that function remains
useful only for the OOS rank-IC-style diagnostics it was originally built
for (``cost_stress_test``, ``execution_delay_stress_test``), which measure
relative degradation, not an absolute annualised Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.metrics import TRADING_DAYS_PER_YEAR, max_drawdown, sharpe_ratio
from backtesting.purged_walk_forward import _trading_day_offset


def build_daily_rebalanced_portfolio_returns(
    predictions_df: pd.DataFrame,
    market_df: pd.DataFrame,
    pred_col: str,
    top_frac: float = 0.2,
    price_col: str = "adjusted_close",
    extra_delay_days: int = 0,
    rebalance_threshold: float | None = None,
) -> pd.DataFrame:
    """One row per TRADING DAY (never per symbol, never per prediction
    date's multi-day-forward target). ``predictions_df`` needs
    ``symbol``/``timestamp``/``pred_col``; ``market_df`` needs
    ``symbol``/``timestamp``/``price_col`` and must cover every trading
    day between (and including) the prediction dates, not just the
    prediction dates themselves.

    Columns returned: ``timestamp``, ``gross_return`` (the day's realised
    long/short book return), ``turnover`` (0.0 on every day the book
    carries over unchanged, the fraction of the book that changed on a
    day a new rebalance takes effect), ``n_long``, ``n_short``.

    ``extra_delay_days`` (V0.3 Stage 12): additional trading days of
    execution delay ON TOP OF the built-in one-day no-look-ahead gap
    (a signal observed on date D always takes effect no earlier than
    D+1; ``extra_delay_days=1`` pushes that to D+2, simulating a slower
    fill). ``price_col`` doubles as the execution-price stress lever --
    pass ``"open"`` for a "trade at next open" assumption instead of the
    default close-to-close.

    ``rebalance_threshold`` (V0.3 Stage 12): if set, a new candidate book
    is only ADOPTED when it differs from the currently held book by at
    least this fraction of book_size (a two-sided book-replacement
    fraction, same convention as ``turnover`` below); otherwise the prior
    book is held through that rebalance date. This is what "threshold-
    based rebalance" means here -- trade only when the signal has moved
    enough to be worth the (unmodelled, see cost_bps in
    :func:`sharpe_audit_report`) transaction cost."""
    if predictions_df.empty or market_df.empty:
        return pd.DataFrame(columns=["timestamp", "gross_return", "turnover", "n_long", "n_short"])

    price_wide = market_df.dropna(subset=[price_col]).pivot_table(index="timestamp", columns="symbol", values=price_col)
    price_wide = price_wide.sort_index()
    daily_ret_wide = price_wide.pct_change()

    rebalance_books: dict[pd.Timestamp, tuple[frozenset[str], frozenset[str]]] = {}
    for date, day_preds in predictions_df.dropna(subset=[pred_col]).groupby("timestamp"):
        n = len(day_preds)
        k = max(1, int(np.floor(n * top_frac)))
        if n < 2 * k:
            continue
        ranked = day_preds.sort_values(pred_col, ascending=False)
        rebalance_books[date] = (frozenset(ranked.iloc[:k]["symbol"]), frozenset(ranked.iloc[-k:]["symbol"]))

    if not rebalance_books:
        return pd.DataFrame(columns=["timestamp", "gross_return", "turnover", "n_long", "n_short"])

    if rebalance_threshold is not None:
        filtered: dict[pd.Timestamp, tuple[frozenset[str], frozenset[str]]] = {}
        held_long: frozenset[str] = frozenset()
        held_short: frozenset[str] = frozenset()
        for date in sorted(rebalance_books):
            cand_long, cand_short = rebalance_books[date]
            if not held_long and not held_short:
                filtered[date] = (cand_long, cand_short)
                held_long, held_short = cand_long, cand_short
                continue
            book_size = max(1, len(held_long) + len(held_short))
            change_frac = len((cand_long ^ held_long) | (cand_short ^ held_short)) / book_size
            if change_frac >= rebalance_threshold:
                filtered[date] = (cand_long, cand_short)
                held_long, held_short = cand_long, cand_short
        rebalance_books = filtered
        if not rebalance_books:
            return pd.DataFrame(columns=["timestamp", "gross_return", "turnover", "n_long", "n_short"])

    if extra_delay_days:
        shifted: dict[pd.Timestamp, tuple[frozenset[str], frozenset[str]]] = {}
        for date in sorted(rebalance_books):
            shifted_date = _trading_day_offset(date, daily_ret_wide.index, extra_delay_days)
            shifted[shifted_date] = rebalance_books[date]
        rebalance_books = shifted

    rebalance_dates = sorted(rebalance_books)
    calendar = [d for d in daily_ret_wide.index if d > rebalance_dates[0]]  # returns only exist strictly after data starts

    rows: list[dict] = []
    active_long: frozenset[str] = frozenset()
    active_short: frozenset[str] = frozenset()
    prev_long: frozenset[str] = frozenset()
    prev_short: frozenset[str] = frozenset()
    book_idx = -1  # index into rebalance_dates of the most recent rebalance date STRICTLY BEFORE the current calendar day

    for date in calendar:
        book_changed_today = False
        while book_idx + 1 < len(rebalance_dates) and rebalance_dates[book_idx + 1] < date:
            book_idx += 1
            active_long, active_short = rebalance_books[rebalance_dates[book_idx]]
            book_changed_today = True

        if book_idx < 0 or date not in daily_ret_wide.index:
            continue

        long_syms = [s for s in active_long if s in daily_ret_wide.columns]
        short_syms = [s for s in active_short if s in daily_ret_wide.columns]
        if not long_syms or not short_syms:
            continue
        long_ret = daily_ret_wide.loc[date, long_syms].mean()
        short_ret = daily_ret_wide.loc[date, short_syms].mean()
        if pd.isna(long_ret) or pd.isna(short_ret):
            continue

        gross_return = 0.5 * float(long_ret) - 0.5 * float(short_ret)
        book_size = max(1, len(active_long) + len(active_short))
        if book_changed_today and (prev_long or prev_short):
            turnover = len((active_long ^ prev_long) | (active_short ^ prev_short)) / book_size
        elif book_changed_today:
            turnover = 1.0  # the very first day a book takes effect -- the whole book is "new"
        else:
            turnover = 0.0  # holding the same book from the prior day, no trades today
        rows.append(
            {
                "timestamp": date, "gross_return": gross_return, "turnover": turnover,
                "n_long": len(active_long), "n_short": len(active_short),
            }
        )
        prev_long, prev_short = active_long, active_short

    return pd.DataFrame(rows)


def sharpe_audit_report(
    portfolio_returns: pd.DataFrame, cost_bps: float = 10.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> dict:
    """The exact statistics V0.3 Stage 2 requires, computed from an
    ``build_daily_rebalanced_portfolio_returns``-shaped chronological
    daily return series -- never from a multi-day-overlapping series."""
    if portfolio_returns.empty:
        return {
            "n_observations": 0, "date_range": None, "mean_daily_return": float("nan"),
            "daily_volatility": float("nan"), "annualization_factor": float(np.sqrt(periods_per_year)),
            "gross_sharpe": float("nan"), "net_sharpe": float("nan"), "mean_turnover": float("nan"),
            "cost_bps_assumption": cost_bps, "total_cost_drag_annualized": float("nan"), "max_drawdown": float("nan"),
        }

    gross = portfolio_returns["gross_return"]
    turnover = portfolio_returns["turnover"]
    cost = turnover * (cost_bps / 10_000.0)
    net = gross - cost
    equity = (1.0 + gross).cumprod()

    return {
        "n_observations": int(len(portfolio_returns)),
        "date_range": [str(portfolio_returns["timestamp"].min()), str(portfolio_returns["timestamp"].max())],
        "mean_daily_return": float(gross.mean()),
        "daily_volatility": float(gross.std(ddof=1)) if len(gross) > 1 else float("nan"),
        "annualization_factor": float(np.sqrt(periods_per_year)),
        "gross_sharpe": sharpe_ratio(gross, periods_per_year=periods_per_year),
        "net_sharpe": sharpe_ratio(net, periods_per_year=periods_per_year),
        "mean_turnover": float(turnover.mean()),
        "cost_bps_assumption": cost_bps,
        "total_cost_drag_annualized": float(cost.mean() * periods_per_year),
        "max_drawdown": max_drawdown(equity),
    }
