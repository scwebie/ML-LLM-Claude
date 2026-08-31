"""Outcome labelling: turn completed predictions into realised ``Outcome`` rows.

A prediction can only be labelled once its 20-day horizon has actually
elapsed in the market data -- this module never estimates or interpolates
an outcome, it only reads the realised forward return once it exists.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from core.schemas import Outcome
from database import repository as repo


def _price_index_lookup(market_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: group.sort_values("timestamp").reset_index(drop=True)
        for symbol, group in market_df.groupby("symbol")
    }


def label_pending_outcomes(
    con: duckdb.DuckDBPyConnection, market_df: pd.DataFrame, benchmark_df: pd.DataFrame, as_of: pd.Timestamp
) -> int:
    """Label every prediction whose 20-day horizon has completed by ``as_of``."""
    pending = repo.get_predictions_without_outcome(con, horizon_days=28, as_of=as_of)
    if pending.empty:
        return 0

    by_symbol = _price_index_lookup(market_df)
    bench = benchmark_df.sort_values("timestamp").reset_index(drop=True)
    bench_index = pd.Series(bench.index.values, index=bench["timestamp"])

    labelled = 0
    for _, pred in pending.iterrows():
        symbol_hist = by_symbol.get(pred["symbol"])
        if symbol_hist is None:
            continue
        ts_matches = symbol_hist.index[symbol_hist["timestamp"] == pred["timestamp"]]
        if len(ts_matches) == 0:
            continue
        idx = ts_matches[0]
        if idx + 20 >= len(symbol_hist):
            continue  # not enough forward data yet

        bench_idx = bench_index.get(pred["timestamp"])
        if bench_idx is None or bench_idx + 20 >= len(bench):
            continue

        close = symbol_hist["close"]
        bench_close = bench["close"]

        stock_5d = close.iloc[idx + 5] / close.iloc[idx] - 1.0
        stock_20d = close.iloc[idx + 20] / close.iloc[idx] - 1.0
        bench_5d = bench_close.iloc[bench_idx + 5] / bench_close.iloc[bench_idx] - 1.0
        bench_20d = bench_close.iloc[bench_idx + 20] / bench_close.iloc[bench_idx] - 1.0

        window_returns = close.iloc[idx : idx + 21].pct_change().dropna()
        realised_vol = float(window_returns.std() * (252 ** 0.5)) if len(window_returns) > 1 else None

        outcome = Outcome(
            prediction_id=pred["prediction_id"],
            realised_excess_return_5d=float(stock_5d - bench_5d),
            realised_excess_return_20d=float(stock_20d - bench_20d),
            realised_volatility=realised_vol,
            completion_timestamp=symbol_hist["timestamp"].iloc[idx + 20],
        )
        try:
            repo.insert_outcome(con, outcome)
            labelled += 1
        except ValueError:
            continue  # already labelled
    return labelled


def get_recent_outcomes(con: duckdb.DuckDBPyConnection, model_version: str | None = None) -> pd.DataFrame:
    query = """
        SELECT p.*, o.realised_excess_return_5d, o.realised_excess_return_20d,
               o.realised_volatility, o.completion_timestamp
        FROM model_predictions p JOIN outcomes o ON p.prediction_id = o.prediction_id
    """
    params: list = []
    if model_version:
        query += " WHERE p.model_version = ?"
        params.append(model_version)
    return con.execute(query, params).fetchdf()
