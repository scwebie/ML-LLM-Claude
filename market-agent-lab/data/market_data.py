"""Market data access layer.

Version 0.1 only ever sources OHLCV data from the deterministic synthetic
generator (``data/synthetic.py``). The functions here are intentionally the
*only* place the rest of the codebase touches market data, so a future
version could swap this module for a real (delayed, licensed) data vendor
without changing agents, features, or the model pipeline.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from data.synthetic import BENCHMARK_SYMBOL, generate_synthetic_dataset, write_synthetic_dataset
from database import repository as repo


def load_all_synthetic_data(con: duckdb.DuckDBPyConnection, seed: int, start_date: str, end_date: str, out_dir) -> dict[str, int]:
    """Generate the synthetic universe and load every table into DuckDB."""
    dataset = generate_synthetic_dataset(seed=seed, start_date=start_date, end_date=end_date)
    write_synthetic_dataset(dataset, out_dir)
    counts = {
        "market": repo.insert_market_observations(con, dataset.market),
        "fundamentals": repo.insert_fundamental_observations(con, dataset.fundamentals),
        "macro": repo.insert_macro_observations(con, dataset.macro),
        "news": repo.insert_news_observations(con, dataset.news),
    }
    return counts


def get_symbols(con: duckdb.DuckDBPyConnection, include_benchmark: bool = False) -> list[str]:
    rows = con.execute("SELECT DISTINCT symbol FROM market_observations ORDER BY symbol").fetchall()
    symbols = [r[0] for r in rows]
    if not include_benchmark:
        symbols = [s for s in symbols if s != BENCHMARK_SYMBOL]
    return symbols


def get_ohlcv(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    return repo.get_market_observations(con, symbols=symbols, start=start, end=end)


def get_benchmark(con: duckdb.DuckDBPyConnection, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    return repo.get_market_observations(con, symbols=[BENCHMARK_SYMBOL], start=start, end=end)
