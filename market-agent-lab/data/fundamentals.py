"""Fundamentals access layer -- always as-of, never period-end, to avoid look-ahead."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from database import repository as repo


def get_fundamentals_asof(con: duckdb.DuckDBPyConnection, symbols: list[str], as_of: datetime) -> pd.DataFrame:
    """Most recent fundamentals *known* as of ``as_of`` (publication-time filtered)."""
    return repo.get_fundamentals_asof(con, symbols, as_of)
