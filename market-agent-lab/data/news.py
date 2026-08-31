"""News / event data access layer (synthetic sentiment in v0.1)."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from database import repository as repo


def get_news(con: duckdb.DuckDBPyConnection, symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    return repo.get_news_observations(con, symbols, start, end)
