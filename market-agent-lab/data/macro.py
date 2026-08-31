"""Macro data access layer -- always as-of, respecting publication/vintage timestamps."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from database import repository as repo


def get_macro_asof(con: duckdb.DuckDBPyConnection, as_of: datetime) -> pd.DataFrame:
    return repo.get_macro_asof(con, as_of)


def get_macro_history_asof(con: duckdb.DuckDBPyConnection, as_of: datetime) -> pd.DataFrame:
    return repo.get_macro_history_asof(con, as_of)


def pivot_macro_wide(macro_df: pd.DataFrame) -> dict[str, float]:
    """Collapse a long macro-as-of frame into {series_name: value}."""
    if macro_df.empty:
        return {}
    return dict(zip(macro_df["series_name"], macro_df["value"], strict=True))
