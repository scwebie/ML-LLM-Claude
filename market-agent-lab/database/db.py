"""DuckDB connection management.

Version 0.1 uses an embedded DuckDB file as local storage. Every function
that touches the database goes through :func:`get_connection`, and all SQL
is plain ANSI-ish SQL with no DuckDB-only extensions in the schema itself,
so a future migration to PostgreSQL/TimescaleDB only needs a new
``db.py`` (same ``repository.py`` call sites, different connection object
and a driver-appropriate DDL/DML dialect).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import duckdb

from core.config import settings
from database.schema import init_schema

_lock = threading.Lock()
_connection: duckdb.DuckDBPyConnection | None = None


def get_connection(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Return a process-wide singleton DuckDB connection, creating schema on first use."""
    global _connection
    with _lock:
        if _connection is None:
            path = Path(db_path) if db_path is not None else settings.duckdb_path
            path.parent.mkdir(parents=True, exist_ok=True)
            _connection = duckdb.connect(str(path))
            init_schema(_connection)
        return _connection


def reset_connection() -> None:
    """Close and drop the cached connection (mainly for tests)."""
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None


@contextmanager
def fresh_connection(db_path: Path | str = ":memory:"):
    """Yield a brand-new, isolated in-memory DuckDB connection with schema applied.

    Useful for tests that must not share state with the on-disk database.
    """
    con = duckdb.connect(str(db_path))
    try:
        init_schema(con)
        yield con
    finally:
        con.close()
