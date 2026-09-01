"""Real price ingestion orchestrator (Phase 3): fetch from primary +
secondary providers, reconcile, and persist -- the real-data analogue of
``data/market_data.py``'s synthetic loader.

``market_observations`` (the same V0.1 table) is the canonical value
store for both synthetic and real data; what changes is which provider
populated it, tracked via ``data_ingestion_runs`` and
``price_reconciliation``, never a column on the hot table itself.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from core.schemas_v2 import IngestionStatus, ProviderCategory
from data.providers.base import finish_ingestion_run, make_ingestion_run
from data.providers.prices.primary import YahooFinancePriceProvider
from data.providers.prices.reconciliation import ReconciliationTolerance, reconcile_bar_sets
from data.providers.prices.secondary import StockAnalysisPriceProvider
from database import repository as repo
from database import repository_v2 as repo_v2

REAL_BENCHMARK_SYMBOL = "SPY"


def _bars_to_dict(bars: list[dict]) -> dict[datetime, dict]:
    return {b["date"]: b for b in bars}


def ingest_prices(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: datetime,
    end: datetime,
    tolerance: ReconciliationTolerance | None = None,
    allow_major_difference: bool = False,
) -> dict:
    """Fetch, reconcile, and persist real daily bars for ``symbols``
    (which should include the benchmark, e.g. SPY). Returns a summary
    dict; never fabricates a bar for a date neither provider has."""
    primary = YahooFinancePriceProvider()
    secondary = StockAnalysisPriceProvider()

    summary = {"symbols": {}, "total_bars_written": 0, "total_reconciliations": 0}
    all_canonical_rows: list[dict] = []
    all_reconciliations = []

    for symbol in symbols:
        run = make_ingestion_run(primary.source_id, ProviderCategory.PRICE)
        primary_bars: dict = {}
        secondary_bars: dict = {}
        primary_error: str | None = None
        secondary_error: str | None = None

        try:
            primary_bars = _bars_to_dict(primary.get_daily_bars(symbol, start, end))
        except Exception as exc:  # noqa: BLE001
            primary_error = str(exc)

        try:
            secondary_bars = _bars_to_dict(secondary.get_daily_bars(symbol, start, end))
        except Exception as exc:  # noqa: BLE001
            secondary_error = str(exc)

        if not primary_bars and not secondary_bars:
            repo_v2.insert_ingestion_run(
                con,
                finish_ingestion_run(run, IngestionStatus.FAILED, records=0, error=f"primary={primary_error}; secondary={secondary_error}"),
            )
            summary["symbols"][symbol] = {"status": "FAILED", "bars": 0, "primary_error": primary_error, "secondary_error": secondary_error}
            continue

        canonical, reconciliations = reconcile_bar_sets(
            symbol, primary.source_id, primary_bars, secondary.source_id, secondary_bars, tolerance=tolerance
        )
        from data.providers.prices.reconciliation import filter_trainable_bars

        trainable = filter_trainable_bars(canonical, allow_major_difference=allow_major_difference)

        for row in trainable:
            all_canonical_rows.append(
                {
                    "symbol": symbol, "timestamp": row["date"], "open": row["open"], "high": row["high"],
                    "low": row["low"], "close": row["close"], "adjusted_close": row["adjusted_close"],
                    "volume": row["volume"],
                }
            )
        all_reconciliations.extend(reconciliations)

        status = IngestionStatus.SUCCESS if (primary_bars and secondary_bars) else IngestionStatus.PARTIAL
        repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, status, records=len(trainable)))
        summary["symbols"][symbol] = {
            "status": status.value, "bars": len(trainable),
            "excluded_major_difference": len(canonical) - len(trainable),
        }

    if all_canonical_rows:
        df = pd.DataFrame(all_canonical_rows)
        n = repo.insert_market_observations(con, df)
        summary["total_bars_written"] = n

    if all_reconciliations:
        repo_v2.insert_price_reconciliations(con, all_reconciliations)
        summary["total_reconciliations"] = len(all_reconciliations)

    return summary
