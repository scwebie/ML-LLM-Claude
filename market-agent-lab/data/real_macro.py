"""Real macro ingestion orchestrator (Phase 6/9): FRED + BLS + Treasury
(+ BEA when a key is configured), written into the same
``macro_observations`` table V0.1's synthetic macro data used -- it
already carries ``publication_timestamp``/``vintage_timestamp``, exactly
what real vintage-aware macro data needs.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from core.schemas_v2 import IngestionStatus, ProviderCategory
from data.providers import registry
from data.providers.base import finish_ingestion_run, make_ingestion_run
from data.providers.macro.bls import SERIES_CATALOG as BLS_CATALOG
from data.providers.macro.bls import BlsProvider
from data.providers.macro.fred import SERIES_CATALOG as FRED_CATALOG
from data.providers.macro.fred import FredProvider
from data.providers.macro.treasury import TreasuryProvider
from database import repository as repo
from database import repository_v2 as repo_v2

DEFAULT_FRED_SERIES = ["DGS10", "DGS2", "DGS3MO", "T10Y2Y", "T10Y3M", "FEDFUNDS", "VIXCLS"]
DEFAULT_BLS_SERIES = ["CUUR0000SA0", "CUUR0000SA0L1E", "LNS14000000", "CES0000000001"]
DEFAULT_TREASURY_SECURITIES = ["Treasury Notes"]


def ingest_macro(con: duckdb.DuckDBPyConnection, start: datetime, end: datetime) -> dict:
    summary: dict[str, dict] = {}
    all_rows: list[dict] = []

    if registry.get_source("fred") and registry.get_source("fred").is_enabled:
        provider = FredProvider()
        for series_id in DEFAULT_FRED_SERIES:
            run = make_ingestion_run(provider.source_id, ProviderCategory.MACRO)
            try:
                rows = provider.get_series(series_id, start, end)
            except Exception as exc:  # noqa: BLE001
                repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.FAILED, error=str(exc)))
                summary[f"fred:{series_id}"] = {"status": "FAILED", "error": str(exc)}
                continue
            name = FRED_CATALOG.get(series_id)
            series_name = f"FRED_{series_id}"
            for r in rows:
                all_rows.append(
                    {
                        "series_name": series_name, "timestamp": r["date"], "value": r["value"],
                        "publication_timestamp": r["publication_date"], "vintage_timestamp": r["publication_date"],
                    }
                )
            repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.SUCCESS, records=len(rows)))
            summary[f"fred:{series_id}"] = {"status": "SUCCESS", "records": len(rows), "name": name.name if name else series_id}
    else:
        summary["fred"] = {"status": "UNAVAILABLE"}

    if registry.get_source("bls") and registry.get_source("bls").is_enabled:
        provider = BlsProvider()
        for series_id in DEFAULT_BLS_SERIES:
            run = make_ingestion_run(provider.source_id, ProviderCategory.MACRO)
            try:
                rows = provider.get_series(series_id, start, end)
            except Exception as exc:  # noqa: BLE001
                repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.FAILED, error=str(exc)))
                summary[f"bls:{series_id}"] = {"status": "FAILED", "error": str(exc)}
                continue
            name = BLS_CATALOG.get(series_id)
            series_name = f"BLS_{series_id}"
            for r in rows:
                all_rows.append(
                    {
                        "series_name": series_name, "timestamp": r["date"], "value": r["value"],
                        "publication_timestamp": r["publication_date"], "vintage_timestamp": r["publication_date"],
                    }
                )
            repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.SUCCESS, records=len(rows)))
            summary[f"bls:{series_id}"] = {
                "status": "SUCCESS", "records": len(rows), "name": name.name if name else series_id,
                "skipped_missing": provider.last_skipped,
            }
    else:
        summary["bls"] = {"status": "UNAVAILABLE"}

    if registry.get_source("treasury") and registry.get_source("treasury").is_enabled:
        provider = TreasuryProvider()
        for security in DEFAULT_TREASURY_SECURITIES:
            run = make_ingestion_run(provider.source_id, ProviderCategory.MACRO)
            try:
                rows = provider.get_series(security, start, end)
            except Exception as exc:  # noqa: BLE001
                repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.FAILED, error=str(exc)))
                summary[f"treasury:{security}"] = {"status": "FAILED", "error": str(exc)}
                continue
            series_name = f"TREASURY_{security.replace(' ', '_').upper()}"
            for r in rows:
                all_rows.append(
                    {
                        "series_name": series_name, "timestamp": r["date"], "value": r["value"],
                        "publication_timestamp": r["publication_date"], "vintage_timestamp": r["publication_date"],
                    }
                )
            repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.SUCCESS, records=len(rows)))
            summary[f"treasury:{security}"] = {"status": "SUCCESS", "records": len(rows)}
    else:
        summary["treasury"] = {"status": "UNAVAILABLE"}

    bea_source = registry.get_source("bea")
    summary["bea"] = {"status": "UNAVAILABLE", "reason": bea_source.notes if bea_source else "not registered"}

    if all_rows:
        df = pd.DataFrame(all_rows)
        n = repo.insert_macro_observations(con, df)
        summary["_total_observations_written"] = n

    return summary
