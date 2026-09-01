"""Read-only public event-probability ingestion orchestrator (Phase 9/18).

Writes to its own namespaced tables (``event_probability_observations``,
``event_symbol_mappings``) -- entirely separate from prices/fundamentals/
macro/news, so this signal can be fully removed from the feature matrix
for ablation testing (the ``eventprob_*`` feature-name convention; see
``features/event_probability_features.py``).
"""

from __future__ import annotations

import duckdb

from agents.event_relevance import compute_event_relevance_for_universe
from core.schemas_v2 import IngestionStatus, ProviderCategory
from data.providers import registry
from data.providers.base import finish_ingestion_run, make_ingestion_run
from data.providers.events.prediction_market_readonly import PredictionMarketReadOnlyProvider
from database import repository_v2 as repo_v2


def ingest_event_probabilities(con: duckdb.DuckDBPyConnection, symbols: list[str], sector_map: dict[str, str], limit: int = 200) -> dict:
    source = registry.get_source("polymarket_readonly")
    if not source or not source.is_enabled:
        return {"status": "UNAVAILABLE", "reason": source.notes if source else "not registered"}

    provider = PredictionMarketReadOnlyProvider()
    run = make_ingestion_run(provider.source_id, ProviderCategory.EVENT_PROBABILITY)
    try:
        observations = provider.get_active_events(limit=limit)
    except Exception as exc:  # noqa: BLE001
        repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.FAILED, error=str(exc)))
        return {"status": "FAILED", "error": str(exc)}

    relevant = [o for o in observations if o.category != "other"]
    repo_v2.insert_event_probability_observations(con, relevant)

    mappings = compute_event_relevance_for_universe(relevant, symbols, sector_map)
    repo_v2.insert_event_symbol_mappings(con, mappings)

    repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.SUCCESS, records=len(relevant)))
    return {
        "status": "SUCCESS", "total_events_fetched": len(observations),
        "relevant_events": len(relevant), "symbol_mappings": len(mappings),
        "categories": sorted({o.category for o in relevant}),
    }
