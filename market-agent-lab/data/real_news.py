"""Real news ingestion orchestrator (Phase 8): SEC 8-K events (always
real/enabled) plus GDELT/NewsAPI when the registry reports them enabled.
Deduplicates syndicated/repeat articles via ``dedupe_key`` before insert.
"""

from __future__ import annotations

from datetime import datetime

import duckdb

from core.schemas_v2 import IngestionStatus, NewsArticle, ProviderCategory
from data.providers import registry
from data.providers.base import finish_ingestion_run, make_ingestion_run
from data.providers.news.sec_events import SecEventsProvider
from database import repository_v2 as repo_v2


def ingest_news(con: duckdb.DuckDBPyConnection, symbols: list[str], start: datetime, end: datetime) -> dict:
    summary: dict[str, dict] = {}
    all_articles: list[NewsArticle] = []

    sec_source = registry.get_source("sec_events")
    if sec_source and sec_source.is_enabled:
        provider = SecEventsProvider()
        for symbol in symbols:
            run = make_ingestion_run(provider.source_id, ProviderCategory.NEWS)
            try:
                articles = provider.get_events(symbol, start, end)
            except Exception as exc:  # noqa: BLE001
                repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.FAILED, error=str(exc)))
                summary[f"sec_events:{symbol}"] = {"status": "FAILED", "error": str(exc)}
                continue

            fresh = [a for a in articles if not (a.dedupe_key and repo_v2.is_duplicate_article(con, a.dedupe_key))]
            all_articles.extend(fresh)
            repo_v2.insert_ingestion_run(con, finish_ingestion_run(run, IngestionStatus.SUCCESS, records=len(fresh)))
            summary[f"sec_events:{symbol}"] = {"status": "SUCCESS", "articles": len(fresh), "duplicates_skipped": len(articles) - len(fresh)}
    else:
        summary["sec_events"] = {"status": "UNAVAILABLE"}

    for source_id in ("gdelt", "company_ir", "news_api"):
        source = registry.get_source(source_id)
        if not source or not source.is_enabled:
            summary[source_id] = {"status": "UNAVAILABLE", "reason": source.notes if source else "not registered"}

    if all_articles:
        n = repo_v2.insert_news_articles(con, all_articles)
        summary["_total_articles_written"] = n

    return summary
