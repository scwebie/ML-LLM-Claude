"""GDELT Doc API news provider (Phase 8).

Implemented as a genuine, correct HTTP client against GDELT's real,
documented, keyless endpoint. In this deployment's network environment
``api.gdeltproject.org`` is unreachable (connection reset on every probe
during development -- see ``data/providers/registry.py``'s
``KNOWN_UNAVAILABLE`` and ``docs/data_sources.md``), so it is registered
disabled by default. If the host becomes reachable, flip
``is_enabled=True`` in the registry and this class works unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.schemas_v2 import NewsArticle, NewsTier, ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "gdelt"
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


class GdeltNewsProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient()

    def search_news(self, query: str, start: datetime, end: datetime) -> list[NewsArticle]:
        params = {
            "query": query, "mode": "artlist", "format": "json", "maxrecords": "250",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"), "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }

        def fetch() -> dict[str, Any]:
            response = self.client.get(BASE_URL, source_id=self.source_id, params=params)
            return response.json()

        try:
            payload = cached_fetch(namespace="gdelt_doc", params={"query": query, "start": start.isoformat(), "end": end.isoformat()}, fetch_fn=fetch)
            HEALTH.record_success(self.source_id, ProviderCategory.NEWS, records=len(payload.get("articles", [])), latency_ms=0.0)
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.NEWS, str(exc))
            raise

        retrieved_at = datetime.now(UTC)
        articles = []
        for row in payload.get("articles", []):
            seendate = row.get("seendate")
            published_at = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ") if seendate else None
            articles.append(
                NewsArticle(
                    headline=row.get("title", ""), published_at=published_at, retrieved_at=retrieved_at,
                    source=self.source_id, publisher=row.get("domain"), tier=NewsTier.TIER_2_FINANCIAL_MEDIA,
                    url=row.get("url"), language=row.get("language"),
                    timestamp_uncertain=published_at is None,
                    dedupe_key=f"gdelt::{row.get('url')}", symbols=[],
                )
            )
        return articles
