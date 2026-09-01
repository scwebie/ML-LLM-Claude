"""NewsAPI.org provider (Phase 8). Requires ``NEWS_API_KEY``; disabled by
the registry without one."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from core.schemas_v2 import NewsArticle, NewsTier
from data.providers.base import RateLimitedClient, cached_fetch

SOURCE_ID = "news_api"
BASE_URL = "https://newsapi.org/v2"


class NewsApiProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient()

    def search_news(self, query: str, start: datetime, end: datetime) -> list[NewsArticle]:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            raise RuntimeError("NEWS_API_KEY is not configured; register a key at https://newsapi.org and set it to enable this provider.")
        params = {"q": query, "from": start.date().isoformat(), "to": end.date().isoformat(), "language": "en", "sortBy": "publishedAt", "apiKey": api_key}

        def fetch() -> dict[str, Any]:
            response = self.client.get(f"{BASE_URL}/everything", source_id=self.source_id, params=params)
            return response.json()

        payload = cached_fetch(namespace="news_api_search", params={"q": query, "from": start.isoformat(), "to": end.isoformat()}, fetch_fn=fetch)
        retrieved_at = datetime.now(UTC)

        articles = []
        for row in payload.get("articles", []):
            published_raw = row.get("publishedAt")
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).replace(tzinfo=None) if published_raw else None
            articles.append(
                NewsArticle(
                    headline=row.get("title", ""), published_at=published_at, retrieved_at=retrieved_at,
                    source=self.source_id, publisher=(row.get("source") or {}).get("name"),
                    tier=NewsTier.TIER_2_FINANCIAL_MEDIA, url=row.get("url"), excerpt=row.get("description"),
                    timestamp_uncertain=published_at is None, dedupe_key=f"news_api::{row.get('url')}", symbols=[],
                )
            )
        return articles
