"""Company investor-relations RSS feeds (Phase 8).

There is no universal free registry mapping symbol -> IR RSS feed URL;
each company publishes its own (if any). This provider is a thin,
functional RSS client, but requires an explicit ``{symbol: feed_url}``
mapping to be supplied by the caller -- it never guesses a URL. Disabled
by default in the registry (``is_enabled=False``) since no such mapping
ships with v0.2; a deployment that wants this source configures
``COMPANY_IR_FEEDS`` (symbol -> URL) and enables it explicitly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

from core.schemas_v2 import NewsArticle, NewsTier
from data.providers.base import RateLimitedClient

SOURCE_ID = "company_ir"


class CompanyIrProvider:
    source_id = SOURCE_ID

    def __init__(self, feed_urls: dict[str, str], client: RateLimitedClient | None = None) -> None:
        self.feed_urls = feed_urls
        self.client = client or RateLimitedClient()

    def get_releases(self, symbol: str, start: datetime, end: datetime) -> list[NewsArticle]:
        feed_url = self.feed_urls.get(symbol)
        if not feed_url:
            return []
        response = self.client.get(feed_url, source_id=self.source_id)
        root = ET.fromstring(response.content)  # noqa: S314 - trusted, explicitly-configured feed URL
        retrieved_at = datetime.now(UTC)

        articles = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date_raw = item.findtext("pubDate")
            published_at = None
            if pub_date_raw:
                try:
                    published_at = parsedate_to_datetime(pub_date_raw).replace(tzinfo=None)
                except (TypeError, ValueError):
                    published_at = None
            if published_at and (published_at < start or published_at > end):
                continue
            articles.append(
                NewsArticle(
                    headline=title, published_at=published_at, retrieved_at=retrieved_at,
                    source=self.source_id, publisher=symbol, tier=NewsTier.TIER_1_OFFICIAL,
                    url=link, timestamp_uncertain=published_at is None,
                    dedupe_key=f"company_ir::{link}", symbols=[symbol],
                )
            )
        return articles
