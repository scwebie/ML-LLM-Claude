"""SEC 8-K filings as a real, official (Tier-1) news/event source (Phase 8).

Reuses the same cached SEC submissions payload
``data/providers/fundamentals/sec.py`` already fetches (same cache
namespace/params -- this incurs zero extra network calls when both are
used together, as they are in ``data/real_news.py``).

``published_at`` is the filing's ``acceptanceDateTime`` when available
(full timestamp precision -- exactly when the filing became public) and
falls back to the filing date (midnight) otherwise, in which case the
article is marked ``timestamp_uncertain`` per the brief's strict rule.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.schemas_v2 import NewsArticle, NewsTier
from data.providers.base import cached_fetch
from data.providers.fundamentals.sec import BASE_URL, WWW_BASE_URL, SecEdgarFundamentalProvider

SOURCE_ID = "sec_events"

# SEC Form 8-K item numbers -> our event_category taxonomy (brief section 17).
ITEM_CATEGORY_MAP: dict[str, str] = {
    "1.01": "M&A", "1.02": "M&A", "1.03": "litigation", "2.01": "M&A",
    "2.02": "earnings", "2.03": "credit", "2.04": "credit", "2.05": "guidance",
    "2.06": "other", "3.01": "regulation", "3.02": "other", "3.03": "other",
    "4.01": "other", "4.02": "other", "5.01": "M&A", "5.02": "management",
    "5.03": "other", "5.07": "other", "6.01": "product", "7.01": "guidance",
    "8.01": "other", "9.01": "other",
}


class SecEventsProvider:
    source_id = SOURCE_ID

    def __init__(self, sec_provider: SecEdgarFundamentalProvider | None = None) -> None:
        self.sec = sec_provider or SecEdgarFundamentalProvider()

    def get_events(self, symbol: str, start: datetime, end: datetime) -> list[NewsArticle]:
        cik = self.sec.resolve_cik(symbol)
        if cik is None:
            return []

        def fetch() -> dict[str, Any]:
            response = self.sec.client.get(f"{BASE_URL}/submissions/CIK{cik}.json", source_id=self.source_id)
            return response.json()

        payload = cached_fetch(namespace="sec_submissions", params={"cik": cik}, fetch_fn=fetch, max_age_seconds=24 * 3600)
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accepted = recent.get("acceptanceDateTime", [])
        accns = recent.get("accessionNumber", [])
        items_list = recent.get("items", [])
        retrieved_at = datetime.now(UTC)

        articles: list[NewsArticle] = []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
            if filing_date < start or filing_date > end:
                continue

            accepted_raw = accepted[i] if i < len(accepted) else None
            timestamp_uncertain = not bool(accepted_raw)
            published_at = datetime.fromisoformat(accepted_raw).replace(tzinfo=None) if accepted_raw else filing_date

            items = items_list[i] if i < len(items_list) else ""
            item_codes = [c.strip() for c in items.split(",") if c.strip()]
            categories = {ITEM_CATEGORY_MAP.get(c, "other") for c in item_codes} or {"other"}
            primary_category = sorted(categories)[0]

            headline = f"{symbol} 8-K filing" + (f" (items {items})" if items else "")
            accn = accns[i]
            articles.append(
                NewsArticle(
                    headline=headline, published_at=published_at, retrieved_at=retrieved_at,
                    source=self.source_id, publisher="U.S. Securities and Exchange Commission",
                    tier=NewsTier.TIER_1_OFFICIAL, url=f"{WWW_BASE_URL}/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}",
                    event_category=primary_category, language="en", excerpt=None,
                    timestamp_uncertain=timestamp_uncertain,
                    dedupe_key=f"sec_8k::{accn}", symbols=[symbol],
                )
            )
        return articles
