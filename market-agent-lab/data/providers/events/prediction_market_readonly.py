"""Read-only public prediction-market provider: Polymarket ``gamma-api``
(Phase 9/18-19). HARD SAFETY BOUNDARY.

This class may ONLY read publicly exposed market-probability data. It has
NO method, anywhere, for placing an order, creating an account, logging
in, holding a wallet balance, or moving funds -- and
``tests/test_prediction_market_readonly.py`` enforces this structurally:
it asserts the class's complete public API is exactly the allow-list
below, so adding an execution-shaped method anywhere in this class would
fail that test, not just a manual code-review check.

Every observed probability is exactly that: an observed market price on a
public exchange, subject to noise, thin liquidity, and manipulation risk
-- never treated as "the true probability" anywhere downstream (see
``docs/data_sources.md`` and ``agents/event_relevance.py``).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from core.schemas_v2 import EventProbabilityObservation, ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "polymarket_readonly"
BASE_URL = "https://gamma-api.polymarket.com"

# The complete, exhaustive set of public (non-dunder) attributes this class
# may ever expose. tests/test_prediction_market_readonly.py fails the
# build if this set ever contains anything execution-shaped, and fails if
# the live class exposes anything beyond this set.
ALLOWED_PUBLIC_API: frozenset[str] = frozenset({"source_id", "get_active_events"})

# Deterministic keyword -> research-category classification (never an LLM
# guess) -- see brief section 18's example categories.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "monetary_policy": ["fed ", "fomc", "interest rate", "rate cut", "rate hike", "powell"],
    "economic_outcomes": ["recession", "gdp", "unemployment", "inflation", "cpi"],
    "elections_policy": ["election", "president", "senate", "congress", "governor"],
    "geopolitical": ["war", "sanctions", "invasion", "ceasefire", "geopolitical"],
    "regulatory": ["sec ", "antitrust", "regulation", "ftc", "lawsuit"],
}


def _classify_category(title: str) -> str:
    lowered = title.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return "other"


class PredictionMarketReadOnlyProvider:
    """Public read-only market-probability data. See module docstring --
    do not add any method here without updating (and justifying, to a
    human reviewer) ALLOWED_PUBLIC_API and the safety test."""

    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self._client = client or RateLimitedClient()  # underscore: not part of the public API surface

    def get_active_events(self, limit: int = 100, category_filter: str | None = None) -> list[EventProbabilityObservation]:
        """Fetch currently active public markets ranked by volume. This
        performs a GET request only -- no request body, no credentials,
        no wallet/session state of any kind."""
        params = {"limit": str(limit), "closed": "false", "order": "volume", "ascending": "false"}

        def fetch() -> Any:
            response = self._client.get(f"{BASE_URL}/events", source_id=self.source_id, params=params)
            return response.json()

        try:
            payload = cached_fetch(namespace="polymarket_events", params=params, fetch_fn=fetch, max_age_seconds=3600)
            HEALTH.record_success(self.source_id, ProviderCategory.EVENT_PROBABILITY, records=len(payload), latency_ms=0.0)
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.EVENT_PROBABILITY, str(exc))
            raise

        retrieved_at = datetime.now(UTC)
        observations: list[EventProbabilityObservation] = []
        for event in payload:
            for market in event.get("markets", []):
                title = market.get("question") or event.get("title", "")
                category = _classify_category(title)
                if category_filter and category != category_filter:
                    continue
                try:
                    outcomes = json.loads(market.get("outcomes", "[]"))
                    prices = json.loads(market.get("outcomePrices", "[]"))
                except (json.JSONDecodeError, TypeError):
                    continue
                if not outcomes or not prices or outcomes[0].strip().lower() != "yes":
                    continue
                probability = float(prices[0])
                end_date_raw = market.get("endDateIso") or market.get("endDate")
                resolution_date = None
                if end_date_raw:
                    try:
                        resolution_date = datetime.fromisoformat(re.sub("Z$", "+00:00", end_date_raw)).replace(tzinfo=None)
                    except ValueError:
                        resolution_date = None

                observations.append(
                    EventProbabilityObservation(
                        event_id=str(market.get("id")), question=title, category=category,
                        observed_timestamp=retrieved_at.replace(tzinfo=None), resolution_date=resolution_date,
                        public_probability=max(0.0, min(1.0, probability)),
                        liquidity_metadata={"liquidity": float(market.get("liquidityNum", 0.0) or 0.0)},
                        volume_metadata={"volume": float(market.get("volumeNum", 0.0) or 0.0), "volume_24hr": float(market.get("volume24hr", 0.0) or 0.0)},
                        source=self.source_id, retrieved_at=retrieved_at,
                    )
                )
        return observations
