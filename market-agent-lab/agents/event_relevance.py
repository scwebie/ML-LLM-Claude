"""Explicit, structured event->symbol relevance mapping (Phase 18).

No LLM is involved anywhere in this module. Relevance is assigned purely
from a fixed, documented rule table keyed on the event's deterministically
classified category (see
``data/providers/events/prediction_market_readonly.py::_classify_category``)
and the symbol's configured sector -- never invented per-symbol by a
model. This is what keeps the read-only event-probability signal
inspectable and ablatable: every relevance score traces back to one row
in ``RELEVANCE_RULES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from core.schemas_v2 import EventProbabilityObservation, EventSymbolMapping


@dataclass(frozen=True)
class RelevanceRule:
    base_relevance: float
    sector_bonus: dict[str, float]
    rationale_category: str


RELEVANCE_RULES: dict[str, RelevanceRule] = {
    "monetary_policy": RelevanceRule(0.40, {}, "monetary_policy"),
    "economic_outcomes": RelevanceRule(0.30, {}, "macro_economic_outcomes"),
    "elections_policy": RelevanceRule(0.20, {"ENERGY": 0.20, "FINANCE": 0.20, "HEALTH": 0.20}, "policy_uncertainty"),
    "geopolitical": RelevanceRule(0.25, {"ENERGY": 0.15}, "geopolitical_risk"),
    "regulatory": RelevanceRule(0.15, {"FINANCE": 0.30, "TECH": 0.30}, "regulatory_exposure"),
    # "other"-category events are deliberately NOT mapped -- see
    # compute_event_relevance's early return. Unrelated event markets
    # (sports, entertainment, etc.) must never be silently wired to a stock.
}


def compute_event_relevance(
    event: EventProbabilityObservation, symbol: str, sector: str | None = None, created_at: datetime | None = None
) -> EventSymbolMapping | None:
    rule = RELEVANCE_RULES.get(event.category or "")
    if rule is None:
        return None
    relevance = min(1.0, rule.base_relevance + rule.sector_bonus.get(sector or "", 0.0))
    return EventSymbolMapping(
        event_id=event.event_id, symbol=symbol, relevance=relevance,
        rationale_category=rule.rationale_category, created_at=created_at or datetime.now(UTC),
    )


def compute_event_relevance_for_universe(
    events: list[EventProbabilityObservation], symbols: list[str], sector_map: dict[str, str]
) -> list[EventSymbolMapping]:
    mappings = []
    for event in events:
        for symbol in symbols:
            mapping = compute_event_relevance(event, symbol, sector_map.get(symbol))
            if mapping is not None:
                mappings.append(mapping)
    return mappings
