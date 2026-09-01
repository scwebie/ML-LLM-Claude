"""HARD SAFETY TEST: the read-only prediction-market provider must never
expose any execution-shaped capability -- no order placement, no account
creation, no wallet, no wagering, ever. This is enforced structurally by
introspection, not by convention or code review alone.
"""

from __future__ import annotations

import json

import pytest
import responses

from agents.event_relevance import compute_event_relevance, compute_event_relevance_for_universe
from core.schemas_v2 import EventProbabilityObservation
from data.providers.events.prediction_market_readonly import (
    ALLOWED_PUBLIC_API,
    PredictionMarketReadOnlyProvider,
)

# Any of these substrings appearing in a public method/attribute name is an
# automatic failure -- a second, independent line of defense beyond the
# strict allow-list check below.
FORBIDDEN_NAME_FRAGMENTS = [
    "order", "buy", "sell", "wager", "bet", "wallet", "deposit", "withdraw",
    "transfer", "sign", "login", "auth", "approve", "trade", "execute",
    "cancel", "fund", "stake", "place_", "submit",
]


def test_forbidden_fragments_list_is_itself_sane():
    # Guards against a typo silently neutering the test below.
    assert "order" in FORBIDDEN_NAME_FRAGMENTS
    assert len(FORBIDDEN_NAME_FRAGMENTS) >= 10


def test_public_api_is_exactly_the_declared_allow_list():
    provider = PredictionMarketReadOnlyProvider()
    public_attrs = {name for name in dir(provider) if not name.startswith("_")}
    assert public_attrs == ALLOWED_PUBLIC_API


def test_allowed_public_api_contains_no_execution_shaped_names():
    for name in ALLOWED_PUBLIC_API:
        lowered = name.lower()
        for fragment in FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in lowered, f"'{name}' contains forbidden fragment '{fragment}'"


def test_class_has_no_execution_shaped_method_anywhere_including_private():
    """Even a private/internal method with execution-shaped semantics
    would be a red flag (someone could still call it) -- scan everything,
    not just the public surface."""
    provider = PredictionMarketReadOnlyProvider()
    for name in dir(provider):
        if name.startswith("__"):
            continue
        lowered = name.lower()
        for fragment in FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in lowered, f"'{name}' contains forbidden fragment '{fragment}' (even as a private method)"


def test_provider_module_has_no_write_http_verbs_referenced():
    """Static check: the module source must never reference requests.post/
    put/delete/patch -- it should only ever GET."""
    import inspect

    import data.providers.events.prediction_market_readonly as module

    source = inspect.getsource(module)
    for verb in (".post(", ".put(", ".delete(", ".patch("):
        assert verb not in source


# --- functional parsing tests (mocked HTTP) -------------------------------------------


def _event_payload() -> list[dict]:
    return [
        {
            "id": "1", "title": "Fed rate decision", "markets": [
                {
                    "id": "m1", "question": "Will the Fed cut rates in September?",
                    "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["0.72", "0.28"]),
                    "liquidityNum": 50000.0, "volumeNum": 200000.0, "volume24hr": 1000.0,
                    "endDateIso": "2025-09-30",
                }
            ],
        },
        {
            "id": "2", "title": "Will it rain in NYC tomorrow", "markets": [
                {
                    "id": "m2", "question": "Will it rain in NYC tomorrow?",
                    "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["0.30", "0.70"]),
                    "liquidityNum": 100.0, "volumeNum": 500.0, "volume24hr": 10.0,
                },
            ],
        },
    ]


@responses.activate
def test_get_active_events_parses_probability_and_classifies_category(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(responses.GET, "https://gamma-api.polymarket.com/events", json=_event_payload(), status=200)

    provider = PredictionMarketReadOnlyProvider()
    observations = provider.get_active_events(limit=10)
    fed_obs = [o for o in observations if o.event_id == "m1"][0]
    assert fed_obs.public_probability == pytest.approx(0.72)
    assert fed_obs.category == "monetary_policy"

    weather_obs = [o for o in observations if o.event_id == "m2"][0]
    assert weather_obs.category == "other"


@responses.activate
def test_get_active_events_probability_is_clamped_to_valid_range(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    payload = _event_payload()
    payload[0]["markets"][0]["outcomePrices"] = json.dumps(["1.5", "-0.5"])  # malformed/out-of-range
    responses.add(responses.GET, "https://gamma-api.polymarket.com/events", json=payload, status=200)

    provider = PredictionMarketReadOnlyProvider()
    observations = provider.get_active_events(limit=10)
    fed_obs = [o for o in observations if o.event_id == "m1"][0]
    assert 0.0 <= fed_obs.public_probability <= 1.0


@responses.activate
def test_category_filter_excludes_non_matching_events(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(responses.GET, "https://gamma-api.polymarket.com/events", json=_event_payload(), status=200)

    provider = PredictionMarketReadOnlyProvider()
    observations = provider.get_active_events(limit=10, category_filter="monetary_policy")
    assert len(observations) == 1
    assert observations[0].category == "monetary_policy"


# --- explicit structured relevance mapping ---------------------------------------------


def test_other_category_events_are_never_mapped_to_any_symbol():
    event = EventProbabilityObservation(
        event_id="weather1", question="Will it rain?", category="other",
        observed_timestamp=__import__("datetime").datetime(2023, 1, 1), public_probability=0.5,
        source="polymarket_readonly", retrieved_at=__import__("datetime").datetime(2023, 1, 1),
    )
    assert compute_event_relevance(event, "AAPL") is None


def test_monetary_policy_event_maps_to_every_symbol_with_base_relevance():
    event = EventProbabilityObservation(
        event_id="fed1", question="Will the Fed cut rates?", category="monetary_policy",
        observed_timestamp=__import__("datetime").datetime(2023, 1, 1), public_probability=0.6,
        source="polymarket_readonly", retrieved_at=__import__("datetime").datetime(2023, 1, 1),
    )
    mapping = compute_event_relevance(event, "AAPL", sector="TECH")
    assert mapping is not None
    assert mapping.relevance == pytest.approx(0.40)
    assert mapping.rationale_category == "monetary_policy"


def test_regulatory_event_relevance_boosted_for_sensitive_sectors():
    event = EventProbabilityObservation(
        event_id="reg1", question="Will the SEC sue X?", category="regulatory",
        observed_timestamp=__import__("datetime").datetime(2023, 1, 1), public_probability=0.4,
        source="polymarket_readonly", retrieved_at=__import__("datetime").datetime(2023, 1, 1),
    )
    tech_mapping = compute_event_relevance(event, "AAPL", sector="TECH")
    consumer_mapping = compute_event_relevance(event, "KO", sector="CONSUMER")
    assert tech_mapping.relevance > consumer_mapping.relevance


def test_relevance_never_exceeds_one():
    event = EventProbabilityObservation(
        event_id="reg2", question="regulation", category="regulatory",
        observed_timestamp=__import__("datetime").datetime(2023, 1, 1), public_probability=0.4,
        source="polymarket_readonly", retrieved_at=__import__("datetime").datetime(2023, 1, 1),
    )
    mapping = compute_event_relevance(event, "AAPL", sector="TECH")
    assert mapping.relevance <= 1.0


def test_compute_relevance_for_universe_only_includes_mappable_events():
    events = [
        EventProbabilityObservation(
            event_id="fed1", question="Fed", category="monetary_policy",
            observed_timestamp=__import__("datetime").datetime(2023, 1, 1), public_probability=0.6,
            source="polymarket_readonly", retrieved_at=__import__("datetime").datetime(2023, 1, 1),
        ),
        EventProbabilityObservation(
            event_id="sport1", question="Sports", category="other",
            observed_timestamp=__import__("datetime").datetime(2023, 1, 1), public_probability=0.6,
            source="polymarket_readonly", retrieved_at=__import__("datetime").datetime(2023, 1, 1),
        ),
    ]
    mappings = compute_event_relevance_for_universe(events, ["AAPL", "MSFT"], {"AAPL": "TECH", "MSFT": "TECH"})
    assert len(mappings) == 2  # only the Fed event mapped, x2 symbols; the sports event mapped to none
    assert all(m.event_id == "fed1" for m in mappings)
