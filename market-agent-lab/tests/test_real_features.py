"""Tests for the real point-in-time feature matrix builder's core
correctness-critical helpers: the event-probability as-of lookup (the
leakage guard for that feature family) and the news-row adapter."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from core.schemas_v2 import NewsTier
from data.real_features import (
    _build_event_probability_lookup,
    _event_probability_features,
    _lookup_asof,
    _news_row_for_agent,
)
from database.db import fresh_connection


def test_lookup_asof_returns_none_when_observation_is_in_the_future():
    history = [(datetime(2024, 1, 1), 0.5)]
    assert _lookup_asof(history, datetime(2023, 1, 1)) is None


def test_lookup_asof_returns_latest_value_at_or_before_as_of():
    history = [(datetime(2023, 1, 1), 0.3), (datetime(2023, 6, 1), 0.6), (datetime(2023, 12, 1), 0.8)]
    assert _lookup_asof(history, datetime(2023, 7, 1)) == pytest.approx(0.6)
    assert _lookup_asof(history, datetime(2023, 12, 1)) == pytest.approx(0.8)
    assert _lookup_asof(history, datetime(2022, 1, 1)) is None


def test_lookup_asof_empty_history_returns_none():
    assert _lookup_asof([], datetime(2023, 1, 1)) is None


def test_event_probability_features_marks_missing_when_no_mapping():
    features = _event_probability_features("AAPL", datetime(2023, 1, 1), {}, {})
    assert features["eventprob_missing"] == 1.0
    assert all(v != v for k, v in features.items() if k.endswith("_probability"))  # all NaN


def test_event_probability_features_weighted_average_across_events():
    symbol_category_events = {("AAPL", "monetary_policy"): [("e1", 0.5), ("e2", 0.5)]}
    obs_by_event = {"e1": [(datetime(2023, 1, 1), 0.8)], "e2": [(datetime(2023, 1, 1), 0.2)]}
    features = _event_probability_features("AAPL", datetime(2023, 6, 1), symbol_category_events, obs_by_event)
    assert features["eventprob_monetary_policy_probability"] == pytest.approx(0.5)
    assert features["eventprob_missing"] == 0.0


def test_event_probability_features_respects_point_in_time_per_event():
    """One event is knowable at as_of, the other isn't yet -- only the
    knowable one should contribute."""
    symbol_category_events = {("AAPL", "monetary_policy"): [("e1", 1.0), ("e2", 1.0)]}
    obs_by_event = {"e1": [(datetime(2023, 1, 1), 0.9)], "e2": [(datetime(2023, 12, 1), 0.1)]}
    features = _event_probability_features("AAPL", datetime(2023, 6, 1), symbol_category_events, obs_by_event)
    assert features["eventprob_monetary_policy_probability"] == pytest.approx(0.9)  # only e1 was known yet


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        yield c


def test_build_event_probability_lookup_empty_db_returns_empty_dicts(con):
    mappings, obs = _build_event_probability_lookup(con, ["AAPL"])
    assert mappings == {}
    assert obs == {}


def test_build_event_probability_lookup_reads_from_db(con):
    from core.schemas_v2 import EventProbabilityObservation, EventSymbolMapping
    from database import repository_v2 as repo_v2

    repo_v2.insert_event_probability_observations(
        con,
        [
            EventProbabilityObservation(
                event_id="fed1", question="Fed?", category="monetary_policy",
                observed_timestamp=datetime(2023, 1, 1), public_probability=0.7,
                source="polymarket_readonly", retrieved_at=datetime(2023, 1, 1),
            )
        ],
    )
    repo_v2.insert_event_symbol_mappings(
        con,
        [
            EventSymbolMapping(
                event_id="fed1", symbol="AAPL", relevance=0.4,
                rationale_category="monetary_policy", created_at=datetime(2023, 1, 1),
            )
        ],
    )
    symbol_category_events, obs_by_event = _build_event_probability_lookup(con, ["AAPL"])
    assert ("AAPL", "monetary_policy") in symbol_category_events
    assert "fed1" in obs_by_event
    features = _event_probability_features("AAPL", datetime(2023, 6, 1), symbol_category_events, obs_by_event)
    assert features["eventprob_monetary_policy_probability"] == pytest.approx(0.7)


# --- news row adapter -----------------------------------------------------------------


def test_news_row_adapter_defaults_when_no_articles():
    row = _news_row_for_agent(pd.DataFrame(columns=["headline", "published_at", "retrieved_at", "source", "tier", "event_category", "timestamp_uncertain"]))
    assert row == {"news_sentiment": 0.0, "event_uncertainty": 0.5, "is_earnings_event": False}


def test_news_row_adapter_detects_earnings_event():
    articles = pd.DataFrame(
        [
            {
                "headline": "Q1 earnings", "published_at": datetime(2023, 1, 1), "retrieved_at": datetime(2023, 1, 1),
                "source": "sec_events", "tier": NewsTier.TIER_1_OFFICIAL.value, "event_category": "earnings",
                "timestamp_uncertain": False,
            }
        ]
    )
    row = _news_row_for_agent(articles)
    assert row["is_earnings_event"] is True
    assert row["news_sentiment"] == pytest.approx(0.10)  # heuristic earnings sentiment
