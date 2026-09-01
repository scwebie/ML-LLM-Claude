"""Tests for the real news pipeline: SEC 8-K parsing (mocked HTTP),
deterministic news feature counts, the heuristic classifier, deduplication,
and the point-in-time news guard."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
import responses

from core.schemas_v2 import NewsArticle, NewsTier
from data.providers.news.sec_events import SecEventsProvider
from database import repository_v2 as repo_v2
from database.db import fresh_connection
from features.news_features import classify_article, compute_deterministic_news_counts


@responses.activate
def test_sec_events_maps_item_codes_to_categories(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 1, "ticker": "TEST", "title": "Test Co"}}, status=200,
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "filingDate": ["2023-02-02", "2023-02-01"],
                "acceptanceDateTime": ["2023-02-02T21:30:33.000Z", "2023-02-01T20:00:00.000Z"],
                "accessionNumber": ["0000000001-23-000001", "0000000001-23-000002"],
                "items": ["2.02,9.01", ""],
            }
        }
    }
    responses.add(responses.GET, "https://data.sec.gov/submissions/CIK0000000001.json", json=submissions, status=200)

    provider = SecEventsProvider()
    events = provider.get_events("TEST", datetime(2023, 1, 1), datetime(2023, 12, 31))
    assert len(events) == 1  # the 10-Q was excluded, only the 8-K became a news item
    assert events[0].event_category == "earnings"
    assert events[0].tier == NewsTier.TIER_1_OFFICIAL
    assert events[0].timestamp_uncertain is False


@responses.activate
def test_sec_events_marks_timestamp_uncertain_when_acceptance_missing(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 1, "ticker": "TEST", "title": "Test Co"}}, status=200,
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K"], "filingDate": ["2023-02-02"], "acceptanceDateTime": [""],
                "accessionNumber": ["0000000001-23-000001"], "items": ["8.01"],
            }
        }
    }
    responses.add(responses.GET, "https://data.sec.gov/submissions/CIK0000000001.json", json=submissions, status=200)

    provider = SecEventsProvider()
    events = provider.get_events("TEST", datetime(2023, 1, 1), datetime(2023, 12, 31))
    assert events[0].timestamp_uncertain is True
    assert events[0].published_at == datetime(2023, 2, 2)  # fell back to filing date, midnight


# --- deterministic news counts -------------------------------------------------------


def test_deterministic_counts_respect_lookback_windows():
    as_of = datetime(2023, 6, 15, 12, 0, 0)
    articles = pd.DataFrame(
        {
            "published_at": [
                as_of - pd.Timedelta(minutes=30),  # within 1h
                as_of - pd.Timedelta(hours=3),  # within 6h, not 1h
                as_of - pd.Timedelta(days=2),  # within 3d
                as_of - pd.Timedelta(days=6),  # within 7d
                as_of - pd.Timedelta(days=30),  # outside all windows
            ],
            "tier": [NewsTier.TIER_1_OFFICIAL.value] * 5,
        }
    )
    counts = compute_deterministic_news_counts(articles, as_of)
    assert counts["n_articles_1h"] == 1
    assert counts["n_articles_6h"] == 2
    assert counts["n_articles_24h"] == 2
    assert counts["n_articles_3d"] == 3
    assert counts["n_articles_7d"] == 4


def test_deterministic_counts_empty_articles_returns_zeros():
    counts = compute_deterministic_news_counts(pd.DataFrame(columns=["published_at", "tier"]), datetime(2023, 1, 1))
    assert counts["n_articles_1h"] == 0.0
    assert counts["hours_since_last_article"] != counts["hours_since_last_article"]  # NaN


def test_source_weighted_count_reflects_tier_weights():
    as_of = datetime(2023, 6, 15)
    articles = pd.DataFrame(
        {
            "published_at": [as_of - pd.Timedelta(hours=1), as_of - pd.Timedelta(hours=2)],
            "tier": [NewsTier.TIER_1_OFFICIAL.value, NewsTier.TIER_3_OTHER.value],
        }
    )
    counts = compute_deterministic_news_counts(articles, as_of)
    assert counts["source_weighted_count_7d"] == pytest.approx(1.0 + 0.3)


# --- heuristic classifier ------------------------------------------------------------


def _article(category: str, tier=NewsTier.TIER_1_OFFICIAL, timestamp_uncertain=False, headline="Test headline") -> NewsArticle:
    return NewsArticle(
        headline=headline, published_at=datetime(2023, 1, 1), retrieved_at=datetime(2023, 1, 1),
        source="test", tier=tier, event_category=category, timestamp_uncertain=timestamp_uncertain, symbols=["TEST"],
    )


def test_classify_article_records_heuristic_provenance_not_fake_llm():
    features = classify_article(_article("earnings"))
    assert features.llm_model == "heuristic_v1"  # never falsely claims a real LLM call


def test_classify_article_litigation_is_negative_sentiment():
    features = classify_article(_article("litigation"))
    assert features.sentiment < 0


def test_classify_article_earnings_is_higher_impact_than_other():
    earnings = classify_article(_article("earnings"))
    other = classify_article(_article("other"))
    assert earnings.impact_magnitude > other.impact_magnitude


def test_classify_article_uncertain_timestamp_raises_uncertainty():
    certain = classify_article(_article("other", timestamp_uncertain=False))
    uncertain = classify_article(_article("other", timestamp_uncertain=True))
    assert uncertain.uncertainty > certain.uncertainty


def test_classify_article_repeated_headline_is_not_novel():
    article = _article("other", headline="Repeated Headline")
    fresh = classify_article(article, prior_headlines_7d=set())
    repeat = classify_article(article, prior_headlines_7d={"Repeated Headline"})
    assert fresh.novelty > repeat.novelty


# --- deduplication + point-in-time news guard -----------------------------------------


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        yield c


def test_duplicate_article_detected_by_dedupe_key(con):
    article = NewsArticle(
        headline="X", published_at=datetime(2023, 1, 1), retrieved_at=datetime(2023, 1, 1),
        source="test", tier=NewsTier.TIER_1_OFFICIAL, dedupe_key="unique-key-1", symbols=["TEST"],
    )
    repo_v2.insert_news_articles(con, [article])
    assert repo_v2.is_duplicate_article(con, "unique-key-1") is True
    assert repo_v2.is_duplicate_article(con, "never-seen") is False


def test_get_news_asof_excludes_future_publication(con):
    older = NewsArticle(
        headline="Old news", published_at=datetime(2023, 1, 1), retrieved_at=datetime(2023, 1, 1),
        source="test", tier=NewsTier.TIER_1_OFFICIAL, dedupe_key="k1", symbols=["TEST"],
    )
    newer = NewsArticle(
        headline="Future news", published_at=datetime(2023, 6, 1), retrieved_at=datetime(2023, 6, 1),
        source="test", tier=NewsTier.TIER_1_OFFICIAL, dedupe_key="k2", symbols=["TEST"],
    )
    repo_v2.insert_news_articles(con, [older, newer])

    asof_march = repo_v2.get_news_asof(con, "TEST", datetime(2023, 3, 1), lookback_days=365)
    assert list(asof_march["headline"]) == ["Old news"]

    asof_july = repo_v2.get_news_asof(con, "TEST", datetime(2023, 7, 1), lookback_days=365)
    assert set(asof_july["headline"]) == {"Old news", "Future news"}


def test_get_news_asof_excludes_timestamp_uncertain_by_default(con):
    uncertain = NewsArticle(
        headline="Uncertain", published_at=datetime(2023, 1, 1), retrieved_at=datetime(2023, 1, 1),
        source="test", tier=NewsTier.TIER_1_OFFICIAL, dedupe_key="k3", symbols=["TEST"], timestamp_uncertain=True,
    )
    repo_v2.insert_news_articles(con, [uncertain])
    result = repo_v2.get_news_asof(con, "TEST", datetime(2023, 6, 1), lookback_days=365)
    assert result.empty
    result_included = repo_v2.get_news_asof(con, "TEST", datetime(2023, 6, 1), lookback_days=365, include_timestamp_uncertain=True)
    assert len(result_included) == 1
