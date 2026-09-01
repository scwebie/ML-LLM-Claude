"""Tests for the SEC EDGAR provider's raw-fact extraction (mocked HTTP,
no live network) and the raw ``fundamental_facts`` as-of join."""

from __future__ import annotations

from datetime import datetime

import pytest
import responses

from core.schemas_v2 import FundamentalFact
from data.providers.fundamentals.sec import SecEdgarFundamentalProvider
from database import repository_v2 as repo_v2
from database.db import fresh_connection


@responses.activate
def test_resolve_cik_matches_ticker_case_insensitively(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}, status=200,
    )
    provider = SecEdgarFundamentalProvider()
    assert provider.resolve_cik("aapl") == "0000320193"
    assert provider.resolve_cik("NOTREAL") is None


@responses.activate
def test_get_company_fundamentals_extracts_relevant_forms_only(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 1, "ticker": "TEST", "title": "Test Co"}}, status=200,
    )
    company_facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {"start": "2023-01-01", "end": "2023-03-31", "val": 100, "accn": "a1", "fy": 2023, "fp": "Q1", "form": "10-Q", "filed": "2023-04-20"},
                            {"start": "2023-01-01", "end": "2023-03-31", "val": 999, "accn": "a2", "fy": 2023, "fp": "Q1", "form": "8-K", "filed": "2023-04-15"},
                        ]
                    }
                }
            }
        }
    }
    responses.add(responses.GET, "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json", json=company_facts, status=200)

    provider = SecEdgarFundamentalProvider()
    facts = provider.get_company_fundamentals("TEST")
    assert len(facts) == 1  # the 8-K entry was excluded (not in RELEVANT_FORMS)
    assert facts[0].tag == "revenue"
    assert facts[0].value == 100.0
    assert facts[0].filed_date == datetime(2023, 4, 20)


@responses.activate
def test_get_company_fundamentals_falls_back_to_older_tag_name(tmp_path, monkeypatch):
    """Older filings use 'Revenues' instead of the post-ASC-606 tag;
    the provider must still find the data."""
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://www.sec.gov/files/company_tickers.json",
        json={"0": {"cik_str": 1, "ticker": "TEST", "title": "Test Co"}}, status=200,
    )
    company_facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {"USD": [{"start": "2016-01-01", "end": "2016-03-31", "val": 55, "accn": "a1", "fy": 2016, "fp": "Q1", "form": "10-Q", "filed": "2016-04-20"}]}
                }
            }
        }
    }
    responses.add(responses.GET, "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json", json=company_facts, status=200)

    provider = SecEdgarFundamentalProvider()
    facts = provider.get_company_fundamentals("TEST")
    assert len(facts) == 1
    assert facts[0].tag == "revenue"
    assert facts[0].value == 55.0


# --- point-in-time fundamental_facts as-of join --------------------------------------


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        yield c


def test_fundamental_facts_asof_excludes_future_filings(con):
    older = FundamentalFact(
        symbol="TEST", cik="1", tag="revenue", unit="USD",
        period_end=datetime(2023, 3, 31), value=100.0, filed_date=datetime(2023, 4, 20),
        source="sec_edgar", retrieved_at=datetime(2023, 4, 20),
    )
    newer = FundamentalFact(
        symbol="TEST", cik="1", tag="revenue", unit="USD",
        period_end=datetime(2023, 6, 30), value=120.0, filed_date=datetime(2023, 7, 25),
        source="sec_edgar", retrieved_at=datetime(2023, 7, 25),
    )
    repo_v2.insert_fundamental_facts(con, [older, newer])

    asof_before_second_filing = repo_v2.get_fundamental_facts_asof(con, "TEST", datetime(2023, 6, 1))
    asof_after_second_filing = repo_v2.get_fundamental_facts_asof(con, "TEST", datetime(2023, 8, 1))

    assert asof_before_second_filing.iloc[0]["value"] == 100.0
    assert asof_after_second_filing.iloc[0]["value"] == 120.0
