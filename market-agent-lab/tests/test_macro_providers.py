"""Tests for macro providers (mocked HTTP, no live network) -- CSV/JSON
parsing correctness and the documented publication-lag approximation."""

from __future__ import annotations

from datetime import datetime

import pytest
import responses

from data.providers.macro.bea import BeaProvider
from data.providers.macro.bls import BlsProvider, _parse_bls_value
from data.providers.macro.fred import FredProvider
from data.providers.macro.treasury import TreasuryProvider


@responses.activate
def test_fred_provider_parses_csv_and_computes_publication_lag(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    csv_body = "observation_date,DGS10\n2023-01-03,3.79\n2023-01-04,3.71\n2023-01-05,.\n"
    responses.add(responses.GET, "https://fred.stlouisfed.org/graph/fredgraph.csv", body=csv_body, status=200)

    provider = FredProvider()
    rows = provider.get_series("DGS10", datetime(2023, 1, 1), datetime(2023, 1, 10))
    assert len(rows) == 2  # the "." missing-value row was excluded
    assert rows[0]["value"] == 3.79
    # DGS10's typical_lag_days is 1 -- next business day.
    assert rows[0]["publication_date"] == datetime(2023, 1, 4)


@responses.activate
def test_fred_provider_filters_by_date_range(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    csv_body = "observation_date,DGS10\n2020-01-03,1.0\n2023-01-04,3.71\n"
    responses.add(responses.GET, "https://fred.stlouisfed.org/graph/fredgraph.csv", body=csv_body, status=200)

    provider = FredProvider()
    rows = provider.get_series("DGS10", datetime(2023, 1, 1), datetime(2023, 1, 10))
    assert len(rows) == 1
    assert rows[0]["date"] == datetime(2023, 1, 4)


@responses.activate
def test_bls_provider_converts_period_to_date_and_lag(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {"series": [{"seriesID": "CUUR0000SA0", "data": [{"year": "2023", "period": "M01", "periodName": "January", "value": "300.5"}]}]},
    }
    responses.add(responses.GET, "https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0", json=payload, status=200)

    provider = BlsProvider()
    rows = provider.get_series("CUUR0000SA0", datetime(2023, 1, 1), datetime(2023, 2, 1))
    assert len(rows) == 1
    assert rows[0]["date"] == datetime(2023, 1, 31)  # last day of January
    assert rows[0]["publication_date"] == datetime(2023, 2, 13)  # +13 days typical CPI lag


@responses.activate
def test_bls_provider_raises_on_failed_status(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://api.bls.gov/publicAPI/v2/timeseries/data/BADSERIES",
        json={"status": "REQUEST_NOT_PROCESSED", "message": ["bad series"]}, status=200,
    )
    provider = BlsProvider()
    with pytest.raises(ValueError):
        provider.get_series("BADSERIES", datetime(2023, 1, 1), datetime(2023, 2, 1))


# --- _parse_bls_value: individual-observation parsing, never raises -----


def test_parse_bls_value_normal_numeric_value():
    assert _parse_bls_value("300.5") == pytest.approx(300.5)


def test_parse_bls_value_dash_is_missing():
    assert _parse_bls_value("-") is None


def test_parse_bls_value_empty_string_is_missing():
    assert _parse_bls_value("") is None
    assert _parse_bls_value("   ") is None


def test_parse_bls_value_none_is_missing():
    assert _parse_bls_value(None) is None


def test_parse_bls_value_na_token_is_missing():
    assert _parse_bls_value("N/A") is None
    assert _parse_bls_value("n/a") is None


def test_parse_bls_value_rejects_non_finite():
    assert _parse_bls_value("nan") is None
    assert _parse_bls_value("inf") is None
    assert _parse_bls_value("-inf") is None


def test_parse_bls_value_rejects_unparseable_string():
    assert _parse_bls_value("not-a-number") is None


# --- BlsProvider.get_series: missing observations don't fail the series -


@responses.activate
def test_bls_provider_mixed_valid_and_invalid_observations(tmp_path, monkeypatch):
    """5/6. A series with some suppressed ('-') observations ingests every
    valid one, skips the invalid ones individually, and does not raise or
    otherwise fail the whole series."""
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "CUUR0000SA0",
                    "data": [
                        {"year": "2023", "period": "M03", "periodName": "March", "value": "301.8"},
                        {"year": "2023", "period": "M02", "periodName": "February", "value": "-"},
                        {"year": "2023", "period": "M01", "periodName": "January", "value": "300.5"},
                    ],
                }
            ]
        },
    }
    responses.add(responses.GET, "https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0", json=payload, status=200)

    provider = BlsProvider()
    rows = provider.get_series("CUUR0000SA0", datetime(2023, 1, 1), datetime(2023, 4, 1))

    # Series did not fail; the two valid observations were ingested, the
    # one suppressed observation was skipped and reported, not fabricated.
    assert len(rows) == 2
    assert provider.last_skipped == 1
    assert all(row["value"] not in (None,) for row in rows)

    # 7. Valid observations remain chronologically sorted despite the
    # source payload listing them out of order and interleaved with the
    # skipped one.
    assert [row["date"] for row in rows] == sorted(row["date"] for row in rows)
    assert rows[0]["date"] == datetime(2023, 1, 31)
    assert rows[1]["date"] == datetime(2023, 3, 31)


@responses.activate
def test_bls_provider_reports_zero_skipped_when_all_observations_valid(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    payload = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {"series": [{"seriesID": "CUUR0000SA0", "data": [{"year": "2023", "period": "M01", "value": "300.5"}]}]},
    }
    responses.add(responses.GET, "https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0", json=payload, status=200)

    provider = BlsProvider()
    rows = provider.get_series("CUUR0000SA0", datetime(2023, 1, 1), datetime(2023, 2, 1))
    assert len(rows) == 1
    assert provider.last_skipped == 0


@responses.activate
def test_treasury_provider_parses_response(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    payload = {"data": [{"record_date": "2023-01-31", "avg_interest_rate_amt": "3.456"}]}
    responses.add(
        responses.GET, "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates",
        json=payload, status=200,
    )
    provider = TreasuryProvider()
    rows = provider.get_series("Treasury Notes", datetime(2023, 1, 1), datetime(2023, 2, 1))
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(3.456)


def test_bea_provider_raises_clear_error_without_key(monkeypatch):
    monkeypatch.delenv("BEA_API_KEY", raising=False)
    provider = BeaProvider()
    with pytest.raises(RuntimeError, match="BEA_API_KEY"):
        provider.get_series("T10101", datetime(2023, 1, 1), datetime(2023, 2, 1))
