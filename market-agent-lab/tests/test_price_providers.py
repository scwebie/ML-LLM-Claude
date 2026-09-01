"""Tests for the real price provider clients, using mocked HTTP responses
(never live network -- that would make the suite flaky/slow and coupled
to third-party uptime). Live-network behaviour is exercised manually via
``main.py ingest-prices`` / ``real-demo``, not in the regular test suite.
"""

from __future__ import annotations

from datetime import datetime

import responses

from data.providers.prices.primary import YahooFinancePriceProvider
from data.providers.prices.secondary import StockAnalysisPriceProvider


def _yahoo_payload() -> dict:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [1672738800, 1672825200],  # 2023-01-03, 2023-01-04 UTC-ish
                    "indicators": {
                        "quote": [
                            {
                                "open": [130.0, 126.0],
                                "high": [130.9, 128.7],
                                "low": [124.2, 125.1],
                                "close": [125.07, 126.36],
                                "volume": [112117500, 89113600],
                            }
                        ],
                        "adjclose": [{"adjclose": [124.8, 126.1]}],
                    },
                    "events": {"dividends": {}, "splits": {}},
                }
            ],
            "error": None,
        }
    }


@responses.activate
def test_yahoo_provider_parses_bars_correctly(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(responses.GET, "https://query1.finance.yahoo.com/v8/finance/chart/AAPL", json=_yahoo_payload(), status=200)

    provider = YahooFinancePriceProvider()
    bars = provider.get_daily_bars("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 10))
    assert len(bars) == 2
    assert bars[0]["close"] == 125.07
    assert bars[0]["adjusted_close"] == 124.8
    assert bars[0]["volume"] == 112117500


@responses.activate
def test_yahoo_provider_skips_bars_with_missing_ohlc(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    payload = _yahoo_payload()
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = None  # halted/missing session
    responses.add(responses.GET, "https://query1.finance.yahoo.com/v8/finance/chart/MSFT", json=payload, status=200)

    provider = YahooFinancePriceProvider()
    bars = provider.get_daily_bars("MSFT", datetime(2023, 1, 1), datetime(2023, 1, 10))
    assert len(bars) == 1  # the None-close row was skipped, not fabricated


@responses.activate
def test_yahoo_provider_raises_on_chart_error(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(
        responses.GET, "https://query1.finance.yahoo.com/v8/finance/chart/BADSYM",
        json={"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}}}, status=200,
    )
    provider = YahooFinancePriceProvider()
    try:
        provider.get_daily_bars("BADSYM", datetime(2023, 1, 1), datetime(2023, 1, 10))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def _stockanalysis_payload() -> dict:
    return {
        "status": 200,
        "data": [
            {"t": "2023-01-03", "o": 130.0, "h": 130.9, "l": 124.2, "c": 125.07, "a": 124.8, "v": 112117500, "ch": -1.0},
            {"t": "2023-01-04", "o": 126.0, "h": 128.7, "l": 125.1, "c": 126.36, "a": 126.1, "v": 89113600, "ch": 1.03},
            {"t": "2020-01-02", "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "a": 100.5, "v": 1000, "ch": 0.5},
        ],
    }


@responses.activate
def test_stockanalysis_provider_parses_and_filters_by_date_range(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(responses.GET, "https://stockanalysis.com/api/symbol/s/AAPL/history", json=_stockanalysis_payload(), status=200)

    provider = StockAnalysisPriceProvider()
    bars = provider.get_daily_bars("AAPL", datetime(2023, 1, 1), datetime(2023, 1, 10))
    assert len(bars) == 2  # the 2020 row is outside the requested range
    assert bars[0]["close"] == 125.07
    assert bars[0]["adjusted_close"] == 124.8


@responses.activate
def test_stockanalysis_provider_raises_on_error_status(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    responses.add(responses.GET, "https://stockanalysis.com/api/symbol/s/BADSYM/history", json={"status": 404, "data": []}, status=200)

    provider = StockAnalysisPriceProvider()
    try:
        provider.get_daily_bars("BADSYM", datetime(2023, 1, 1), datetime(2023, 1, 10))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
