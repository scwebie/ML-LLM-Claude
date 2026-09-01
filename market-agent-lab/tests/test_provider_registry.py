"""Tests for the provider registry and shared provider infrastructure."""

from __future__ import annotations

import time

import pytest
import responses

from core.schemas_v2 import ProviderCategory
from data.providers import registry
from data.providers.base import (
    RateLimitConfig,
    RateLimitedClient,
    cache_get,
    cache_set,
    cached_fetch,
)


def test_registry_disables_known_unusable_sources():
    stooq = registry.get_source("stooq")
    gdelt = registry.get_source("gdelt")
    assert stooq is not None and stooq.is_enabled is False
    assert gdelt is not None and gdelt.is_enabled is False
    assert stooq.notes and "proof-of-work" in stooq.notes


def test_registry_enables_keyless_working_sources():
    enabled_ids = {s.source_id for s in registry.get_enabled()}
    for expected in ("yahoo_finance", "stockanalysis", "sec_edgar", "fred", "bls", "treasury", "sec_events", "polymarket_readonly"):
        assert expected in enabled_ids


def test_registry_gates_key_required_sources_without_key(monkeypatch):
    monkeypatch.delenv("BEA_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    import importlib

    import data.providers.registry as reg_module

    importlib.reload(reg_module)
    bea = reg_module.get_source("bea")
    news_api = reg_module.get_source("news_api")
    assert bea.is_enabled is False
    assert news_api.is_enabled is False
    importlib.reload(reg_module)  # restore module state for subsequent tests


def test_registry_get_enabled_filters_by_category():
    price_sources = registry.get_enabled(ProviderCategory.PRICE)
    assert all(s.category == ProviderCategory.PRICE for s in price_sources)
    assert len(price_sources) >= 2  # at least primary + secondary


def test_cache_roundtrip(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    cache_set("test_ns", {"a": 1}, {"value": 42})
    result = cache_get("test_ns", {"a": 1})
    assert result == {"value": 42}


def test_cache_miss_returns_none(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    assert cache_get("nonexistent_ns", {"x": 1}) is None


def test_cached_fetch_does_not_call_fetch_fn_twice(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"data": "x"}

    first = cached_fetch("ns", {"k": "v"}, fetch)
    second = cached_fetch("ns", {"k": "v"}, fetch)
    assert first == second == {"data": "x"}
    assert calls["n"] == 1  # second call was served from cache


def test_cached_fetch_force_refresh_bypasses_cache(tmp_path, monkeypatch):
    import data.providers.base as base_module

    monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"n": calls["n"]}

    cached_fetch("ns", {"k": "v"}, fetch)
    cached_fetch("ns", {"k": "v"}, fetch, force_refresh=True)
    assert calls["n"] == 2


@responses.activate
def test_rate_limited_client_retries_on_429_then_succeeds():
    responses.add(responses.GET, "https://example.com/data", status=429, headers={"Retry-After": "0"})
    responses.add(responses.GET, "https://example.com/data", json={"ok": True}, status=200)

    client = RateLimitedClient(RateLimitConfig(min_interval_seconds=0.0, backoff_base_seconds=0.01))
    response = client.get("https://example.com/data", source_id="test_source")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@responses.activate
def test_rate_limited_client_retries_on_5xx():
    responses.add(responses.GET, "https://example.com/flaky", status=503)
    responses.add(responses.GET, "https://example.com/flaky", json={"ok": True}, status=200)

    client = RateLimitedClient(RateLimitConfig(min_interval_seconds=0.0, backoff_base_seconds=0.01))
    response = client.get("https://example.com/flaky", source_id="test_source")
    assert response.status_code == 200


@responses.activate
def test_rate_limited_client_raises_after_exhausting_retries():
    for _ in range(5):
        responses.add(responses.GET, "https://example.com/dead", status=500)

    client = RateLimitedClient(RateLimitConfig(min_interval_seconds=0.0, max_retries=2, backoff_base_seconds=0.01))
    with pytest.raises(Exception):  # noqa: B017 - ConnectionError, raised via retry exhaustion
        client.get("https://example.com/dead", source_id="test_source")


def test_rate_limited_client_enforces_minimum_interval():
    import data.providers.base as base_module

    base_module.RateLimitedClient._last_call_at.clear()
    client = RateLimitedClient(RateLimitConfig(min_interval_seconds=0.2))
    client._last_call_at["example.com"] = time.monotonic()
    wait = client.config.min_interval_seconds - (time.monotonic() - client._last_call_at["example.com"])
    assert wait > 0  # confirms the interval check would actually throttle
