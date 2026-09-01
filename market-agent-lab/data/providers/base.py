"""Provider abstraction layer (Phase 2 of Version 0.2).

Defines the vendor-neutral interfaces every real-data provider implements,
plus shared infrastructure every provider client needs: an on-disk cache
(so external APIs are never hit twice for the same historical data),
best-effort rate limiting with exponential backoff, and health tracking.

The rest of the system (feature engines, agents, the model pipeline) only
ever consumes the normalized schemas in ``core.schemas`` /
``core.schemas_v2`` -- never a vendor-specific response object. Swapping
or adding a provider means writing one new class here; nothing downstream
changes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import requests

from core.config import settings
from core.logging import get_logger
from core.schemas_v2 import (
    DataIngestionRun,
    EventProbabilityObservation,
    FundamentalFact,
    IngestionStatus,
    NewsArticle,
    ProviderCategory,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Vendor-neutral interfaces
# --------------------------------------------------------------------------


@runtime_checkable
class PriceDataProvider(Protocol):
    source_id: str

    def get_daily_bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """Return raw daily bars: list of dicts with at least
        open/high/low/close/volume/date keys, in the provider's own units.
        Normalisation into ``core.schemas.MarketObservation`` happens in
        the caller (``data/providers/prices`` reconciliation layer), not
        here, so each provider stays a thin, testable HTTP client."""
        ...


@runtime_checkable
class FundamentalDataProvider(Protocol):
    source_id: str

    def get_company_fundamentals(self, symbol: str) -> list[FundamentalFact]:
        ...


@runtime_checkable
class MacroDataProvider(Protocol):
    source_id: str

    def get_series(self, series_id: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        ...


@runtime_checkable
class NewsProvider(Protocol):
    source_id: str

    def search_news(self, query: str, start: datetime, end: datetime) -> list[NewsArticle]:
        ...


@runtime_checkable
class EventProbabilityDataProvider(Protocol):
    """Read-only. See data/providers/events/prediction_market_readonly.py
    for the hard safety guarantee that no implementation of this Protocol
    may expose an order/wager/wallet/auth method."""

    source_id: str

    def get_active_events(self, category: str | None = None) -> list[EventProbabilityObservation]:
        ...


# --------------------------------------------------------------------------
# Provider health
# --------------------------------------------------------------------------


@dataclass
class ProviderHealth:
    source_id: str
    category: ProviderCategory
    enabled: bool = True
    last_success: datetime | None = None
    last_failure: datetime | None = None
    last_error: str | None = None
    records_downloaded: int = 0
    last_latency_ms: float | None = None
    rate_limited: bool = False
    unavailable_reason: str | None = None  # e.g. "network policy denial", "requires API key"


class ProviderHealthTracker:
    """In-memory health registry for the lifetime of a process; also
    persisted per ingestion attempt via ``database.repository`` so the
    dashboard/API can show provider status across runs."""

    def __init__(self) -> None:
        self._health: dict[str, ProviderHealth] = {}

    def get(self, source_id: str) -> ProviderHealth | None:
        return self._health.get(source_id)

    def record_success(self, source_id: str, category: ProviderCategory, records: int, latency_ms: float) -> None:
        h = self._health.setdefault(source_id, ProviderHealth(source_id=source_id, category=category))
        h.last_success = datetime.now(UTC)
        h.records_downloaded += records
        h.last_latency_ms = latency_ms
        h.rate_limited = False

    def record_failure(self, source_id: str, category: ProviderCategory, error: str, rate_limited: bool = False) -> None:
        h = self._health.setdefault(source_id, ProviderHealth(source_id=source_id, category=category))
        h.last_failure = datetime.now(UTC)
        h.last_error = error
        h.rate_limited = rate_limited

    def mark_unavailable(self, source_id: str, category: ProviderCategory, reason: str) -> None:
        h = self._health.setdefault(source_id, ProviderHealth(source_id=source_id, category=category))
        h.enabled = False
        h.unavailable_reason = reason

    def all(self) -> dict[str, ProviderHealth]:
        return dict(self._health)


HEALTH = ProviderHealthTracker()


# --------------------------------------------------------------------------
# On-disk cache
# --------------------------------------------------------------------------

CACHE_DIR = settings.data_store_dir / "cache"


def _cache_key(namespace: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"namespace": namespace, **params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_get(namespace: str, params: dict[str, Any], max_age_seconds: float | None = None) -> Any | None:
    """Return cached JSON payload for (namespace, params), or None on a
    miss / stale entry. ``max_age_seconds=None`` means cached data never
    expires (appropriate for immutable historical data)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(namespace, params)}.json"
    if not path.exists():
        return None
    if max_age_seconds is not None:
        age = time.time() - path.stat().st_mtime
        if age > max_age_seconds:
            return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def cache_set(namespace: str, params: dict[str, Any], payload: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(namespace, params)}.json"
    path.write_text(json.dumps(payload, default=str))


def cached_fetch(
    namespace: str,
    params: dict[str, Any],
    fetch_fn,
    force_refresh: bool = False,
    max_age_seconds: float | None = None,
) -> Any:
    """Reuse previously downloaded data by default; only calls ``fetch_fn``
    on a cache miss or when ``force_refresh=True``."""
    if not force_refresh:
        cached = cache_get(namespace, params, max_age_seconds)
        if cached is not None:
            return cached
    result = fetch_fn()
    cache_set(namespace, params, result)
    return result


# --------------------------------------------------------------------------
# Rate-limited HTTP client with retry/backoff
# --------------------------------------------------------------------------


@dataclass
class RateLimitConfig:
    min_interval_seconds: float = 0.5
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    timeout_seconds: float = 20.0


class RateLimitedClient:
    """Thin wrapper around ``requests`` enforcing a minimum interval
    between requests per host and retrying with exponential backoff on
    429/5xx/timeout. Provider clients should route every HTTP call through
    this rather than calling ``requests`` directly."""

    _last_call_at: dict[str, float] = {}

    def __init__(self, config: RateLimitConfig | None = None, user_agent: str | None = None) -> None:
        self.config = config or RateLimitConfig()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent or "market-agent-lab research (contact: research@example.com)"

    def get(self, url: str, source_id: str, **kwargs) -> requests.Response:
        host = url.split("/")[2] if "://" in url else url
        last = self._last_call_at.get(host, 0.0)
        wait = self.config.min_interval_seconds - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.config.timeout_seconds, **kwargs)
                self._last_call_at[host] = time.monotonic()
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", self.config.backoff_base_seconds * (2**attempt)))
                    logger.warning("rate_limited", source_id=source_id, retry_after=retry_after)
                    time.sleep(retry_after)
                    continue
                if response.status_code >= 500:
                    time.sleep(self.config.backoff_base_seconds * (2**attempt))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:  # noqa: PERF203
                last_exc = exc
                time.sleep(self.config.backoff_base_seconds * (2**attempt))
        raise ConnectionError(f"{source_id}: request to {url} failed after retries") from last_exc


def make_ingestion_run(source_id: str, category: ProviderCategory) -> DataIngestionRun:
    return DataIngestionRun(source_id=source_id, category=category, started_at=datetime.now(UTC), status=IngestionStatus.FAILED)


def finish_ingestion_run(run: DataIngestionRun, status: IngestionStatus, records: int = 0, error: str | None = None) -> DataIngestionRun:
    return run.model_copy(update={"finished_at": datetime.now(UTC), "status": status, "records_ingested": records, "error_message": error})
