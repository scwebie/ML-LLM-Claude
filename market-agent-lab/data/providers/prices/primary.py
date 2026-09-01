"""Primary real price provider: Yahoo Finance chart API.

No API key required. The endpoint blocks the default ``requests``/``curl``
User-Agent with HTTP 429, but responds normally to a browser-like one --
this is Yahoo's own bot-mitigation, not a network policy issue, so a
realistic ``User-Agent`` header is set on every request.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.schemas_v2 import ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "yahoo_finance"
BASE_URL = "https://query1.finance.yahoo.com"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


class YahooFinancePriceProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient(user_agent=_BROWSER_UA)

    def _fetch_raw(self, symbol: str, start: datetime, end: datetime) -> dict[str, Any]:
        period1 = int(start.replace(tzinfo=UTC).timestamp())
        period2 = int(end.replace(tzinfo=UTC).timestamp())
        url = f"{BASE_URL}/v8/finance/chart/{symbol}"
        params = {"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"}

        def fetch() -> dict[str, Any]:
            import time

            t0 = time.monotonic()
            response = self.client.get(url, source_id=self.source_id, params=params)
            latency = (time.monotonic() - t0) * 1000
            payload = response.json()
            error = payload.get("chart", {}).get("error")
            if error:
                raise ValueError(f"yahoo_finance error for {symbol}: {error}")
            HEALTH.record_success(self.source_id, ProviderCategory.PRICE, records=1, latency_ms=latency)
            return payload

        try:
            return cached_fetch(
                namespace="yahoo_daily_bars",
                params={"symbol": symbol, "start": start.isoformat(), "end": end.isoformat()},
                fetch_fn=fetch,
            )
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.PRICE, str(exc))
            raise

    def get_daily_bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        payload = self._fetch_raw(symbol, start, end)
        result = payload.get("chart", {}).get("result")
        if not result:
            return []
        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        adjclose = result.get("indicators", {}).get("adjclose", [{}])
        adjclose = adjclose[0].get("adjclose") if adjclose else [None] * len(timestamps)

        bars = []
        for i, ts in enumerate(timestamps):
            o, h, low, c, v = (quote.get(k, [None] * len(timestamps))[i] for k in ("open", "high", "low", "close", "volume"))
            if o is None or h is None or low is None or c is None:
                continue  # a genuine missing/halted session -- skip, never fabricate
            a = adjclose[i] if adjclose and i < len(adjclose) else c
            bars.append(
                {
                    "date": datetime.fromtimestamp(ts, tz=UTC).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0),
                    "open": float(o), "high": float(h), "low": float(low), "close": float(c),
                    "adjusted_close": float(a) if a is not None else float(c),
                    "volume": int(v) if v is not None else 0,
                }
            )
        return bars

    def get_corporate_action_events(self, symbol: str, start: datetime, end: datetime) -> dict[str, Any]:
        """Raw dividend/split event payload from the same chart response
        (``events=div,splits`` was requested above), used by
        ``data/corporate_actions.py`` -- kept as raw vendor shape here
        deliberately, normalisation happens at the call site."""
        payload = self._fetch_raw(symbol, start, end)
        result = payload.get("chart", {}).get("result")
        if not result:
            return {}
        return result[0].get("events", {})
