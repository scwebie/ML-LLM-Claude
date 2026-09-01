"""Secondary/validation real price provider: stockanalysis.com API.

Independent of Yahoo Finance (different vendor, different infrastructure)
-- used purely for cross-source reconciliation, never as the sole source
of truth. No API key required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.schemas_v2 import ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "stockanalysis"
BASE_URL = "https://stockanalysis.com/api"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


class StockAnalysisPriceProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient(user_agent=_BROWSER_UA)

    def get_daily_bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        url = f"{BASE_URL}/symbol/s/{symbol}/history"
        params = {"range": "10Y", "period": "Daily"}

        def fetch() -> dict[str, Any]:
            import time

            t0 = time.monotonic()
            response = self.client.get(url, source_id=self.source_id, params=params)
            latency = (time.monotonic() - t0) * 1000
            payload = response.json()
            if payload.get("status") != 200:
                raise ValueError(f"stockanalysis error for {symbol}: {payload}")
            HEALTH.record_success(self.source_id, ProviderCategory.PRICE, records=len(payload.get("data", [])), latency_ms=latency)
            return payload

        try:
            payload = cached_fetch(
                namespace="stockanalysis_daily_bars",
                params={"symbol": symbol},  # full history cached once per symbol; filtered by date below
                fetch_fn=fetch,
            )
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.PRICE, str(exc))
            raise

        bars = []
        for row in payload.get("data", []):
            date = datetime.strptime(row["t"], "%Y-%m-%d")
            if date < start or date > end:
                continue
            bars.append(
                {
                    "date": date, "open": float(row["o"]), "high": float(row["h"]), "low": float(row["l"]),
                    "close": float(row["c"]), "adjusted_close": float(row.get("a", row["c"])),
                    "volume": int(row.get("v", 0)),
                }
            )
        return sorted(bars, key=lambda b: b["date"])
