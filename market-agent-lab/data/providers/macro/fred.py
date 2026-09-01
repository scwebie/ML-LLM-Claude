"""FRED macro data provider (Phase 6) via the keyless ``fredgraph.csv``
download endpoint.

Known limitation (documented, not silently glossed over): the full FRED
API's realtime/ALFRED vintage data (exact historical publication and
revision timestamps) requires a registered ``FRED_API_KEY``, which is not
configured here. Without it, this provider only has the *current*
(latest-revised) series values. For genuinely daily market-rate series
(Treasury yields, Fed funds) that is not actually a limitation -- those
publish same-day/next-business-day with no meaningful revision, so
``typical_lag_days`` below is exact. See ``docs/data_sources.md``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.schemas_v2 import ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "fred"
BASE_URL = "https://fred.stlouisfed.org"


@dataclass(frozen=True)
class FredSeriesMeta:
    series_id: str
    name: str
    typical_lag_days: int  # calendar days after the observation date until it was/would be publicly known


# A representative, useful subset (brief section 8's examples), all
# confirmed reachable via the keyless CSV endpoint.
SERIES_CATALOG: dict[str, FredSeriesMeta] = {
    "DGS10": FredSeriesMeta("DGS10", "10-Year Treasury Yield", 1),
    "DGS2": FredSeriesMeta("DGS2", "2-Year Treasury Yield", 1),
    "DGS3MO": FredSeriesMeta("DGS3MO", "3-Month Treasury Yield", 1),
    "T10Y2Y": FredSeriesMeta("T10Y2Y", "10Y-2Y Treasury Spread", 1),
    "T10Y3M": FredSeriesMeta("T10Y3M", "10Y-3M Treasury Spread", 1),
    "FEDFUNDS": FredSeriesMeta("FEDFUNDS", "Effective Federal Funds Rate (monthly)", 3),
    "INDPRO": FredSeriesMeta("INDPRO", "Industrial Production Index", 45),
    "VIXCLS": FredSeriesMeta("VIXCLS", "CBOE Volatility Index", 1),
}


class FredProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient()

    def get_series(self, series_id: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        def fetch() -> str:
            response = self.client.get(f"{BASE_URL}/graph/fredgraph.csv", source_id=self.source_id, params={"id": series_id})
            return response.text

        try:
            text = cached_fetch(namespace="fred_series", params={"series_id": series_id}, fetch_fn=fetch, max_age_seconds=24 * 3600)
            HEALTH.record_success(self.source_id, ProviderCategory.MACRO, records=1, latency_ms=0.0)
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.MACRO, str(exc))
            raise

        reader = csv.DictReader(io.StringIO(text))
        meta = SERIES_CATALOG.get(series_id, FredSeriesMeta(series_id, series_id, 7))
        rows = []
        for row in reader:
            date_str = row.get("observation_date")
            value_str = row.get(series_id)
            if not date_str or value_str in (None, ".", ""):
                continue
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date < start or date > end:
                continue
            rows.append(
                {
                    "date": date, "value": float(value_str),
                    "publication_date": date + timedelta(days=meta.typical_lag_days),
                }
            )
        return rows
