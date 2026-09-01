"""Bureau of Labor Statistics macro provider (Phase 6).

Works unregistered (BLS public API v2) at a lower daily-request cap; set
``BLS_API_KEY`` to raise it -- not required for the request volumes this
project needs. Same vintage-timestamp limitation as ``fred.py``: BLS's
public API does not expose the historical release timestamp per data
point, so ``typical_lag_days`` (BLS's own well-documented, regular release
schedule) is used as a conservative publication-timestamp approximation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.schemas_v2 import ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "bls"
BASE_URL = "https://api.bls.gov/publicAPI/v2"

_PERIOD_TO_MONTH = {f"M{i:02d}": i for i in range(1, 13)}


@dataclass(frozen=True)
class BlsSeriesMeta:
    series_id: str
    name: str
    typical_lag_days: int  # from period end to BLS release, per BLS's published schedule


SERIES_CATALOG: dict[str, BlsSeriesMeta] = {
    "CUUR0000SA0": BlsSeriesMeta("CUUR0000SA0", "CPI-U (all items)", 13),
    "CUUR0000SA0L1E": BlsSeriesMeta("CUUR0000SA0L1E", "Core CPI (ex food & energy)", 13),
    "WPUFD4": BlsSeriesMeta("WPUFD4", "PPI (final demand)", 13),
    "LNS14000000": BlsSeriesMeta("LNS14000000", "Unemployment Rate", 5),
    "CES0000000001": BlsSeriesMeta("CES0000000001", "Total Nonfarm Payrolls", 5),
    "CES0500000003": BlsSeriesMeta("CES0500000003", "Average Hourly Earnings", 5),
    "JTS000000000000000JOL": BlsSeriesMeta("JTS000000000000000JOL", "JOLTS Job Openings", 35),
}


class BlsProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient()

    def get_series(self, series_id: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"startyear": str(start.year), "endyear": str(end.year)}
        if os.getenv("BLS_API_KEY"):
            params["registrationkey"] = os.getenv("BLS_API_KEY")

        def fetch() -> dict[str, Any]:
            response = self.client.get(f"{BASE_URL}/timeseries/data/{series_id}", source_id=self.source_id, params=params)
            payload = response.json()
            if payload.get("status") != "REQUEST_SUCCEEDED":
                raise ValueError(f"bls error for {series_id}: {payload.get('message')}")
            return payload

        try:
            payload = cached_fetch(
                namespace="bls_series", params={"series_id": series_id, "start_year": start.year, "end_year": end.year},
                fetch_fn=fetch, max_age_seconds=24 * 3600,
            )
            HEALTH.record_success(self.source_id, ProviderCategory.MACRO, records=1, latency_ms=0.0)
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.MACRO, str(exc))
            raise

        meta = SERIES_CATALOG.get(series_id, BlsSeriesMeta(series_id, series_id, 30))
        series_list = payload.get("Results", {}).get("series", [])
        if not series_list:
            return []

        rows = []
        for entry in series_list[0].get("data", []):
            period = entry.get("period")
            month = _PERIOD_TO_MONTH.get(period)
            if month is None:  # skip annual (M13) or unrecognised periods
                continue
            year = int(entry["year"])
            # Period-end date = last day of the reported month, approximated as day 28 + roll to next month start - 1 day.
            if month == 12:
                period_end = datetime(year, 12, 31)
            else:
                period_end = datetime(year, month + 1, 1) - timedelta(days=1)
            if period_end < start or period_end > end:
                continue
            rows.append(
                {
                    "date": period_end, "value": float(entry["value"]),
                    "publication_date": period_end + timedelta(days=meta.typical_lag_days),
                }
            )
        return sorted(rows, key=lambda r: r["date"])
