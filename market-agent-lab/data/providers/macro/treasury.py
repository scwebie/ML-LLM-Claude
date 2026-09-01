"""US Treasury Fiscal Data API macro provider (Phase 6).

No API key required. Uses the "Average Interest Rates on U.S. Treasury
Securities" dataset -- distinct from (and complementary to) the FRED
Treasury *yield* series in ``fred.py``: this reports the average interest
rate the government is actually paying across its marketable debt, a
useful fiscal-conditions signal in its own right, not a yield-curve
duplicate.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from core.schemas_v2 import ProviderCategory
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "treasury"
BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

# Treasury publishes this dataset with a documented ~2-3 week lag after month end.
TYPICAL_LAG_DAYS = 20


class TreasuryProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient()

    def get_series(self, security_desc: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        """``security_desc`` e.g. "Treasury Notes", "Treasury Bonds", "Treasury Bills"."""
        params = {
            "filter": f"record_date:gte:{start.date()},record_date:lte:{end.date()},security_desc:eq:{security_desc}",
            "sort": "record_date", "page[size]": "10000",
        }

        def fetch() -> dict[str, Any]:
            response = self.client.get(f"{BASE_URL}/v2/accounting/od/avg_interest_rates", source_id=self.source_id, params=params)
            return response.json()

        try:
            payload = cached_fetch(
                namespace="treasury_avg_interest_rates",
                params={"security_desc": security_desc, "start": start.isoformat(), "end": end.isoformat()},
                fetch_fn=fetch, max_age_seconds=24 * 3600,
            )
            HEALTH.record_success(self.source_id, ProviderCategory.MACRO, records=len(payload.get("data", [])), latency_ms=0.0)
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.MACRO, str(exc))
            raise

        rows = []
        for row in payload.get("data", []):
            date = datetime.strptime(row["record_date"], "%Y-%m-%d")
            rows.append(
                {
                    "date": date, "value": float(row["avg_interest_rate_amt"]),
                    "publication_date": date + timedelta(days=TYPICAL_LAG_DAYS),
                }
            )
        return rows
