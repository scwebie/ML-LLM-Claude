"""Bureau of Economic Analysis macro provider (Phase 6).

Dataset-list metadata is reachable unauthenticated, but real series
retrieval (GDP, PCE, personal income, corporate profits) requires a
registered ``BEA_API_KEY`` (free at https://apps.bea.gov/api/signup/).
Without one, ``data/providers/registry.py`` marks this source disabled
and callers should skip it -- this class still raises a clear,
actionable error rather than returning fabricated data if called anyway.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from data.providers.base import RateLimitedClient, cached_fetch

SOURCE_ID = "bea"
BASE_URL = "https://apps.bea.gov/api/data/"


class BeaProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        self.client = client or RateLimitedClient()

    def get_series(self, table_name: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        api_key = os.getenv("BEA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "BEA_API_KEY is not configured. BEA's dataset-list metadata is reachable without a "
                "key, but real series retrieval is not; register a free key at "
                "https://apps.bea.gov/api/signup/ and set BEA_API_KEY to enable this provider."
            )
        params = {"UserID": api_key, "method": "GetData", "datasetname": "NIPA", "TableName": table_name, "Frequency": "Q", "Year": "ALL", "ResultFormat": "JSON"}

        def fetch() -> dict[str, Any]:
            response = self.client.get(BASE_URL, source_id=self.source_id, params=params)
            return response.json()

        payload = cached_fetch(namespace="bea_series", params={"table": table_name}, fetch_fn=fetch, max_age_seconds=24 * 3600)
        return payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
