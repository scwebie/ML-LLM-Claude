"""SEC EDGAR fundamentals provider (Phase 5) -- the authoritative source
for US-company structured fundamental data.

No API key required; SEC's fair-access policy requires only a descriptive
``User-Agent`` identifying the requester (enforced below). Every fact
carries its ``filed_date`` -- the ONLY timestamp ever used for an as-of
join anywhere downstream (see ``database/repository_v2.py::
get_fundamental_facts_asof``). ``period_end`` is purely descriptive of
what the number covers and must never be used to filter what a
point-in-time query can see.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.schemas_v2 import FundamentalFact, ProviderCategory, SecFiling
from data.providers.base import HEALTH, RateLimitedClient, cached_fetch

SOURCE_ID = "sec_edgar"
BASE_URL = "https://data.sec.gov"
WWW_BASE_URL = "https://www.sec.gov"
_UA = "market-agent-lab research (contact: research@example.com)"

# Concept -> candidate us-gaap XBRL tags, most-recent-standard first. SEC
# tag usage has changed over time (e.g. ASC 606 revenue recognition
# changed the standard revenue tag around 2018); trying each in order and
# merging whatever is present is the practical way to get a continuous
# series without silently picking a tag that simply doesn't exist for a
# given company/period.
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
}

RELEVANT_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}


class SecEdgarFundamentalProvider:
    source_id = SOURCE_ID

    def __init__(self, client: RateLimitedClient | None = None) -> None:
        # SEC asks for >=100ms between requests per their fair-access guidance;
        self.client = client or RateLimitedClient(user_agent=_UA)
        from data.providers.base import RateLimitConfig

        self.client.config = RateLimitConfig(min_interval_seconds=0.15, max_retries=3, backoff_base_seconds=1.0)

    def resolve_cik(self, symbol: str) -> str | None:
        def fetch() -> dict[str, Any]:
            response = self.client.get(f"{WWW_BASE_URL}/files/company_tickers.json", source_id=self.source_id)
            return response.json()

        mapping = cached_fetch(namespace="sec_company_tickers", params={}, fetch_fn=fetch, max_age_seconds=7 * 24 * 3600)
        for row in mapping.values():
            if row["ticker"].upper() == symbol.upper():
                return str(row["cik_str"]).zfill(10)
        return None

    def _get_company_facts(self, cik: str) -> dict[str, Any]:
        def fetch() -> dict[str, Any]:
            import time

            t0 = time.monotonic()
            response = self.client.get(f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json", source_id=self.source_id)
            latency = (time.monotonic() - t0) * 1000
            HEALTH.record_success(self.source_id, ProviderCategory.FUNDAMENTAL, records=1, latency_ms=latency)
            return response.json()

        try:
            return cached_fetch(namespace="sec_company_facts", params={"cik": cik}, fetch_fn=fetch)
        except Exception as exc:  # noqa: BLE001
            HEALTH.record_failure(self.source_id, ProviderCategory.FUNDAMENTAL, str(exc))
            raise

    def get_company_fundamentals(self, symbol: str) -> list[FundamentalFact]:
        cik = self.resolve_cik(symbol)
        if cik is None:
            return []
        facts_payload = self._get_company_facts(cik)
        us_gaap = facts_payload.get("facts", {}).get("us-gaap", {})
        retrieved_at = datetime.now(UTC)

        results: list[FundamentalFact] = []
        for concept, candidate_tags in CONCEPT_TAGS.items():
            for xbrl_tag in candidate_tags:
                tag_data = us_gaap.get(xbrl_tag)
                if not tag_data:
                    continue
                for unit, entries in tag_data.get("units", {}).items():
                    for entry in entries:
                        form = entry.get("form")
                        if form not in RELEVANT_FORMS:
                            continue
                        filed = entry.get("filed")
                        end = entry.get("end")
                        if not filed or not end:
                            continue
                        results.append(
                            FundamentalFact(
                                symbol=symbol, cik=cik, tag=concept, unit=unit,
                                period_start=datetime.strptime(entry["start"], "%Y-%m-%d") if entry.get("start") else None,
                                period_end=datetime.strptime(end, "%Y-%m-%d"),
                                value=float(entry["val"]), accession_number=entry.get("accn"),
                                form_type=form, fiscal_year=entry.get("fy"), fiscal_period=entry.get("fp"),
                                filed_date=datetime.strptime(filed, "%Y-%m-%d"),
                                source=self.source_id, retrieved_at=retrieved_at,
                            )
                        )
                break  # first candidate tag with any data wins for this concept

        return results

    def get_filings(self, symbol: str, form_types: set[str] | None = None) -> list[SecFiling]:
        """Filing metadata from the submissions endpoint -- also the basis
        for the SEC-8-K-as-news source in Stage 8 (`form_types={"8-K"}`)."""
        cik = self.resolve_cik(symbol)
        if cik is None:
            return []

        def fetch() -> dict[str, Any]:
            response = self.client.get(f"{BASE_URL}/submissions/CIK{cik}.json", source_id=self.source_id)
            return response.json()

        payload = cached_fetch(namespace="sec_submissions", params={"cik": cik}, fetch_fn=fetch, max_age_seconds=24 * 3600)
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accepted = recent.get("acceptanceDateTime", [])
        accns = recent.get("accessionNumber", [])
        periods = recent.get("reportDate", [])
        retrieved_at = datetime.now(UTC)

        filings: list[SecFiling] = []
        for i, form in enumerate(forms):
            if form_types is not None and form not in form_types:
                continue
            filings.append(
                SecFiling(
                    accession_number=accns[i], cik=cik, symbol=symbol, form_type=form,
                    filing_period_end=datetime.strptime(periods[i], "%Y-%m-%d") if i < len(periods) and periods[i] else None,
                    filing_date=datetime.strptime(dates[i], "%Y-%m-%d"),
                    accepted_timestamp=datetime.fromisoformat(accepted[i]) if i < len(accepted) and accepted[i] else None,
                    source_url=f"{WWW_BASE_URL}/Archives/edgar/data/{int(cik)}/{accns[i].replace('-', '')}",
                    retrieved_at=retrieved_at,
                )
            )
        return filings
