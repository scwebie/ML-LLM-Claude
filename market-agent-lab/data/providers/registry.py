"""Provider registry: the single place that knows which providers exist,
whether each is currently enabled, and why.

A provider is enabled only if: (a) it doesn't need a key, or its key is
configured, AND (b) it isn't in the hard-coded ``KNOWN_UNAVAILABLE`` list
documenting sources probed during development that turned out not to be
usable from an automated HTTP client (see docs/data_sources.md for the
per-source rationale -- e.g. Stooq requires a JavaScript proof-of-work
challenge; GDELT's API host was unreachable). This registry is what
``main.py``'s ``ingest-*`` commands and the dashboard's Data tab both read
to decide what actually ran.
"""

from __future__ import annotations

import os

from core.schemas_v2 import DataSource, NewsTier, ProviderCategory

# Sources probed during development that cannot be used as a plain HTTP
# client from this codebase, with the concrete reason. Not a policy
# decision -- purely a technical/legal usability finding. If one of these
# becomes usable (e.g. a headless-browser bridge, or the host becomes
# reachable), remove it here and re-enable normally.
KNOWN_UNAVAILABLE: dict[str, str] = {
    "stooq": "returns a JavaScript proof-of-work bot-verification challenge instead of data; not solvable by a plain HTTP client",
    "gdelt": "api.gdeltproject.org is unreachable from this network (connection reset/timeout on every attempt)",
}


def _has_key(env_var: str) -> bool:
    return bool(os.getenv(env_var))


PROVIDER_CATALOG: list[DataSource] = [
    # --- Prices ---------------------------------------------------------------
    DataSource(
        source_id="yahoo_finance", name="Yahoo Finance (chart API)", category=ProviderCategory.PRICE,
        requires_api_key=False, base_url="https://query1.finance.yahoo.com",
        notes="Primary price source. No key required; needs a browser-like User-Agent.",
        is_enabled="yahoo_finance" not in KNOWN_UNAVAILABLE,
    ),
    DataSource(
        source_id="stockanalysis", name="StockAnalysis.com API", category=ProviderCategory.PRICE,
        requires_api_key=False, base_url="https://stockanalysis.com/api",
        notes="Secondary/validation price source, independent of Yahoo. No key required.",
        is_enabled="stockanalysis" not in KNOWN_UNAVAILABLE,
    ),
    DataSource(
        source_id="stooq", name="Stooq", category=ProviderCategory.PRICE, requires_api_key=False,
        base_url="https://stooq.com", notes=KNOWN_UNAVAILABLE.get("stooq"), is_enabled=False,
    ),
    # --- Fundamentals -----------------------------------------------------------
    DataSource(
        source_id="sec_edgar", name="SEC EDGAR XBRL company facts", category=ProviderCategory.FUNDAMENTAL,
        requires_api_key=False, base_url="https://data.sec.gov",
        notes="Authoritative source for US-company fundamentals. No key; requires a descriptive User-Agent per SEC fair-access policy.",
        is_enabled=True,
    ),
    # --- Macro --------------------------------------------------------------------
    DataSource(
        source_id="fred", name="FRED (fredgraph.csv)", category=ProviderCategory.MACRO, requires_api_key=False,
        base_url="https://fred.stlouisfed.org", notes="CSV download endpoint; no key required.", is_enabled=True,
    ),
    DataSource(
        source_id="bls", name="Bureau of Labor Statistics public API v2", category=ProviderCategory.MACRO,
        requires_api_key=False, base_url="https://api.bls.gov",
        notes="Works unregistered at a lower daily request cap; BLS_API_KEY raises the cap.", is_enabled=True,
    ),
    DataSource(
        source_id="treasury", name="US Treasury Fiscal Data API", category=ProviderCategory.MACRO,
        requires_api_key=False, base_url="https://api.fiscaldata.treasury.gov", notes="No key required.", is_enabled=True,
    ),
    DataSource(
        source_id="bea", name="Bureau of Economic Analysis API", category=ProviderCategory.MACRO,
        requires_api_key=True, base_url="https://apps.bea.gov/api",
        notes="Dataset-list metadata is reachable unauthenticated, but real series retrieval requires a registered BEA_API_KEY.",
        is_enabled=_has_key("BEA_API_KEY"),
    ),
    # --- News -----------------------------------------------------------------------
    DataSource(
        source_id="sec_events", name="SEC 8-K filings (as news/event source)", category=ProviderCategory.NEWS,
        tier=NewsTier.TIER_1_OFFICIAL.value, requires_api_key=False, base_url="https://data.sec.gov",
        notes="Official corporate event disclosures; same client as the fundamentals provider.", is_enabled=True,
    ),
    DataSource(
        source_id="gdelt", name="GDELT Doc API", category=ProviderCategory.NEWS,
        tier=NewsTier.TIER_2_FINANCIAL_MEDIA.value, requires_api_key=False,
        base_url="https://api.gdeltproject.org", notes=KNOWN_UNAVAILABLE.get("gdelt"), is_enabled=False,
    ),
    DataSource(
        source_id="company_ir", name="Company investor-relations RSS feeds", category=ProviderCategory.NEWS,
        tier=NewsTier.TIER_1_OFFICIAL.value, requires_api_key=False, base_url=None,
        notes="No universal free feed registry exists; per-company RSS URLs must be configured explicitly. Disabled by default.",
        is_enabled=False,
    ),
    DataSource(
        source_id="news_api", name="NewsAPI.org", category=ProviderCategory.NEWS,
        tier=NewsTier.TIER_2_FINANCIAL_MEDIA.value, requires_api_key=True, base_url="https://newsapi.org",
        notes="Requires NEWS_API_KEY.", is_enabled=_has_key("NEWS_API_KEY"),
    ),
    # --- Read-only prediction-market research signal ---------------------------------
    DataSource(
        source_id="polymarket_readonly", name="Polymarket (gamma-api, read-only)", category=ProviderCategory.EVENT_PROBABILITY,
        requires_api_key=False, base_url="https://gamma-api.polymarket.com",
        notes="Public read-only market-probability data. No key. No order/wallet/auth capability exists in this codebase.",
        is_enabled=True,
    ),
]


def get_catalog() -> list[DataSource]:
    return list(PROVIDER_CATALOG)


def get_enabled(category: ProviderCategory | None = None) -> list[DataSource]:
    sources = [s for s in PROVIDER_CATALOG if s.is_enabled]
    if category is not None:
        sources = [s for s in sources if s.category == category]
    return sources


def get_source(source_id: str) -> DataSource | None:
    for s in PROVIDER_CATALOG:
        if s.source_id == source_id:
            return s
    return None
