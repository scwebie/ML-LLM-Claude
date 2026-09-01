"""Version 0.2 domain schemas: real-data provenance, corporate actions,
point-in-time universes, news/events, and the read-only prediction-market
research signal.

Kept in a separate module from ``core/schemas.py`` deliberately: V0.1's
schemas (``MarketObservation``, ``FundamentalObservation``,
``MacroObservation``, ...) are untouched by V0.2 -- they remain the
provider-agnostic value tables for prices/fundamentals/macro, populated
identically whether the source is the synthetic generator or a real
provider. Provenance for *which* provider populated a given ingestion run
is tracked separately here (``DataIngestionRun``, ``DataSource``,
``DataLineage``) rather than bloating every row of the hot-path tables --
see ``docs/data_sources.md`` for the full rationale.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# Provider / provenance
# --------------------------------------------------------------------------


class ProviderCategory(StrEnum):
    PRICE = "PRICE"
    FUNDAMENTAL = "FUNDAMENTAL"
    MACRO = "MACRO"
    NEWS = "NEWS"
    EVENT_PROBABILITY = "EVENT_PROBABILITY"


class IngestionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"  # provider deliberately disabled or unreachable


class DataSource(BaseModel):
    """A registered data provider (whether currently enabled or not)."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    name: str
    category: ProviderCategory
    tier: str | None = None  # news source-quality tier; see NewsTier
    requires_api_key: bool = False
    base_url: str | None = None
    notes: str | None = None
    is_enabled: bool = True


class DataIngestionRun(BaseModel):
    """One attempt to fetch data from one provider. Always recorded, success or not."""

    run_id: str = Field(default_factory=_new_id)
    source_id: str
    category: ProviderCategory
    started_at: datetime
    finished_at: datetime | None = None
    status: IngestionStatus
    records_ingested: int = 0
    error_message: str | None = None
    symbols: list[str] = Field(default_factory=list)


class DataLineage(BaseModel):
    """Traces one feature-store row back to the source record(s) that produced it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    feature_version: str
    symbol: str
    timestamp: datetime
    source_table: str
    source_ref: str


# --------------------------------------------------------------------------
# Corporate actions / point-in-time universe
# --------------------------------------------------------------------------


class CorporateActionType(StrEnum):
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    DIVIDEND = "DIVIDEND"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    MERGER = "MERGER"
    DELISTING = "DELISTING"


class CorporateAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    symbol: str
    action_type: CorporateActionType
    ex_date: datetime
    ratio: float | None = None  # e.g. 4.0 for a 4-for-1 split
    cash_amount: float | None = None  # per-share cash dividend
    new_symbol: str | None = None  # for SYMBOL_CHANGE / MERGER
    source: str
    retrieved_at: datetime


class UniverseMembership(BaseModel):
    """One (universe, symbol) membership interval.

    ``end_date is None`` means still a member as of the most recent
    refresh. See ``docs/point_in_time_data.md`` for the survivorship-bias
    caveat: v0.2 does not have a licensed historical S&P constituent
    change feed, so membership intervals default to
    ``[today, None)`` for the configured universe unless explicitly
    back-filled from a legitimate source.
    """

    id: str = Field(default_factory=_new_id)
    universe_name: str
    symbol: str
    start_date: datetime
    end_date: datetime | None = None
    source: str
    notes: str | None = None


# --------------------------------------------------------------------------
# Price reconciliation
# --------------------------------------------------------------------------


class ReconciliationStatus(StrEnum):
    VALIDATED = "VALIDATED"
    MINOR_DIFFERENCE = "MINOR_DIFFERENCE"
    MAJOR_DIFFERENCE = "MAJOR_DIFFERENCE"
    SECONDARY_MISSING = "SECONDARY_MISSING"
    PRIMARY_MISSING = "PRIMARY_MISSING"


class PriceReconciliation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    symbol: str
    date: datetime
    primary_source: str | None = None
    primary_close: float | None = None
    secondary_source: str | None = None
    secondary_close: float | None = None
    abs_pct_diff: float | None = None
    status: ReconciliationStatus
    created_at: datetime


# --------------------------------------------------------------------------
# SEC fundamentals
# --------------------------------------------------------------------------


class SecFiling(BaseModel):
    model_config = ConfigDict(frozen=True)

    accession_number: str
    cik: str
    symbol: str
    form_type: str
    filing_period_end: datetime | None = None
    filing_date: datetime
    accepted_timestamp: datetime | None = None
    source_url: str | None = None
    retrieved_at: datetime


class FundamentalFact(BaseModel):
    """One raw XBRL fact from an SEC filing -- the authoritative, point-in-time
    unit of fundamental data. ``filed_date`` is the ONLY timestamp that may
    ever be used for an as-of join; ``period_end`` is purely descriptive."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    symbol: str
    cik: str
    tag: str  # e.g. "Revenues", "EarningsPerShareDiluted"
    unit: str  # e.g. "USD", "USD/shares", "shares"
    period_start: datetime | None = None
    period_end: datetime
    value: float
    accession_number: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    filed_date: datetime
    source: str
    retrieved_at: datetime


# --------------------------------------------------------------------------
# News / events
# --------------------------------------------------------------------------


class NewsTier(StrEnum):
    TIER_1_OFFICIAL = "TIER_1_OFFICIAL"  # SEC, central banks, government agencies
    TIER_1_MAJOR_NEWS = "TIER_1_MAJOR_NEWS"
    TIER_2_FINANCIAL_MEDIA = "TIER_2_FINANCIAL_MEDIA"
    TIER_3_OTHER = "TIER_3_OTHER"
    UNKNOWN = "UNKNOWN"


# Configured, explicit source-quality weights -- never invented by an LLM.
NEWS_TIER_WEIGHTS: dict[NewsTier, float] = {
    NewsTier.TIER_1_OFFICIAL: 1.0,
    NewsTier.TIER_1_MAJOR_NEWS: 0.85,
    NewsTier.TIER_2_FINANCIAL_MEDIA: 0.6,
    NewsTier.TIER_3_OTHER: 0.3,
    NewsTier.UNKNOWN: 0.1,
}


class NewsArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    article_id: str = Field(default_factory=_new_id)
    headline: str
    published_at: datetime | None
    retrieved_at: datetime
    source: str
    publisher: str | None = None
    tier: NewsTier
    url: str | None = None
    event_category: str | None = None
    language: str | None = "en"
    excerpt: str | None = None
    timestamp_uncertain: bool = False
    dedupe_key: str | None = None
    symbols: list[str] = Field(default_factory=list)


class NewsAgentFeatures(BaseModel):
    """LLM-structured classification of one article. Model name, prompt
    version, and generation timestamp are always recorded (Phase 17)."""

    id: str = Field(default_factory=_new_id)
    article_id: str
    symbol: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    impact_magnitude: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    event_category: str
    expected_horizon: str
    llm_model: str
    prompt_version: str
    generated_at: datetime


# --------------------------------------------------------------------------
# Read-only public event-probability research signal
#
# HARD SAFETY BOUNDARY: this schema (and every provider that populates it)
# is READ-ONLY. There is no field here for an order, a stake, a wallet
# balance, or account credentials, and there never will be -- see
# data/providers/events/prediction_market_readonly.py and
# tests/test_prediction_market_readonly.py, which asserts by introspection
# that no execution-shaped method exists anywhere on the provider class.
# --------------------------------------------------------------------------


class EventProbabilityObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    event_id: str
    question: str
    category: str | None = None
    observed_timestamp: datetime
    resolution_date: datetime | None = None
    public_probability: float = Field(ge=0.0, le=1.0)
    liquidity_metadata: dict[str, float] | None = None
    volume_metadata: dict[str, float] | None = None
    source: str
    retrieved_at: datetime


class EventSymbolMapping(BaseModel):
    """Explicit, structured relevance mapping from a public event to a
    symbol. No LLM may assign relevance without going through this
    schema's bounded, categorised field -- see agents/event_relevance.py."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    event_id: str
    symbol: str
    relevance: float = Field(ge=0.0, le=1.0)
    rationale_category: str
    created_at: datetime


# --------------------------------------------------------------------------
# Data quality / leakage auditing
# --------------------------------------------------------------------------


class QualitySeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DataQualityFlag(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    category: str
    entity_ref: str
    observation_timestamp: datetime | None = None
    flag_type: str
    severity: QualitySeverity
    details: str | None = None
    created_at: datetime


class LeakageAuditResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    run_id: str
    check_type: str
    entity_ref: str | None = None
    prediction_timestamp: datetime | None = None
    information_timestamp: datetime | None = None
    passed: bool
    details: str | None = None
    created_at: datetime


# --------------------------------------------------------------------------
# Final untouched holdout period (Stage 12)
# --------------------------------------------------------------------------


class HoldoutAccessLog(BaseModel):
    """One recorded access to the final holdout evaluation period.

    Every call to ``backtesting.holdout.evaluate_on_holdout`` writes one of
    these rows -- there is no code path that reads holdout-period rows for
    model selection, hyperparameter tuning, or feature engineering without
    it being logged here. This is the audit trail that lets a reviewer
    confirm the holdout was actually held out: it should contain exactly
    one row per model formally evaluated, all logged at the end of the
    project, never during walk-forward development."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    accessed_at: datetime
    purpose: str
    model_version: str
    holdout_start: datetime
    holdout_end: datetime
    n_rows: int
    symbols: list[str] = Field(default_factory=list)
