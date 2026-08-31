"""Strongly typed domain models shared by every layer of market-agent-lab.

These are the Phase 1 data contracts. A few design rules apply everywhere:

* Every observation that could leak future information into a backtest
  (fundamentals, macro data) carries an explicit ``publication_timestamp``
  (the moment the information became publicly knowable) *separate* from
  the ``timestamp``/``period`` it describes. Feature builders must always
  join on ``publication_timestamp <= as_of``, never on the period end date,
  or they will silently introduce look-ahead bias.
* ``ModelPrediction``, ``Outcome``, and ``PaperFill`` are immutable records
  (``frozen=True``): once a prediction or a fill is written it is a fact of
  history and must never be mutated in place. Corrections happen by writing
  a new row, never by editing an old one.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class RiskApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RiskReasonCode(StrEnum):
    """Explicit, exhaustive reason codes returned by the deterministic risk engine."""

    OK = "OK"
    RISK_POSITION_LIMIT = "RISK_POSITION_LIMIT"
    RISK_GROSS_EXPOSURE = "RISK_GROSS_EXPOSURE"
    RISK_NET_EXPOSURE = "RISK_NET_EXPOSURE"
    RISK_SECTOR_CONCENTRATION = "RISK_SECTOR_CONCENTRATION"
    RISK_PORTFOLIO_VOLATILITY = "RISK_PORTFOLIO_VOLATILITY"
    RISK_DAILY_LOSS_LIMIT = "RISK_DAILY_LOSS_LIMIT"
    RISK_DRAWDOWN = "RISK_DRAWDOWN"
    STALE_DATA = "STALE_DATA"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    INVALID_PRICE = "INVALID_PRICE"
    KILL_SWITCH = "KILL_SWITCH"


class VolatilityRegime(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"


class GrowthRegime(StrEnum):
    CONTRACTION = "CONTRACTION"
    SLOWDOWN = "SLOWDOWN"
    EXPANSION = "EXPANSION"
    OVERHEATING = "OVERHEATING"


class InflationRegime(StrEnum):
    DEFLATIONARY = "DEFLATIONARY"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class MonetaryPolicyRegime(StrEnum):
    EASING = "EASING"
    NEUTRAL = "NEUTRAL"
    TIGHTENING = "TIGHTENING"


class PromotionDecision(StrEnum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


# --------------------------------------------------------------------------
# Market / fundamental / macro observations
# --------------------------------------------------------------------------


class MarketObservation(BaseModel):
    """A single daily OHLCV bar for one symbol. Synthetic in v0.1."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    adjusted_close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @field_validator("high")
    @classmethod
    def _high_is_high(cls, v: float, info) -> float:  # noqa: ANN001
        low = info.data.get("low")
        if low is not None and v < low:
            raise ValueError("high must be >= low")
        return v


class FundamentalObservation(BaseModel):
    """A fundamental data point as it would have been known at publication time.

    ``publication_timestamp`` MUST be strictly used for any as-of join --
    ``reporting_period_end`` is only descriptive of what period the figures
    cover, and using it for joins is exactly how look-ahead bias creeps in
    (real companies report Q4 numbers ~6-8 weeks after quarter end).
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    publication_timestamp: datetime
    reporting_period_end: datetime

    revenue: float
    revenue_growth: float | None = None
    eps: float
    eps_growth: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    free_cash_flow: float | None = None
    fcf_margin: float | None = None
    roic: float | None = None
    debt: float | None = None
    cash: float | None = None

    # Valuation metrics, where available
    pe_ratio: float | None = None
    ev_to_ebitda: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None


class MacroObservation(BaseModel):
    """One macro time-series print, with vintage tracking.

    ``timestamp`` is the period the observation describes (e.g. the month a
    CPI reading covers). ``publication_timestamp`` is when it was first
    released. ``vintage_timestamp`` allows storing later revisions of the
    same period without overwriting the originally-published value used by
    historical backtests.
    """

    model_config = ConfigDict(frozen=True)

    series_name: str
    timestamp: datetime
    value: float
    publication_timestamp: datetime
    vintage_timestamp: datetime | None = None


# --------------------------------------------------------------------------
# Agent output
# --------------------------------------------------------------------------


class AgentReport(BaseModel):
    """Structured output emitted by any research agent.

    Agents never execute trades -- they only ever produce ``AgentReport``
    instances that downstream feature building / the ML model may consume.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    agent_version: str
    symbol: str
    timestamp: datetime
    features: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None


# --------------------------------------------------------------------------
# Model prediction / outcome
# --------------------------------------------------------------------------


class ModelPrediction(BaseModel):
    """Immutable ML model output. Never mutated after being recorded."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str = Field(default_factory=_new_id)
    model_version: str
    timestamp: datetime
    symbol: str
    predicted_excess_return_5d: float
    predicted_excess_return_20d: float
    probability_positive_5d: float = Field(ge=0.0, le=1.0)
    probability_positive_20d: float = Field(ge=0.0, le=1.0)
    predicted_volatility: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    feature_version: str


class Outcome(BaseModel):
    """The realised, labelled outcome of a previously recorded prediction."""

    model_config = ConfigDict(frozen=True)

    prediction_id: str
    realised_excess_return_5d: float | None = None
    realised_excess_return_20d: float | None = None
    realised_volatility: float | None = None
    completion_timestamp: datetime


# --------------------------------------------------------------------------
# Paper trading
# --------------------------------------------------------------------------


class PaperOrder(BaseModel):
    """A proposed simulated order. ``risk_approval_status`` starts PENDING
    and is only ever transitioned by the deterministic risk engine."""

    order_id: str = Field(default_factory=_new_id)
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)
    order_type: OrderType
    proposed_price: float = Field(gt=0)
    timestamp: datetime
    strategy_version: str
    risk_approval_status: RiskApprovalStatus = RiskApprovalStatus.PENDING
    risk_reason_codes: list[RiskReasonCode] = Field(default_factory=list)
    limit_price: float | None = None


class PaperFill(BaseModel):
    """Immutable simulated fill record."""

    model_config = ConfigDict(frozen=True)

    fill_id: str = Field(default_factory=_new_id)
    order_id: str
    fill_timestamp: datetime
    fill_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    slippage: float
    commission: float = Field(ge=0.0)
