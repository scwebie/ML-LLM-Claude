"""Market Overview Agent.

Consumes already-computed macro z-scores (``features/macro.py``) plus a
cross-sectional market-breadth ratio and classifies the overall regime.
Regime classification is done with fixed, documented thresholds on the
macro z-scores -- again, no LLM involvement in the numeric output.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agents.base import AGENT_VERSION, clip, maybe_llm_reasoning, safe_tanh
from core.schemas import (
    AgentReport,
    GrowthRegime,
    InflationRegime,
    MonetaryPolicyRegime,
    VolatilityRegime,
)

AGENT_NAME = "market_overview_agent"

# Ordinal encodings so regime labels can also travel through AgentReport's
# numeric `features` dict for ML consumption.
_VOL_REGIME_CODE = {VolatilityRegime.LOW: 0, VolatilityRegime.NORMAL: 1, VolatilityRegime.ELEVATED: 2, VolatilityRegime.CRISIS: 3}
_GROWTH_REGIME_CODE = {GrowthRegime.CONTRACTION: 0, GrowthRegime.SLOWDOWN: 1, GrowthRegime.EXPANSION: 2, GrowthRegime.OVERHEATING: 3}
_INFLATION_REGIME_CODE = {InflationRegime.DEFLATIONARY: 0, InflationRegime.LOW: 1, InflationRegime.MODERATE: 2, InflationRegime.HIGH: 3}
_POLICY_REGIME_CODE = {MonetaryPolicyRegime.EASING: 0, MonetaryPolicyRegime.NEUTRAL: 1, MonetaryPolicyRegime.TIGHTENING: 2}


class MarketOverviewOutput(BaseModel):
    risk_appetite_score: float = Field(ge=-1.0, le=1.0)
    liquidity_score: float = Field(ge=-1.0, le=1.0)
    volatility_regime: VolatilityRegime
    growth_regime: GrowthRegime
    inflation_regime: InflationRegime
    monetary_policy_regime: MonetaryPolicyRegime
    market_breadth_score: float = Field(ge=-1.0, le=1.0)
    macro_confidence: float = Field(ge=0.0, le=1.0)


def _vol_regime(vol_z: float) -> VolatilityRegime:
    if vol_z >= 2.0:
        return VolatilityRegime.CRISIS
    if vol_z >= 1.0:
        return VolatilityRegime.ELEVATED
    if vol_z <= -1.0:
        return VolatilityRegime.LOW
    return VolatilityRegime.NORMAL


def _growth_regime(growth_z: float) -> GrowthRegime:
    if growth_z >= 1.5:
        return GrowthRegime.OVERHEATING
    if growth_z >= 0.0:
        return GrowthRegime.EXPANSION
    if growth_z >= -1.5:
        return GrowthRegime.SLOWDOWN
    return GrowthRegime.CONTRACTION


def _inflation_regime(inflation_z: float) -> InflationRegime:
    if inflation_z >= 1.5:
        return InflationRegime.HIGH
    if inflation_z >= 0.0:
        return InflationRegime.MODERATE
    if inflation_z >= -1.5:
        return InflationRegime.LOW
    return InflationRegime.DEFLATIONARY


def _policy_regime(rates_z: float) -> MonetaryPolicyRegime:
    if rates_z >= 0.5:
        return MonetaryPolicyRegime.TIGHTENING
    if rates_z <= -0.5:
        return MonetaryPolicyRegime.EASING
    return MonetaryPolicyRegime.NEUTRAL


def _score(macro_features: dict, breadth_ratio: float) -> MarketOverviewOutput:
    vol_z = macro_features.get("SYN_VOL_INDEX_zscore", 0.0) or 0.0
    growth_z = macro_features.get("SYN_GROWTH_INDEX_zscore", 0.0) or 0.0
    inflation_z = macro_features.get("SYN_INFLATION_zscore", 0.0) or 0.0
    rates_z = macro_features.get("SYN_RATES_zscore", 0.0) or 0.0

    risk_appetite_score = clip(safe_tanh(growth_z * 0.5 - vol_z * 0.5), -1.0, 1.0)
    liquidity_score = clip(safe_tanh(-rates_z * 0.6 - vol_z * 0.2), -1.0, 1.0)
    market_breadth_score = clip((breadth_ratio - 0.5) * 2.0, -1.0, 1.0)

    n_present = sum(1 for k in ("SYN_VOL_INDEX_zscore", "SYN_GROWTH_INDEX_zscore", "SYN_INFLATION_zscore", "SYN_RATES_zscore") if k in macro_features)
    macro_confidence = clip(0.25 + 0.75 * (n_present / 4.0), 0.0, 1.0)

    return MarketOverviewOutput(
        risk_appetite_score=risk_appetite_score,
        liquidity_score=liquidity_score,
        volatility_regime=_vol_regime(vol_z),
        growth_regime=_growth_regime(growth_z),
        inflation_regime=_inflation_regime(inflation_z),
        monetary_policy_regime=_policy_regime(rates_z),
        market_breadth_score=market_breadth_score,
        macro_confidence=macro_confidence,
    )


def analyze(
    as_of: datetime, macro_features: dict, breadth_ratio: float, use_llm: bool = True
) -> tuple[MarketOverviewOutput, AgentReport]:
    output = _score(macro_features, breadth_ratio)

    reasoning = None
    if use_llm:
        reasoning = maybe_llm_reasoning(
            system_prompt=(
                "You are a macro market-overview research assistant. You are "
                "given already-computed regime classifications and must only "
                "summarise them in one short sentence."
            ),
            user_prompt=f"scores={output.model_dump()}",
        )

    report = AgentReport(
        agent=AGENT_NAME,
        agent_version=AGENT_VERSION,
        symbol="_MARKET_",
        timestamp=as_of,
        features={
            "macro_risk_appetite_score": output.risk_appetite_score,
            "macro_liquidity_score": output.liquidity_score,
            "macro_volatility_regime_code": float(_VOL_REGIME_CODE[output.volatility_regime]),
            "macro_growth_regime_code": float(_GROWTH_REGIME_CODE[output.growth_regime]),
            "macro_inflation_regime_code": float(_INFLATION_REGIME_CODE[output.inflation_regime]),
            "macro_policy_regime_code": float(_POLICY_REGIME_CODE[output.monetary_policy_regime]),
            "macro_market_breadth_score": output.market_breadth_score,
            "macro_confidence": output.macro_confidence,
        },
        confidence=output.macro_confidence,
        evidence_refs=["features/macro.py:compute_macro_features"],
        reasoning_summary=reasoning,
    )
    return output, report
