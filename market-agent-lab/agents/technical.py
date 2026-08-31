"""Technical Analysis Agent.

Consumes already-computed technical indicators (``features/technical.py``)
and produces bounded, documented-range scores. It performs zero indicator
math of its own -- only bounded, deterministic combinations of the inputs
it is handed.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agents.base import AGENT_VERSION, clip, maybe_llm_reasoning, safe_tanh
from core.schemas import AgentReport

AGENT_NAME = "technical_agent"


class TechnicalAgentOutput(BaseModel):
    """All scores documented range: see field descriptions."""

    trend_score: float = Field(ge=-1.0, le=1.0, description="Direction/strength of price trend")
    momentum_score: float = Field(ge=-1.0, le=1.0, description="Short/medium-term momentum")
    volatility_risk: float = Field(ge=0.0, le=1.0, description="0=calm, 1=extreme")
    volume_confirmation: float = Field(ge=-1.0, le=1.0, description="Does volume confirm the move")
    technical_confidence: float = Field(ge=0.0, le=1.0)


def _score(tech_row: dict) -> TechnicalAgentOutput:
    dist_20 = tech_row.get("dist_sma_20") or 0.0
    dist_50 = tech_row.get("dist_sma_50") or 0.0
    dist_200 = tech_row.get("dist_sma_200") or 0.0
    trend_raw = 0.5 * (dist_20 or 0.0) + 0.3 * (dist_50 or 0.0) + 0.2 * (dist_200 or 0.0)
    trend_score = safe_tanh(trend_raw * 8.0)

    rsi = tech_row.get("rsi_14")
    rsi_component = ((rsi - 50.0) / 50.0) if rsi == rsi and rsi is not None else 0.0
    macd = tech_row.get("macd") or 0.0
    macd_signal = tech_row.get("macd_signal") or 0.0
    macd_hist = macd - macd_signal
    ret_20 = tech_row.get("return_20d") or 0.0
    momentum_raw = 0.4 * rsi_component + 0.3 * safe_tanh(macd_hist * 4.0) + 0.3 * safe_tanh(ret_20 * 4.0)
    momentum_score = clip(momentum_raw, -1.0, 1.0)

    vol_20 = tech_row.get("realised_vol_20d")
    volatility_risk = clip((vol_20 or 0.0) / 0.60, 0.0, 1.0)

    rel_vol = tech_row.get("relative_volume_20d")
    rel_vol_component = safe_tanh(((rel_vol or 1.0) - 1.0) * 2.0)
    direction_sign = 1.0 if trend_score >= 0 else -1.0
    volume_confirmation = clip(rel_vol_component * direction_sign, -1.0, 1.0)

    completeness = sum(
        1
        for k in ("dist_sma_20", "dist_sma_50", "rsi_14", "macd", "realised_vol_20d", "relative_volume_20d")
        if tech_row.get(k) == tech_row.get(k) and tech_row.get(k) is not None
    ) / 6.0
    agreement = 1.0 - abs(trend_score - momentum_score) / 2.0
    technical_confidence = clip(0.3 + 0.4 * completeness + 0.3 * agreement, 0.0, 1.0)

    return TechnicalAgentOutput(
        trend_score=trend_score,
        momentum_score=momentum_score,
        volatility_risk=volatility_risk,
        volume_confirmation=volume_confirmation,
        technical_confidence=technical_confidence,
    )


def analyze(symbol: str, as_of: datetime, tech_row: dict, use_llm: bool = True) -> tuple[TechnicalAgentOutput, AgentReport]:
    output = _score(tech_row)

    reasoning = None
    if use_llm:
        reasoning = maybe_llm_reasoning(
            system_prompt=(
                "You are a technical-analysis research assistant. You are given "
                "already-computed scores and must only summarise them in one "
                "short sentence. Never invent new numbers."
            ),
            user_prompt=f"symbol={symbol} scores={output.model_dump()}",
        )

    report = AgentReport(
        agent=AGENT_NAME,
        agent_version=AGENT_VERSION,
        symbol=symbol,
        timestamp=as_of,
        features={
            "technical_trend_score": output.trend_score,
            "technical_momentum_score": output.momentum_score,
            "technical_volatility_risk": output.volatility_risk,
            "technical_volume_confirmation": output.volume_confirmation,
            "technical_confidence": output.technical_confidence,
        },
        confidence=output.technical_confidence,
        evidence_refs=["features/technical.py:compute_technical_features"],
        reasoning_summary=reasoning,
    )
    return output, report
