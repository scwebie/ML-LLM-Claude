"""Fundamental Analysis Agent.

Consumes already-computed fundamental features (``features/fundamental.py``)
-- cross-sectional z-scores and ratios -- and produces bounded scores. It
performs no fundamental-ratio math of its own.

Note on ``guidance_score``: Version 0.1's synthetic dataset has no analyst
guidance / forward-estimate revision series, so this score is a documented
proxy built from EPS-growth consistency rather than real guidance data.
This limitation is called out in ``docs/model_design.md``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agents.base import AGENT_VERSION, clip, maybe_llm_reasoning, safe_tanh
from core.schemas import AgentReport

AGENT_NAME = "fundamental_agent"


class FundamentalAgentOutput(BaseModel):
    growth_score: float = Field(ge=-1.0, le=1.0)
    profitability_score: float = Field(ge=-1.0, le=1.0)
    balance_sheet_score: float = Field(ge=-1.0, le=1.0)
    valuation_score: float = Field(ge=-1.0, le=1.0)
    earnings_quality_score: float = Field(ge=-1.0, le=1.0)
    guidance_score: float = Field(ge=-1.0, le=1.0, description="Proxy: EPS growth consistency, no real guidance data in v0.1")
    fundamental_confidence: float = Field(ge=0.0, le=1.0)


def _score(fund_row: dict) -> FundamentalAgentOutput:
    growth_score = safe_tanh((fund_row.get("growth_zscore") or 0.0) * 0.7)
    profitability_score = safe_tanh((fund_row.get("profitability_zscore") or 0.0) * 0.7)
    valuation_score = safe_tanh((fund_row.get("valuation_zscore") or 0.0) * 0.7)

    debt_to_cash = fund_row.get("debt_to_cash")
    balance_sheet_score = safe_tanh(-((debt_to_cash or 1.0) - 1.0))

    fcf_margin = fund_row.get("fcf_margin")
    operating_margin = fund_row.get("operating_margin")
    if fcf_margin is not None and operating_margin is not None and fcf_margin == fcf_margin and operating_margin == operating_margin:
        quality_gap = abs(operating_margin - fcf_margin)
        earnings_quality_score = clip(1.0 - quality_gap * 4.0, -1.0, 1.0)
    else:
        earnings_quality_score = 0.0

    eps_growth = fund_row.get("eps_growth")
    revenue_growth = fund_row.get("revenue_growth")
    if eps_growth is not None and revenue_growth is not None and eps_growth == eps_growth and revenue_growth == revenue_growth:
        consistency_gap = abs(eps_growth - revenue_growth)
        guidance_score = clip(safe_tanh(min(eps_growth, revenue_growth)) - clip(consistency_gap, 0.0, 1.0), -1.0, 1.0)
    else:
        guidance_score = 0.0

    present_fields = [
        fund_row.get(k)
        for k in ("revenue_growth", "eps_growth", "gross_margin", "operating_margin", "fcf_margin", "roic")
    ]
    completeness = sum(1 for v in present_fields if v is not None and v == v) / len(present_fields)
    fundamental_confidence = clip(0.3 + 0.7 * completeness, 0.0, 1.0)

    return FundamentalAgentOutput(
        growth_score=growth_score,
        profitability_score=profitability_score,
        balance_sheet_score=balance_sheet_score,
        valuation_score=valuation_score,
        earnings_quality_score=earnings_quality_score,
        guidance_score=guidance_score,
        fundamental_confidence=fundamental_confidence,
    )


def analyze(symbol: str, as_of: datetime, fund_row: dict, use_llm: bool = True) -> tuple[FundamentalAgentOutput, AgentReport]:
    output = _score(fund_row)

    reasoning = None
    if use_llm:
        reasoning = maybe_llm_reasoning(
            system_prompt=(
                "You are a fundamental-analysis research assistant. You are given "
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
            "fundamental_growth_score": output.growth_score,
            "fundamental_profitability_score": output.profitability_score,
            "fundamental_balance_sheet_score": output.balance_sheet_score,
            "fundamental_valuation_score": output.valuation_score,
            "fundamental_earnings_quality_score": output.earnings_quality_score,
            "fundamental_guidance_score": output.guidance_score,
            "fundamental_confidence": output.fundamental_confidence,
        },
        confidence=output.fundamental_confidence,
        evidence_refs=["features/fundamental.py:compute_fundamental_features"],
        reasoning_summary=reasoning,
    )
    return output, report
