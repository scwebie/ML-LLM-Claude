"""News / Event Intelligence Agent.

Consumes already-computed daily news sentiment / event-uncertainty values
(``data/news.py``, synthetic in v0.1) and produces bounded scores.

Hard safety boundary: this agent analyses ordinary financial/economic news
and scheduled corporate events ONLY. It must never be pointed at
prediction-market or betting-platform data sources -- there is no such
data source anywhere in this codebase.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agents.base import AGENT_VERSION, clip, maybe_llm_reasoning, safe_tanh
from core.schemas import AgentReport

AGENT_NAME = "event_intelligence_agent"


class EventIntelligenceOutput(BaseModel):
    event_sentiment: float = Field(ge=-1.0, le=1.0)
    event_uncertainty: float = Field(ge=0.0, le=1.0)
    earnings_event_score: float = Field(ge=-1.0, le=1.0)
    macro_event_score: float = Field(ge=-1.0, le=1.0)
    news_sentiment: float = Field(ge=-1.0, le=1.0)
    expected_catalyst_direction: float = Field(ge=-1.0, le=1.0, description="Signed magnitude; + = bullish catalyst expected")
    event_confidence: float = Field(ge=0.0, le=1.0)


def _score(news_row: dict, macro_surprise: float) -> EventIntelligenceOutput:
    news_sentiment = clip(news_row.get("news_sentiment", 0.0) or 0.0, -1.0, 1.0)
    event_uncertainty = clip(news_row.get("event_uncertainty", 0.0) or 0.0, 0.0, 1.0)
    is_earnings = bool(news_row.get("is_earnings_event", False))

    earnings_event_score = news_sentiment * (1.5 if is_earnings else 0.0)
    earnings_event_score = clip(earnings_event_score, -1.0, 1.0)

    macro_event_score = clip(safe_tanh(macro_surprise), -1.0, 1.0)

    event_sentiment = clip(0.7 * news_sentiment + 0.3 * macro_event_score, -1.0, 1.0)
    expected_catalyst_direction = clip(event_sentiment * (1.0 - 0.5 * event_uncertainty), -1.0, 1.0)
    event_confidence = clip(1.0 - event_uncertainty, 0.0, 1.0)

    return EventIntelligenceOutput(
        event_sentiment=event_sentiment,
        event_uncertainty=event_uncertainty,
        earnings_event_score=earnings_event_score,
        macro_event_score=macro_event_score,
        news_sentiment=news_sentiment,
        expected_catalyst_direction=expected_catalyst_direction,
        event_confidence=event_confidence,
    )


def analyze(
    symbol: str, as_of: datetime, news_row: dict, macro_surprise: float = 0.0, use_llm: bool = True
) -> tuple[EventIntelligenceOutput, AgentReport]:
    output = _score(news_row, macro_surprise)

    reasoning = None
    if use_llm:
        reasoning = maybe_llm_reasoning(
            system_prompt=(
                "You are a news/event-intelligence research assistant covering "
                "ordinary financial and economic news only (never prediction "
                "markets or betting platforms). Summarise the given scores in "
                "one short sentence without inventing new numbers."
            ),
            user_prompt=f"symbol={symbol} scores={output.model_dump()}",
        )

    report = AgentReport(
        agent=AGENT_NAME,
        agent_version=AGENT_VERSION,
        symbol=symbol,
        timestamp=as_of,
        features={f"event_{k}": float(v) for k, v in output.model_dump().items()},
        confidence=output.event_confidence,
        evidence_refs=["data/news.py:get_news"],
        reasoning_summary=reasoning,
    )
    return output, report
