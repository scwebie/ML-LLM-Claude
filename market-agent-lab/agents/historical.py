"""Historical Research Agent.

Wraps the deterministic similarity engine in ``features/historical.py``
(standardised Euclidean distance nearest-neighbour search) in an
``AgentReport``. The agent performs no similarity math itself -- it only
formats the already-computed :class:`~features.historical.SimilarityResult`.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from agents.base import AGENT_VERSION, clip, maybe_llm_reasoning
from core.schemas import AgentReport
from features.historical import SimilarityResult

AGENT_NAME = "historical_research_agent"


class HistoricalAgentOutput(BaseModel):
    num_analogues: int = Field(ge=0)
    avg_return_5d: float | None = None
    median_return_5d: float | None = None
    prob_positive_5d: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_return_20d: float | None = None
    median_return_20d: float | None = None
    prob_positive_20d: float | None = Field(default=None, ge=0.0, le=1.0)
    p10_return_20d: float | None = None
    p90_return_20d: float | None = None
    similarity_confidence: float = Field(ge=0.0, le=1.0)


def analyze(
    symbol: str, as_of: datetime, similarity: SimilarityResult, use_llm: bool = True
) -> tuple[HistoricalAgentOutput, AgentReport]:
    output = HistoricalAgentOutput(**similarity.__dict__)

    reasoning = None
    if use_llm:
        reasoning = maybe_llm_reasoning(
            system_prompt=(
                "You are a research assistant summarising a nearest-neighbour "
                "historical-analogue study in one short sentence. Never invent "
                "numbers beyond what is given."
            ),
            user_prompt=f"symbol={symbol} result={output.model_dump()}",
        )

    features = {f"hist_{k}": (v if v is not None else float("nan")) for k, v in output.model_dump().items()}
    features = {k: float(v) for k, v in features.items()}

    report = AgentReport(
        agent=AGENT_NAME,
        agent_version=AGENT_VERSION,
        symbol=symbol,
        timestamp=as_of,
        features=features,
        confidence=clip(output.similarity_confidence, 0.0, 1.0),
        evidence_refs=["features/historical.py:find_historical_analogues"],
        reasoning_summary=reasoning,
    )
    return output, report
