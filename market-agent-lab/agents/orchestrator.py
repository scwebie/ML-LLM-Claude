"""Orchestrator Agent.

Coordinates the five research agents for one (symbol, as_of) point,
combines their outputs, computes the Phase-11 agent-disagreement metric,
and returns a single flat numeric feature dict ready for the Feature
Store. The orchestrator NEVER places or approves orders -- it only ever
emits features. Trading decisions are made downstream by the ML model,
the deterministic Portfolio Decision Engine, and the deterministic Risk
Engine, in that order.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime

import duckdb

from agents import event_intelligence, fundamental, historical, market_overview, technical
from core.schemas import AgentReport
from database import repository as repo
from features.historical import SimilarityResult


@dataclass
class OrchestratedFeatures:
    symbol: str
    timestamp: datetime
    features: dict[str, float]
    reports: list[AgentReport]


def _fundamental_composite(output: fundamental.FundamentalAgentOutput) -> float:
    return statistics.fmean(
        [
            output.growth_score,
            output.profitability_score,
            output.balance_sheet_score,
            output.valuation_score,
            output.earnings_quality_score,
            output.guidance_score,
        ]
    )


def _historical_directional(output: historical.HistoricalAgentOutput) -> float:
    import math

    if output.avg_return_5d is None:
        return 0.0
    return math.tanh(output.avg_return_5d * 20.0)


def run_research_agents(
    symbol: str,
    as_of: datetime,
    tech_row: dict,
    fund_row: dict,
    macro_features: dict,
    breadth_ratio: float,
    similarity: SimilarityResult,
    news_row: dict,
    macro_surprise: float = 0.0,
    use_llm: bool = False,
) -> OrchestratedFeatures:
    """Run all five research agents and combine their outputs.

    ``use_llm`` defaults to False here because this function is meant to be
    called for every (symbol, date) pair when building the historical
    Feature Store -- doing that against a live LLM would be slow, costly,
    and (crucially) non-deterministic, which would break reproducible ML
    training. Live/interactive callers (e.g. the API's single-prediction
    endpoint) may opt in with ``use_llm=True``.
    """
    tech_out, tech_report = technical.analyze(symbol, as_of, tech_row, use_llm=use_llm)
    fund_out, fund_report = fundamental.analyze(symbol, as_of, fund_row, use_llm=use_llm)
    macro_out, macro_report = market_overview.analyze(as_of, macro_features, breadth_ratio, use_llm=use_llm)
    hist_out, hist_report = historical.analyze(symbol, as_of, similarity, use_llm=use_llm)
    event_out, event_report = event_intelligence.analyze(symbol, as_of, news_row, macro_surprise, use_llm=use_llm)

    agent_scores = [
        tech_out.trend_score,
        _fundamental_composite(fund_out),
        macro_out.risk_appetite_score,
        _historical_directional(hist_out),
        event_out.event_sentiment,
    ]
    agent_disagreement = float(statistics.pstdev(agent_scores))
    mean_confidence = statistics.fmean(
        [tech_out.technical_confidence, fund_out.fundamental_confidence, macro_out.macro_confidence, hist_out.similarity_confidence, event_out.event_confidence]
    )
    disagreement_penalised_confidence = max(0.0, mean_confidence * (1.0 - agent_disagreement))

    features: dict[str, float] = {}
    for report in (tech_report, fund_report, macro_report, hist_report, event_report):
        features.update(report.features)
    features["agent_disagreement"] = agent_disagreement
    features["agent_mean_confidence"] = mean_confidence
    features["agent_composite_confidence"] = disagreement_penalised_confidence
    features["agent_composite_score"] = float(statistics.fmean(agent_scores))

    return OrchestratedFeatures(
        symbol=symbol,
        timestamp=as_of,
        features=features,
        reports=[tech_report, fund_report, macro_report, hist_report, event_report],
    )


def persist_reports(con: duckdb.DuckDBPyConnection, reports: list[AgentReport]) -> None:
    repo.insert_agent_reports(con, reports)
