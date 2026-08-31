"""Tests for research agents: output ranges, and the Phase-11 agent-disagreement metric."""

from __future__ import annotations

from datetime import datetime

import pytest

from agents import event_intelligence, fundamental, market_overview, technical
from agents.orchestrator import run_research_agents
from core.schemas import GrowthRegime, InflationRegime, MonetaryPolicyRegime, VolatilityRegime
from features.historical import SimilarityResult


def test_technical_agent_scores_within_documented_ranges():
    tech_row = {
        "dist_sma_20": 0.05, "dist_sma_50": 0.03, "dist_sma_200": 0.01,
        "rsi_14": 70.0, "macd": 0.5, "macd_signal": 0.3, "realised_vol_20d": 0.25,
        "relative_volume_20d": 1.5, "return_20d": 0.08,
    }
    output, report = technical.analyze("SYN_X", datetime(2023, 1, 1), tech_row, use_llm=False)
    assert -1.0 <= output.trend_score <= 1.0
    assert -1.0 <= output.momentum_score <= 1.0
    assert 0.0 <= output.volatility_risk <= 1.0
    assert -1.0 <= output.volume_confirmation <= 1.0
    assert 0.0 <= output.technical_confidence <= 1.0
    assert report.confidence == output.technical_confidence
    assert report.reasoning_summary is None  # no API key configured in test env


def test_technical_agent_uptrend_produces_positive_trend_score():
    up_row = {"dist_sma_20": 0.10, "dist_sma_50": 0.08, "dist_sma_200": 0.05, "rsi_14": 65, "macd": 1, "macd_signal": 0.5, "realised_vol_20d": 0.2, "relative_volume_20d": 1.2, "return_20d": 0.1}
    down_row = {"dist_sma_20": -0.10, "dist_sma_50": -0.08, "dist_sma_200": -0.05, "rsi_14": 35, "macd": -1, "macd_signal": -0.5, "realised_vol_20d": 0.2, "relative_volume_20d": 1.2, "return_20d": -0.1}
    up_output, _ = technical.analyze("SYN_X", datetime(2023, 1, 1), up_row, use_llm=False)
    down_output, _ = technical.analyze("SYN_X", datetime(2023, 1, 1), down_row, use_llm=False)
    assert up_output.trend_score > 0
    assert down_output.trend_score < 0
    assert up_output.trend_score > down_output.trend_score


def test_fundamental_agent_scores_within_documented_ranges():
    fund_row = {
        "growth_zscore": 1.0, "profitability_zscore": 0.5, "valuation_zscore": -0.5,
        "debt_to_cash": 0.8, "fcf_margin": 0.15, "operating_margin": 0.18,
        "eps_growth": 0.1, "revenue_growth": 0.08,
    }
    output, report = fundamental.analyze("SYN_X", datetime(2023, 1, 1), fund_row, use_llm=False)
    for value in (output.growth_score, output.profitability_score, output.balance_sheet_score, output.valuation_score, output.earnings_quality_score, output.guidance_score):
        assert -1.0 <= value <= 1.0
    assert 0.0 <= output.fundamental_confidence <= 1.0


def test_market_overview_agent_regime_classification():
    macro_features = {
        "SYN_VOL_INDEX_zscore": 2.5, "SYN_GROWTH_INDEX_zscore": -2.0,
        "SYN_INFLATION_zscore": 2.0, "SYN_RATES_zscore": 1.0,
    }
    output, report = market_overview.analyze(datetime(2023, 1, 1), macro_features, breadth_ratio=0.2, use_llm=False)
    assert output.volatility_regime == VolatilityRegime.CRISIS
    assert output.growth_regime == GrowthRegime.CONTRACTION
    assert output.inflation_regime == InflationRegime.HIGH
    assert output.monetary_policy_regime == MonetaryPolicyRegime.TIGHTENING
    assert -1.0 <= output.market_breadth_score <= 1.0


def test_event_intelligence_agent_scores_within_documented_ranges():
    news_row = {"news_sentiment": 0.6, "event_uncertainty": 0.3, "is_earnings_event": True}
    output, report = event_intelligence.analyze("SYN_X", datetime(2023, 1, 1), news_row, macro_surprise=0.5, use_llm=False)
    for value in (output.event_sentiment, output.earnings_event_score, output.macro_event_score, output.news_sentiment, output.expected_catalyst_direction):
        assert -1.0 <= value <= 1.0
    assert 0.0 <= output.event_uncertainty <= 1.0
    assert 0.0 <= output.event_confidence <= 1.0


def test_orchestrator_agent_disagreement_is_zero_when_agents_agree():
    """If every agent's directional score is identical, disagreement must be exactly 0."""
    # Craft inputs that push every agent to a neutral/zero directional score.
    tech_row = {"dist_sma_20": 0.0, "dist_sma_50": 0.0, "dist_sma_200": 0.0, "rsi_14": 50.0, "macd": 0.0, "macd_signal": 0.0, "realised_vol_20d": 0.2, "relative_volume_20d": 1.0, "return_20d": 0.0}
    # fcf_margin is deliberately offset from operating_margin by exactly 0.25
    # so earnings_quality_score (1 - gap*4) comes out to 0, keeping every
    # fundamental sub-score -- and hence the composite -- at exactly 0.
    fund_row = {"growth_zscore": 0.0, "profitability_zscore": 0.0, "valuation_zscore": 0.0, "debt_to_cash": 1.0, "fcf_margin": -0.15, "operating_margin": 0.1, "eps_growth": 0.0, "revenue_growth": 0.0}
    macro_features = {"SYN_VOL_INDEX_zscore": 0.0, "SYN_GROWTH_INDEX_zscore": 0.0, "SYN_INFLATION_zscore": 0.0, "SYN_RATES_zscore": 0.0}
    similarity = SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0)
    news_row = {"news_sentiment": 0.0, "event_uncertainty": 0.5, "is_earnings_event": False}

    result = run_research_agents(
        symbol="SYN_X", as_of=datetime(2023, 1, 1), tech_row=tech_row, fund_row=fund_row,
        macro_features=macro_features, breadth_ratio=0.5, similarity=similarity, news_row=news_row,
        macro_surprise=0.0, use_llm=False,
    )
    assert result.features["agent_disagreement"] == pytest.approx(0.0, abs=1e-9)


def test_orchestrator_agent_disagreement_positive_when_agents_conflict():
    tech_row = {"dist_sma_20": 0.15, "dist_sma_50": 0.12, "dist_sma_200": 0.10, "rsi_14": 80.0, "macd": 2.0, "macd_signal": 0.5, "realised_vol_20d": 0.15, "relative_volume_20d": 2.0, "return_20d": 0.2}
    fund_row = {"growth_zscore": -2.0, "profitability_zscore": -2.0, "valuation_zscore": -2.0, "debt_to_cash": 3.0, "fcf_margin": -0.1, "operating_margin": -0.2, "eps_growth": -0.3, "revenue_growth": -0.2}
    macro_features = {"SYN_VOL_INDEX_zscore": 0.0, "SYN_GROWTH_INDEX_zscore": 0.0, "SYN_INFLATION_zscore": 0.0, "SYN_RATES_zscore": 0.0}
    similarity = SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0)
    news_row = {"news_sentiment": 0.0, "event_uncertainty": 0.5, "is_earnings_event": False}

    result = run_research_agents(
        symbol="SYN_X", as_of=datetime(2023, 1, 1), tech_row=tech_row, fund_row=fund_row,
        macro_features=macro_features, breadth_ratio=0.5, similarity=similarity, news_row=news_row,
        macro_surprise=0.0, use_llm=False,
    )
    assert result.features["agent_disagreement"] > 0.0
    # Five distinct AgentReports (one per research agent) must be returned.
    assert len(result.reports) == 5
    assert {r.agent for r in result.reports} == {
        "technical_agent", "fundamental_agent", "market_overview_agent",
        "historical_research_agent", "event_intelligence_agent",
    }


def test_orchestrator_disagreement_penalises_composite_confidence():
    """Higher disagreement must never *increase* composite confidence relative
    to the plain mean of individual agent confidences."""
    tech_row = {"dist_sma_20": 0.15, "dist_sma_50": 0.12, "dist_sma_200": 0.10, "rsi_14": 80.0, "macd": 2.0, "macd_signal": 0.5, "realised_vol_20d": 0.15, "relative_volume_20d": 2.0, "return_20d": 0.2}
    fund_row = {"growth_zscore": -2.0, "profitability_zscore": -2.0, "valuation_zscore": -2.0, "debt_to_cash": 3.0, "fcf_margin": -0.1, "operating_margin": -0.2, "eps_growth": -0.3, "revenue_growth": -0.2}
    macro_features = {"SYN_VOL_INDEX_zscore": 0.0, "SYN_GROWTH_INDEX_zscore": 0.0, "SYN_INFLATION_zscore": 0.0, "SYN_RATES_zscore": 0.0}
    similarity = SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0)
    news_row = {"news_sentiment": 0.0, "event_uncertainty": 0.5, "is_earnings_event": False}
    result = run_research_agents(
        symbol="SYN_X", as_of=datetime(2023, 1, 1), tech_row=tech_row, fund_row=fund_row,
        macro_features=macro_features, breadth_ratio=0.5, similarity=similarity, news_row=news_row,
        macro_surprise=0.0, use_llm=False,
    )
    assert result.features["agent_composite_confidence"] <= result.features["agent_mean_confidence"] + 1e-9
