"""News/event features (Phase 17): deterministic counts + a deterministic
heuristic structured classifier.

Both halves respect the point-in-time rule -- callers must already have
filtered articles to ``published_at <= as_of`` (see
``database.repository_v2.get_news_asof``) before calling anything here;
this module performs no filtering of its own.

On the LLM-assisted classification: this codebase's OpenAI integration
(``agents/base.py::maybe_llm_reasoning``) is deliberately restricted to
producing an optional natural-language gloss on top of already-computed
numbers -- never the numbers themselves, so training data stays
reproducible without a live network/API key. The same discipline applies
here: ``classify_article`` always produces its numeric/categorical fields
deterministically from explicit, configured rules (never an LLM), and
records ``llm_model="heuristic_v1"`` so it is never confused with a real
model call in the provenance trail.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.schemas_v2 import NEWS_TIER_WEIGHTS, NewsAgentFeatures, NewsArticle, NewsTier

LOOKBACK_WINDOWS_HOURS = {"1h": 1, "6h": 6, "24h": 24, "3d": 72, "7d": 168}

_CATEGORY_SENTIMENT: dict[str, float] = {
    "earnings": 0.10, "guidance": 0.05, "product": 0.10, "M&A": 0.0,
    "litigation": -0.30, "regulation": -0.15, "credit": -0.10, "management": -0.05, "other": 0.0,
}
_CATEGORY_IMPACT: dict[str, float] = {
    "earnings": 0.6, "M&A": 0.6, "regulation": 0.6, "litigation": 0.6,
    "guidance": 0.4, "credit": 0.4, "management": 0.3, "product": 0.3, "other": 0.15,
}
_CATEGORY_HORIZON: dict[str, str] = {
    "earnings": "5-20d", "guidance": "5-20d", "M&A": "long-term", "litigation": "long-term",
    "regulation": "long-term", "management": "1-5d", "product": "1-5d", "credit": "1-5d", "other": "intraday",
}

PROMPT_VERSION = "heuristic_v1"


def compute_deterministic_news_counts(articles: pd.DataFrame, as_of: datetime) -> dict[str, float]:
    """``articles``: already point-in-time-filtered rows with
    ``published_at`` and ``tier`` columns."""
    counts: dict[str, float] = {}
    if articles.empty:
        for key in LOOKBACK_WINDOWS_HOURS:
            counts[f"n_articles_{key}"] = 0.0
        counts["source_weighted_count_7d"] = 0.0
        counts["hours_since_last_article"] = float("nan")
        return counts

    published = pd.to_datetime(articles["published_at"])
    hours_ago = (pd.Timestamp(as_of) - published).dt.total_seconds() / 3600.0

    for key, window_hours in LOOKBACK_WINDOWS_HOURS.items():
        counts[f"n_articles_{key}"] = float((hours_ago <= window_hours).sum())

    weights = articles["tier"].map(lambda t: NEWS_TIER_WEIGHTS.get(NewsTier(t), NEWS_TIER_WEIGHTS[NewsTier.UNKNOWN]))
    within_7d = hours_ago <= LOOKBACK_WINDOWS_HOURS["7d"]
    counts["source_weighted_count_7d"] = float(weights[within_7d].sum())
    counts["hours_since_last_article"] = float(hours_ago.min())
    return counts


def classify_article(article: NewsArticle, prior_headlines_7d: set[str] | None = None) -> NewsAgentFeatures:
    """Deterministic heuristic structured classification -- see module
    docstring for why this is not an LLM call by default."""
    category = article.event_category or "other"
    sentiment = _CATEGORY_SENTIMENT.get(category, 0.0)
    impact = _CATEGORY_IMPACT.get(category, 0.15)
    horizon = _CATEGORY_HORIZON.get(category, "intraday")

    uncertainty = 0.2 if article.tier == NewsTier.TIER_1_OFFICIAL else 0.4
    if article.timestamp_uncertain:
        uncertainty = min(1.0, uncertainty + 0.2)

    novelty = 1.0
    if prior_headlines_7d and article.headline in prior_headlines_7d:
        novelty = 0.1  # a repeated/syndicated headline is not novel

    return NewsAgentFeatures(
        article_id=article.article_id, symbol=article.symbols[0] if article.symbols else "",
        sentiment=sentiment, impact_magnitude=impact, uncertainty=uncertainty, novelty=novelty,
        event_category=category, expected_horizon=horizon,
        llm_model="heuristic_v1", prompt_version=PROMPT_VERSION, generated_at=article.retrieved_at,
    )
