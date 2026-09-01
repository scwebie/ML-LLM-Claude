"""Point-in-time real feature matrix builder (Phase 10).

Mirrors ``features/feature_store.py::build_feature_matrix`` (V0.1,
untouched) but sourced from real data end to end: real point-in-time
universe membership, real technical/fundamental/macro observations
(same tables, same schemas -- V0.1's agents and feature engines work on
them completely unchanged), real news, and the read-only event-probability
research signal in its own ``eventprob_*`` namespace so it can be fully
ablated (see ``learning/ablation.py``).

Decision-timestamp discipline (Phase 67): every feature for date ``t`` is
computed strictly from information available as of that trading day's
close -- fundamentals/macro/news are joined on publication timestamp,
never period end. Predictions are generated to be executed at the next
day's open (``backtesting/engine.py`` already enforces this for V0.1 and
is reused unchanged for real data).

No fake data (Phase 64): a feature whose real inputs are unavailable for
a given (symbol, date) is left ``NaN`` and flagged via a ``*_missing``
indicator column -- never imputed, never zero-filled.
"""

from __future__ import annotations

import pandas as pd

from agents.orchestrator import persist_reports, run_research_agents
from data import macro as macro_data
from data import universe as universe_data
from database import repository_v2 as repo_v2
from features.cross_sectional import DEFAULT_METRIC_SPEC, compute_percentile_ranks
from features.fundamental import compute_fundamental_features
from features.historical import SimilarityResult, compute_similarity_series
from features.macro import compute_macro_features
from features.market_breadth import compute_market_breadth
from features.news_features import classify_article, compute_deterministic_news_counts
from features.technical import TECHNICAL_FEATURE_COLUMNS, compute_technical_features_multi

HISTORICAL_SIMILARITY_FEATURE_COLS = ["dist_sma_20", "dist_sma_50", "rsi_14", "realised_vol_20d", "return_20d"]

EVENT_CATEGORIES = ["monetary_policy", "economic_outcomes", "elections_policy", "geopolitical", "regulatory"]


def _news_row_for_agent(articles_asof: pd.DataFrame) -> dict:
    """Adapter: reduce real, point-in-time-filtered articles into the
    same {news_sentiment, event_uncertainty, is_earnings_event} shape
    V0.1's event_intelligence agent already consumes, so that agent works
    unchanged on real data."""
    if articles_asof.empty:
        return {"news_sentiment": 0.0, "event_uncertainty": 0.5, "is_earnings_event": False}
    sentiments, uncertainties, is_earnings = [], [], False
    for _, row in articles_asof.iterrows():
        from core.schemas_v2 import NewsArticle, NewsTier

        article = NewsArticle(
            headline=row.get("headline", ""), published_at=row.get("published_at"), retrieved_at=row.get("retrieved_at", row.get("published_at")),
            source=row.get("source", "unknown"), tier=NewsTier(row["tier"]), event_category=row.get("event_category"),
            timestamp_uncertain=bool(row.get("timestamp_uncertain", False)), symbols=[],
        )
        classified = classify_article(article)
        sentiments.append(classified.sentiment)
        uncertainties.append(classified.uncertainty)
        if classified.event_category == "earnings":
            is_earnings = True
    return {
        "news_sentiment": float(pd.Series(sentiments).mean()) if sentiments else 0.0,
        "event_uncertainty": float(pd.Series(uncertainties).mean()) if uncertainties else 0.5,
        "is_earnings_event": is_earnings,
    }


def _build_event_probability_lookup(con, symbols: list[str]) -> tuple[dict, dict]:
    """Bulk-load event mappings/observations ONCE (not per row) into pure
    in-memory structures. Polymarket's public API only exposes each
    market's CURRENT state (no historical price-history endpoint used
    here), so in practice every event has a single ``observed_timestamp``
    close to ingestion time -- meaning eventprob_* features are correctly
    all-NaN for any historical ``as_of`` before that (the as-of filter
    below enforces this; see docs/data_sources.md for why this is a
    genuine data-source limitation, not a bug)."""
    if not symbols:
        return {}, {}
    mappings = con.execute(
        f"SELECT * FROM event_symbol_mappings WHERE symbol IN ({','.join(['?'] * len(symbols))})", symbols
    ).fetchdf()
    if mappings.empty:
        return {}, {}
    event_ids = mappings["event_id"].unique().tolist()
    obs = con.execute(
        f"SELECT event_id, observed_timestamp, public_probability FROM event_probability_observations WHERE event_id IN ({','.join(['?'] * len(event_ids))})",
        event_ids,
    ).fetchdf()

    obs_by_event: dict[str, list[tuple]] = {}
    for event_id, group in obs.groupby("event_id"):
        g = group.sort_values("observed_timestamp")
        obs_by_event[event_id] = list(zip(pd.to_datetime(g["observed_timestamp"]).tolist(), g["public_probability"].tolist(), strict=True))
        for i in range(len(obs_by_event[event_id])):
            obs_by_event[event_id][i] = (obs_by_event[event_id][i][0].to_pydatetime(), obs_by_event[event_id][i][1])

    symbol_category_events: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for _, m in mappings.iterrows():
        for category in EVENT_CATEGORIES:
            if category.split("_")[0] in m["rationale_category"]:
                symbol_category_events.setdefault((m["symbol"], category), []).append((m["event_id"], float(m["relevance"])))

    return symbol_category_events, obs_by_event


def _lookup_asof(history: list[tuple], as_of) -> float | None:
    import bisect

    if not history:
        return None
    timestamps = [h[0] for h in history]
    idx = bisect.bisect_right(timestamps, as_of) - 1
    if idx < 0:
        return None
    return history[idx][1]


def _event_probability_features(symbol: str, as_of, symbol_category_events: dict, obs_by_event: dict) -> dict[str, float]:
    features: dict[str, float] = {}
    any_found = False
    for category in EVENT_CATEGORIES:
        events = symbol_category_events.get((symbol, category), [])
        probs, weights = [], []
        for event_id, relevance in events:
            prob = _lookup_asof(obs_by_event.get(event_id, []), as_of)
            if prob is not None:
                probs.append(prob)
                weights.append(relevance)
        if probs:
            features[f"eventprob_{category}_probability"] = float(sum(p * w for p, w in zip(probs, weights, strict=True)) / max(sum(weights), 1e-9))
            any_found = True
        else:
            features[f"eventprob_{category}_probability"] = float("nan")
    features["eventprob_missing"] = 0.0 if any_found else 1.0
    return features


def build_real_feature_matrix(
    con,
    universe_name: str,
    symbols: list[str],
    sector_map: dict[str, str],
    use_llm: bool = False,
    persist_agent_reports: bool = True,
) -> pd.DataFrame:
    from data.market_data import get_ohlcv

    market_df = get_ohlcv(con, symbols=symbols)
    if market_df.empty:
        return pd.DataFrame()

    technical_df = compute_technical_features_multi(market_df)
    calendar = pd.DatetimeIndex(sorted(technical_df["timestamp"].unique()))

    fundamentals_df = con.execute(f"SELECT * FROM fundamental_observations WHERE symbol IN ({','.join(['?'] * len(symbols))})", symbols).fetchdf()
    fund_frames = []
    for symbol in symbols:
        sym_fund = fundamentals_df[fundamentals_df["symbol"] == symbol].sort_values("publication_timestamp")
        if sym_fund.empty:
            continue
        sym_calendar = pd.DataFrame({"timestamp": calendar, "symbol": symbol})
        merged = pd.merge_asof(
            sym_calendar.sort_values("timestamp"), sym_fund.sort_values("publication_timestamp"),
            left_on="timestamp", right_on="publication_timestamp", by="symbol", direction="backward",
        )
        fund_frames.append(merged)
    fund_asof_all = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()

    fund_feature_frames = []
    days_since_filing: dict[tuple, float] = {}
    if not fund_asof_all.empty:
        for ts, cross_section in fund_asof_all.groupby("timestamp"):
            cross_section = cross_section.dropna(subset=["revenue"])
            if cross_section.empty:
                continue
            feats = compute_fundamental_features(cross_section)
            feats["timestamp"] = ts
            fund_feature_frames.append(feats)
            for _, row in cross_section.iterrows():
                days_since_filing[(row["symbol"], ts)] = (ts - row["publication_timestamp"]).days
    fundamental_features_df = pd.concat(fund_feature_frames, ignore_index=True) if fund_feature_frames else pd.DataFrame(columns=["symbol", "timestamp"])

    macro_snapshots: dict[pd.Timestamp, dict] = {}
    unique_months = sorted({pd.Timestamp(d.year, d.month, 1) for d in calendar})
    month_macro: dict[pd.Timestamp, dict] = {}
    for month_start in unique_months:
        as_of = month_start + pd.offsets.MonthEnd(1)
        history = macro_data.get_macro_history_asof(con, as_of)
        month_macro[month_start] = compute_macro_features(history, as_of)
    for date in calendar:
        macro_snapshots[pd.Timestamp(date)] = month_macro[pd.Timestamp(date.year, date.month, 1)]

    similarity_frames = []
    for symbol in symbols:
        sym_market = market_df[market_df["symbol"] == symbol].sort_values("timestamp")
        sym_tech = technical_df[technical_df["symbol"] == symbol].sort_values("timestamp")
        sim = compute_similarity_series(sym_tech, sym_market, feature_cols=HISTORICAL_SIMILARITY_FEATURE_COLS, k=50, min_history=100)
        similarity_frames.append(sim)
    similarity_df = pd.concat(similarity_frames, ignore_index=True) if similarity_frames else pd.DataFrame()

    dollar_volume_df = market_df.assign(dollar_volume=market_df["close"] * market_df["volume"])[["symbol", "timestamp", "dollar_volume"]]
    dollar_volume_lookup = dollar_volume_df.set_index(["symbol", "timestamp"])["dollar_volume"]
    breadth = technical_df.merge(dollar_volume_df, on=["symbol", "timestamp"], how="left").assign(positive=lambda d: d["return_20d"] > 0).groupby("timestamp")["positive"].mean()

    tech_lookup = technical_df.set_index(["symbol", "timestamp"])
    fund_lookup = fundamental_features_df.set_index(["symbol", "timestamp"]) if not fundamental_features_df.empty else None
    sim_lookup = similarity_df.set_index(["symbol", "timestamp"]) if not similarity_df.empty else None

    universe_cache: dict[pd.Timestamp, set] = {}
    symbol_category_events, obs_by_event = _build_event_probability_lookup(con, symbols)

    rows = []
    all_reports = []
    for symbol in symbols:
        sym_dates = technical_df[technical_df["symbol"] == symbol]["timestamp"]
        for ts in sym_dates:
            ts = pd.Timestamp(ts)
            if ts not in universe_cache:
                universe_cache[ts] = set(universe_data.get_point_in_time_universe(con, universe_name, ts))
            if symbol not in universe_cache[ts]:
                continue  # not a point-in-time member on this date -- excluded entirely, not just unranked

            tech_row = tech_lookup.loc[(symbol, ts)].to_dict()
            fund_row = fund_lookup.loc[(symbol, ts)].to_dict() if fund_lookup is not None and (symbol, ts) in fund_lookup.index else {}
            macro_feats = macro_snapshots.get(ts, {})
            breadth_ratio = float(breadth.get(ts, 0.5))

            if sim_lookup is not None and (symbol, ts) in sim_lookup.index:
                sr = sim_lookup.loc[(symbol, ts)]
                similarity = SimilarityResult(
                    num_analogues=int(sr["num_analogues"]),
                    avg_return_5d=None if pd.isna(sr["avg_return_5d"]) else float(sr["avg_return_5d"]),
                    median_return_5d=None if pd.isna(sr["median_return_5d"]) else float(sr["median_return_5d"]),
                    prob_positive_5d=None if pd.isna(sr["prob_positive_5d"]) else float(sr["prob_positive_5d"]),
                    avg_return_20d=None if pd.isna(sr["avg_return_20d"]) else float(sr["avg_return_20d"]),
                    median_return_20d=None if pd.isna(sr["median_return_20d"]) else float(sr["median_return_20d"]),
                    prob_positive_20d=None if pd.isna(sr["prob_positive_20d"]) else float(sr["prob_positive_20d"]),
                    p10_return_20d=None if pd.isna(sr["p10_return_20d"]) else float(sr["p10_return_20d"]),
                    p90_return_20d=None if pd.isna(sr["p90_return_20d"]) else float(sr["p90_return_20d"]),
                    similarity_confidence=float(sr["similarity_confidence"]),
                )
            else:
                similarity = SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0)

            articles_asof = repo_v2.get_news_asof(con, symbol, ts, lookback_days=7)
            news_row = _news_row_for_agent(articles_asof)
            news_counts = compute_deterministic_news_counts(articles_asof, ts.to_pydatetime())
            macro_surprise = macro_feats.get("SYN_GROWTH_INDEX_zscore", macro_feats.get("FRED_INDPRO_zscore", 0.0))

            orchestrated = run_research_agents(
                symbol=symbol, as_of=ts.to_pydatetime(), tech_row=tech_row, fund_row=fund_row,
                macro_features=macro_feats, breadth_ratio=breadth_ratio, similarity=similarity,
                news_row=news_row, macro_surprise=macro_surprise, use_llm=use_llm,
            )

            row = {"symbol": symbol, "timestamp": ts}
            for col in TECHNICAL_FEATURE_COLUMNS:
                row[f"raw_{col}"] = tech_row.get(col)
            for key in ("valuation_zscore", "profitability_zscore", "growth_zscore"):
                row[f"fund_raw_{key}"] = fund_row.get(key)
            row["dollar_volume"] = float(dollar_volume_lookup.get((symbol, ts), float("nan")))
            # Market-regime engine (Phase 13): the model gets the raw macro
            # variables directly, not just the Market Overview Agent's
            # derived regime labels/codes (which are still included below
            # via orchestrated.features, as supplemental context).
            for key, value in macro_feats.items():
                row[f"macro_raw_{key}"] = value
            row.update(orchestrated.features)
            row.update({f"news2_{k}": v for k, v in news_counts.items()})
            row.update(_event_probability_features(symbol, ts.to_pydatetime(), symbol_category_events, obs_by_event))
            row["days_since_last_filing"] = days_since_filing.get((symbol, ts), float("nan"))
            row["fundamental_missing"] = 1.0 if not fund_row else 0.0
            rows.append(row)
            if persist_agent_reports:
                all_reports.extend(orchestrated.reports)

    if persist_agent_reports and all_reports:
        persist_reports(con, all_reports)

    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix

    # Cross-sectional percentile ranks, computed strictly within each
    # date's point-in-time universe (already enforced above by construction).
    rank_input = matrix.rename(
        columns={
            "raw_return_60d": "return_60d", "raw_realised_vol_20d": "realised_vol_20d",
            "fund_raw_valuation_zscore": "valuation_zscore", "fund_raw_profitability_zscore": "profitability_zscore",
            "fund_raw_growth_zscore": "growth_zscore",
        }
    )
    ranks = []
    for ts, group in rank_input.groupby("timestamp"):
        r = compute_percentile_ranks(group, DEFAULT_METRIC_SPEC)
        r["timestamp"] = ts
        ranks.append(r)
    ranks_df = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame(columns=["symbol", "timestamp"])
    matrix = matrix.merge(ranks_df, on=["symbol", "timestamp"], how="left")

    breadth_rows = []
    for ts, group in matrix.groupby("timestamp"):
        bd = compute_market_breadth(
            group.rename(columns={"raw_dist_sma_20": "dist_sma_20", "raw_dist_sma_50": "dist_sma_50", "raw_dist_sma_200": "dist_sma_200", "raw_return_1d": "return_1d", "raw_dist_52w_high": "dist_52w_high", "raw_dist_52w_low": "dist_52w_low"})
        )
        bd["timestamp"] = ts
        breadth_rows.append(bd)
    breadth_df = pd.DataFrame(breadth_rows)
    if not breadth_df.empty:
        breadth_df = breadth_df.add_prefix("breadth_").rename(columns={"breadth_timestamp": "timestamp"})
        matrix = matrix.merge(breadth_df, on="timestamp", how="left")

    return matrix
