"""Feature Store.

Combines the outputs of every feature-engineering and research-agent
stage (technical, fundamental, macro, historical-similarity, news/event,
and agent-derived scores + disagreement) into one versioned, DuckDB-backed
table: ``feature_snapshots(feature_version, symbol, timestamp,
features_json)``.

This is the single hand-off point in the architecture diagram between
"Research Agents / Structured Features" and "ML Prediction Model" -- the
model layer (``models/``) only ever reads from here, never straight from
raw market/fundamental/macro data. Feature *values* are frozen once
written for a given ``(feature_version, symbol, timestamp)``; changing the
feature engineering logic requires bumping ``feature_version`` rather than
mutating history in place, mirroring the immutability discipline used for
predictions.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import polars as pl

from agents.orchestrator import persist_reports, run_research_agents
from core.logging import get_logger
from data import macro as macro_data
from data import market_data
from data import news as news_data
from features.fundamental import compute_fundamental_features
from features.historical import compute_similarity_series
from features.macro import compute_macro_features
from features.technical import TECHNICAL_FEATURE_COLUMNS, compute_technical_features_multi

logger = get_logger(__name__)

DEFAULT_FEATURE_VERSION = "fv1"

HISTORICAL_SIMILARITY_FEATURE_COLS = ["dist_sma_20", "dist_sma_50", "rsi_14", "realised_vol_20d", "return_20d"]


def _macro_feature_snapshots(con: duckdb.DuckDBPyConnection, calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, dict[str, float]]:
    """Precompute macro features for every calendar date (shared across symbols)."""
    snapshots: dict[pd.Timestamp, dict[str, float]] = {}
    # Macro publishes monthly; recomputing at every unique publication month
    # boundary and forward-filling is far cheaper than one DB round-trip per
    # trading day, and is exactly equivalent because compute_macro_features
    # only depends on data known as-of each date.
    unique_months = sorted({pd.Timestamp(d.year, d.month, 1) for d in calendar})
    month_features: dict[pd.Timestamp, dict[str, float]] = {}
    for month_start in unique_months:
        as_of = month_start + pd.offsets.MonthEnd(1)
        history = macro_data.get_macro_history_asof(con, as_of)
        month_features[month_start] = compute_macro_features(history, as_of)
    for date in calendar:
        key = pd.Timestamp(date.year, date.month, 1)
        snapshots[pd.Timestamp(date)] = month_features[key]
    return snapshots


def build_feature_matrix(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    use_llm: bool = False,
    persist_agent_reports: bool = True,
) -> pd.DataFrame:
    """Build the full historical feature matrix for every symbol.

    Returns a long DataFrame: one row per (symbol, timestamp), every
    feature as a column. Does not itself write to the feature store --
    see :func:`store_feature_matrix`.
    """
    symbols = symbols or market_data.get_symbols(con)
    market_df = market_data.get_ohlcv(con, symbols=symbols)
    fundamentals_df = con.execute("SELECT * FROM fundamental_observations").fetchdf()

    logger.info("computing_technical_features", n_symbols=len(symbols))
    technical_df = compute_technical_features_multi(market_df)

    calendar = pd.DatetimeIndex(sorted(technical_df["timestamp"].unique()))
    macro_snapshots = _macro_feature_snapshots(con, calendar)

    # Cross-sectional fundamentals-as-of, per symbol, forward-filled daily
    # via merge_asof on publication_timestamp (strictly backward-looking).
    logger.info("computing_fundamental_features")
    fund_frames = []
    for symbol in symbols:
        sym_calendar = pd.DataFrame({"timestamp": calendar, "symbol": symbol})
        sym_fund = fundamentals_df[fundamentals_df["symbol"] == symbol].sort_values("publication_timestamp")
        if sym_fund.empty:
            continue
        merged = pd.merge_asof(
            sym_calendar.sort_values("timestamp"),
            sym_fund.sort_values("publication_timestamp"),
            left_on="timestamp",
            right_on="publication_timestamp",
            by="symbol",
            direction="backward",
        )
        fund_frames.append(merged)
    fund_asof_all = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame()

    fund_feature_frames = []
    if not fund_asof_all.empty:
        for _timestamp, cross_section in fund_asof_all.groupby("timestamp"):
            cross_section = cross_section.dropna(subset=["revenue"])
            if cross_section.empty:
                continue
            feats = compute_fundamental_features(cross_section)
            feats["timestamp"] = _timestamp
            fund_feature_frames.append(feats)
    fundamental_features_df = (
        pd.concat(fund_feature_frames, ignore_index=True) if fund_feature_frames else pd.DataFrame(columns=["symbol", "timestamp"])
    )

    # Historical similarity, per symbol (vectorised batch).
    logger.info("computing_historical_similarity")
    similarity_frames = []
    for symbol in symbols:
        sym_market = market_df[market_df["symbol"] == symbol].sort_values("timestamp")
        sym_tech = technical_df[technical_df["symbol"] == symbol].sort_values("timestamp")
        sim = compute_similarity_series(
            sym_tech, sym_market, feature_cols=HISTORICAL_SIMILARITY_FEATURE_COLS, k=50, min_history=260
        )
        similarity_frames.append(sim)
    similarity_df = pd.concat(similarity_frames, ignore_index=True)

    news_df = news_data.get_news(con, symbols, calendar.min(), calendar.max())

    # Cross-sectional market breadth per day (fraction of symbols with
    # positive 20-day return), used by the Market Overview Agent.
    breadth = (
        technical_df.assign(positive=lambda d: d["return_20d"] > 0)
        .groupby("timestamp")["positive"]
        .mean()
    )

    logger.info("running_research_agents", n_symbols=len(symbols), n_dates=len(calendar))
    tech_lookup = technical_df.set_index(["symbol", "timestamp"])
    fund_lookup = fundamental_features_df.set_index(["symbol", "timestamp"]) if not fundamental_features_df.empty else None
    sim_lookup = similarity_df.set_index(["symbol", "timestamp"])
    news_lookup = news_df.set_index(["symbol", "timestamp"]) if not news_df.empty else None

    rows = []
    all_reports = []
    for symbol in symbols:
        sym_dates = technical_df[technical_df["symbol"] == symbol]["timestamp"]
        for ts in sym_dates:
            tech_row = tech_lookup.loc[(symbol, ts)].to_dict()
            fund_row = fund_lookup.loc[(symbol, ts)].to_dict() if fund_lookup is not None and (symbol, ts) in fund_lookup.index else {}
            macro_feats = macro_snapshots.get(pd.Timestamp(ts), {})
            breadth_ratio = float(breadth.get(ts, 0.5))
            similarity_row = sim_lookup.loc[(symbol, ts)]
            from features.historical import SimilarityResult

            similarity = SimilarityResult(
                num_analogues=int(similarity_row["num_analogues"]),
                avg_return_5d=_none_if_nan(similarity_row["avg_return_5d"]),
                median_return_5d=_none_if_nan(similarity_row["median_return_5d"]),
                prob_positive_5d=_none_if_nan(similarity_row["prob_positive_5d"]),
                avg_return_20d=_none_if_nan(similarity_row["avg_return_20d"]),
                median_return_20d=_none_if_nan(similarity_row["median_return_20d"]),
                prob_positive_20d=_none_if_nan(similarity_row["prob_positive_20d"]),
                p10_return_20d=_none_if_nan(similarity_row["p10_return_20d"]),
                p90_return_20d=_none_if_nan(similarity_row["p90_return_20d"]),
                similarity_confidence=float(similarity_row["similarity_confidence"]),
            )
            news_row = news_lookup.loc[(symbol, ts)].to_dict() if news_lookup is not None and (symbol, ts) in news_lookup.index else {}
            macro_surprise = macro_feats.get("SYN_GROWTH_INDEX_zscore", 0.0)

            orchestrated = run_research_agents(
                symbol=symbol,
                as_of=ts,
                tech_row=tech_row,
                fund_row=fund_row,
                macro_features=macro_feats,
                breadth_ratio=breadth_ratio,
                similarity=similarity,
                news_row=news_row,
                macro_surprise=macro_surprise,
                use_llm=use_llm,
            )
            row = {"symbol": symbol, "timestamp": ts}
            for col in TECHNICAL_FEATURE_COLUMNS:
                row[f"raw_{col}"] = tech_row.get(col)
            row.update(orchestrated.features)
            rows.append(row)
            if persist_agent_reports:
                all_reports.extend(orchestrated.reports)

    if persist_agent_reports and all_reports:
        persist_reports(con, all_reports)

    return pd.DataFrame(rows)


def _none_if_nan(value) -> float | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return None
    return float(value)


def store_feature_matrix(con: duckdb.DuckDBPyConnection, feature_version: str, matrix: pd.DataFrame) -> int:
    """Persist a feature matrix into ``feature_snapshots`` using Polars for
    the fast columnar -> JSON-per-row conversion, then bulk-load into DuckDB."""
    if matrix.empty:
        return 0
    feature_cols = [c for c in matrix.columns if c not in ("symbol", "timestamp")]
    pl_df = pl.from_pandas(matrix[feature_cols])
    features_json = pl_df.select(pl.struct(feature_cols).alias("features")).to_series().to_list()

    staging = pd.DataFrame(
        {
            "feature_version": feature_version,
            "symbol": matrix["symbol"].to_numpy(),
            "timestamp": matrix["timestamp"].to_numpy(),
            "features_json": [json.dumps(_clean_nans(d)) for d in features_json],
        }
    )
    con.execute("DELETE FROM feature_snapshots WHERE feature_version = ?", [feature_version])
    con.register("_feature_staging", staging)
    con.execute(
        "INSERT INTO feature_snapshots SELECT feature_version, symbol, timestamp, features_json FROM _feature_staging"
    )
    con.unregister("_feature_staging")
    return len(staging)


def _clean_nans(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def load_feature_matrix(
    con: duckdb.DuckDBPyConnection,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a stored feature matrix back into a wide pandas DataFrame."""
    query = "SELECT symbol, timestamp, features_json FROM feature_snapshots WHERE feature_version = ?"
    params: list = [feature_version]
    if symbols:
        query += f" AND symbol IN ({','.join(['?'] * len(symbols))})"
        params.extend(symbols)
    df = con.execute(query, params).fetchdf()
    if df.empty:
        return df
    feature_dicts = df["features_json"].map(json.loads)
    features_wide = pd.DataFrame(list(feature_dicts))
    return pd.concat([df[["symbol", "timestamp"]].reset_index(drop=True), features_wide], axis=1)
