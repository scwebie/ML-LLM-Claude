"""Typed CRUD access for the Version 0.2 real-data tables.

Kept separate from ``database/repository.py`` (V0.1) purely for file size
-- same conventions (bulk upsert via ``_upsert_df``, one function per
table), same rule (no raw SQL for these tables anywhere else).
"""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pandas as pd

from core.schemas_v2 import (
    CorporateAction,
    DataIngestionRun,
    DataLineage,
    DataQualityFlag,
    DataSource,
    EventProbabilityObservation,
    EventSymbolMapping,
    FundamentalFact,
    LeakageAuditResult,
    NewsArticle,
    PriceReconciliation,
    SecFiling,
    UniverseMembership,
)
from database.repository import _upsert_df

# --------------------------------------------------------------------------
# Provider / provenance
# --------------------------------------------------------------------------


def upsert_data_source(con: duckdb.DuckDBPyConnection, source: DataSource) -> None:
    con.execute(
        "INSERT OR REPLACE INTO data_sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source.source_id, source.name, source.category.value, source.tier,
            source.requires_api_key, source.base_url, source.notes, source.is_enabled,
        ],
    )


def get_data_sources(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM data_sources ORDER BY category, source_id").fetchdf()


def insert_ingestion_run(con: duckdb.DuckDBPyConnection, run: DataIngestionRun) -> None:
    con.execute(
        "INSERT OR REPLACE INTO data_ingestion_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run.run_id, run.source_id, run.category.value, run.started_at, run.finished_at,
            run.status.value, run.records_ingested, run.error_message, json.dumps(run.symbols),
        ],
    )


def get_ingestion_runs(con: duckdb.DuckDBPyConnection, source_id: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM data_ingestion_runs"
    params: list = []
    if source_id:
        query += " WHERE source_id = ?"
        params.append(source_id)
    query += " ORDER BY started_at DESC"
    return con.execute(query, params).fetchdf()


def insert_lineage(con: duckdb.DuckDBPyConnection, records: list[DataLineage]) -> int:
    if not records:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": r.id, "feature_version": r.feature_version, "symbol": r.symbol,
                "timestamp": r.timestamp, "source_table": r.source_table, "source_ref": r.source_ref,
                "created_at": datetime.now(),
            }
            for r in records
        ]
    )
    con.register("_lineage_df", df)
    con.execute("INSERT INTO data_lineage SELECT * FROM _lineage_df")
    con.unregister("_lineage_df")
    return len(records)


def get_lineage(con: duckdb.DuckDBPyConnection, symbol: str, timestamp: datetime, feature_version: str) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM data_lineage WHERE symbol = ? AND timestamp = ? AND feature_version = ?",
        [symbol, timestamp, feature_version],
    ).fetchdf()


# --------------------------------------------------------------------------
# Corporate actions / universe
# --------------------------------------------------------------------------


def insert_corporate_actions(con: duckdb.DuckDBPyConnection, actions: list[CorporateAction]) -> int:
    if not actions:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": a.id, "symbol": a.symbol, "action_type": a.action_type.value, "ex_date": a.ex_date,
                "ratio": a.ratio, "cash_amount": a.cash_amount, "new_symbol": a.new_symbol,
                "source": a.source, "retrieved_at": a.retrieved_at,
            }
            for a in actions
        ]
    )
    return _upsert_df(con, "corporate_actions", df, ["id"])


def get_corporate_actions(con: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    return con.execute("SELECT * FROM corporate_actions WHERE symbol = ? ORDER BY ex_date", [symbol]).fetchdf()


def insert_universe_membership(con: duckdb.DuckDBPyConnection, memberships: list[UniverseMembership]) -> int:
    if not memberships:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": m.id, "universe_name": m.universe_name, "symbol": m.symbol,
                "start_date": m.start_date, "end_date": m.end_date, "source": m.source, "notes": m.notes,
            }
            for m in memberships
        ]
    )
    return _upsert_df(con, "universe_membership", df, ["id"])


def get_point_in_time_universe(con: duckdb.DuckDBPyConnection, universe_name: str, as_of: datetime) -> list[str]:
    """Symbols that were members of ``universe_name`` at ``as_of`` -- i.e.
    ``start_date <= as_of AND (end_date IS NULL OR end_date > as_of)``."""
    rows = con.execute(
        "SELECT DISTINCT symbol FROM universe_membership "
        "WHERE universe_name = ? AND start_date <= ? AND (end_date IS NULL OR end_date > ?)",
        [universe_name, as_of, as_of],
    ).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------
# Price reconciliation
# --------------------------------------------------------------------------


def insert_price_reconciliations(con: duckdb.DuckDBPyConnection, records: list[PriceReconciliation]) -> int:
    if not records:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": r.id, "symbol": r.symbol, "date": r.date, "primary_source": r.primary_source,
                "primary_close": r.primary_close, "secondary_source": r.secondary_source,
                "secondary_close": r.secondary_close, "abs_pct_diff": r.abs_pct_diff,
                "status": r.status.value, "created_at": r.created_at,
            }
            for r in records
        ]
    )
    return _upsert_df(con, "price_reconciliation", df, ["id"])


def get_price_reconciliations(con: duckdb.DuckDBPyConnection, symbol: str | None = None, status: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM price_reconciliation WHERE 1=1"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if status:
        query += " AND status = ?"
        params.append(status)
    return con.execute(query, params).fetchdf()


# --------------------------------------------------------------------------
# SEC fundamentals
# --------------------------------------------------------------------------


def insert_sec_filings(con: duckdb.DuckDBPyConnection, filings: list[SecFiling]) -> int:
    if not filings:
        return 0
    df = pd.DataFrame(
        [
            {
                "accession_number": f.accession_number, "cik": f.cik, "symbol": f.symbol,
                "form_type": f.form_type, "filing_period_end": f.filing_period_end,
                "filing_date": f.filing_date, "accepted_timestamp": f.accepted_timestamp,
                "source_url": f.source_url, "retrieved_at": f.retrieved_at,
            }
            for f in filings
        ]
    )
    return _upsert_df(con, "sec_filings", df, ["accession_number"])


def insert_fundamental_facts(con: duckdb.DuckDBPyConnection, facts: list[FundamentalFact]) -> int:
    if not facts:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": f.id, "symbol": f.symbol, "cik": f.cik, "tag": f.tag, "unit": f.unit,
                "period_start": f.period_start, "period_end": f.period_end, "value": f.value,
                "accession_number": f.accession_number, "form_type": f.form_type,
                "fiscal_year": f.fiscal_year, "fiscal_period": f.fiscal_period,
                "filed_date": f.filed_date, "source": f.source, "retrieved_at": f.retrieved_at,
            }
            for f in facts
        ]
    )
    return _upsert_df(con, "fundamental_facts", df, ["id"])


def get_fundamental_facts_asof(con: duckdb.DuckDBPyConnection, symbol: str, as_of: datetime, tags: list[str] | None = None) -> pd.DataFrame:
    """For each requested tag, the most recent fact whose ``filed_date`` is
    <= as_of. This is the point-in-time-correct read path -- period_end is
    never used for filtering."""
    query = """
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY tag ORDER BY filed_date DESC, period_end DESC
            ) AS rn
            FROM fundamental_facts
            WHERE symbol = ? AND filed_date <= ?
    """
    params: list = [symbol, as_of]
    if tags:
        query += f" AND tag IN ({','.join(['?'] * len(tags))})"
        params.extend(tags)
    query += ") WHERE rn = 1"
    df = con.execute(query, params).fetchdf()
    return df.drop(columns=["rn"], errors="ignore")


# --------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------


def insert_news_articles(con: duckdb.DuckDBPyConnection, articles: list[NewsArticle]) -> int:
    if not articles:
        return 0
    df = pd.DataFrame(
        [
            {
                "article_id": a.article_id, "headline": a.headline, "published_at": a.published_at,
                "retrieved_at": a.retrieved_at, "source": a.source, "publisher": a.publisher,
                "tier": a.tier.value, "url": a.url, "event_category": a.event_category,
                "language": a.language, "excerpt": a.excerpt,
                "timestamp_uncertain": a.timestamp_uncertain, "dedupe_key": a.dedupe_key,
            }
            for a in articles
        ]
    )
    n = _upsert_df(con, "news_articles", df, ["article_id"])

    entity_rows = [{"article_id": a.article_id, "symbol": s, "relevance": 1.0} for a in articles for s in a.symbols]
    if entity_rows:
        edf = pd.DataFrame(entity_rows)
        _upsert_df(con, "news_entities", edf, ["article_id", "symbol"])
    return n


def get_news_asof(
    con: duckdb.DuckDBPyConnection, symbol: str, as_of: datetime, lookback_days: int = 30,
    include_timestamp_uncertain: bool = False,
) -> pd.DataFrame:
    """News for ``symbol`` published strictly at or before ``as_of``. This
    is THE point-in-time guard for the news pipeline: an ingestion
    timestamp is never substituted for a missing publication timestamp,
    and by default TIMESTAMP_UNCERTAIN articles are excluded entirely."""
    query = """
        SELECT a.*, e.relevance FROM news_articles a
        JOIN news_entities e ON a.article_id = e.article_id
        WHERE e.symbol = ? AND a.published_at IS NOT NULL AND a.published_at <= ?
          AND a.published_at >= ?
    """
    params: list = [symbol, as_of, as_of - pd.Timedelta(days=lookback_days)]
    if not include_timestamp_uncertain:
        query += " AND a.timestamp_uncertain = FALSE"
    return con.execute(query, params).fetchdf()


def is_duplicate_article(con: duckdb.DuckDBPyConnection, dedupe_key: str) -> bool:
    row = con.execute("SELECT 1 FROM news_articles WHERE dedupe_key = ? LIMIT 1", [dedupe_key]).fetchone()
    return row is not None


# --------------------------------------------------------------------------
# Event probability (read-only research signal)
# --------------------------------------------------------------------------


def insert_event_probability_observations(con: duckdb.DuckDBPyConnection, observations: list[EventProbabilityObservation]) -> int:
    if not observations:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": o.id, "event_id": o.event_id, "question": o.question, "category": o.category,
                "observed_timestamp": o.observed_timestamp, "resolution_date": o.resolution_date,
                "public_probability": o.public_probability,
                "liquidity_json": json.dumps(o.liquidity_metadata) if o.liquidity_metadata else None,
                "volume_json": json.dumps(o.volume_metadata) if o.volume_metadata else None,
                "source": o.source, "retrieved_at": o.retrieved_at,
            }
            for o in observations
        ]
    )
    return _upsert_df(con, "event_probability_observations", df, ["id"])


def get_event_probabilities_asof(con: duckdb.DuckDBPyConnection, event_id: str, as_of: datetime) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM event_probability_observations WHERE event_id = ? AND observed_timestamp <= ? "
        "ORDER BY observed_timestamp DESC LIMIT 1",
        [event_id, as_of],
    ).fetchdf()


def insert_event_symbol_mappings(con: duckdb.DuckDBPyConnection, mappings: list[EventSymbolMapping]) -> int:
    if not mappings:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": m.id, "event_id": m.event_id, "symbol": m.symbol, "relevance": m.relevance,
                "rationale_category": m.rationale_category, "created_at": m.created_at,
            }
            for m in mappings
        ]
    )
    return _upsert_df(con, "event_symbol_mappings", df, ["id"])


def get_event_mappings_for_symbol(con: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    return con.execute("SELECT * FROM event_symbol_mappings WHERE symbol = ?", [symbol]).fetchdf()


# --------------------------------------------------------------------------
# Data quality / leakage auditing
# --------------------------------------------------------------------------


def insert_quality_flags(con: duckdb.DuckDBPyConnection, flags: list[DataQualityFlag]) -> int:
    if not flags:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": f.id, "category": f.category, "entity_ref": f.entity_ref,
                "observation_timestamp": f.observation_timestamp, "flag_type": f.flag_type,
                "severity": f.severity.value, "details": f.details, "created_at": f.created_at,
            }
            for f in flags
        ]
    )
    return _upsert_df(con, "data_quality_flags", df, ["id"])


def get_quality_flags(con: duckdb.DuckDBPyConnection, category: str | None = None, severity: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM data_quality_flags WHERE 1=1"
    params: list = []
    if category:
        query += " AND category = ?"
        params.append(category)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    return con.execute(query, params).fetchdf()


def insert_leakage_audit_results(con: duckdb.DuckDBPyConnection, results: list[LeakageAuditResult]) -> int:
    if not results:
        return 0
    df = pd.DataFrame(
        [
            {
                "id": r.id, "run_id": r.run_id, "check_type": r.check_type, "entity_ref": r.entity_ref,
                "prediction_timestamp": r.prediction_timestamp, "information_timestamp": r.information_timestamp,
                "passed": r.passed, "details": r.details, "created_at": r.created_at,
            }
            for r in results
        ]
    )
    return _upsert_df(con, "leakage_audit_results", df, ["id"])


def get_leakage_audit_results(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    return con.execute("SELECT * FROM leakage_audit_results WHERE run_id = ?", [run_id]).fetchdf()


# --------------------------------------------------------------------------
# Model evaluations (robustness suite outputs)
# --------------------------------------------------------------------------


def insert_model_evaluation(con: duckdb.DuckDBPyConnection, model_version: str, evaluation_type: str, payload: dict) -> None:
    import uuid

    con.execute(
        "INSERT INTO model_evaluations VALUES (?, ?, ?, ?, ?)",
        [uuid.uuid4().hex, model_version, evaluation_type, json.dumps(payload, default=str), datetime.now()],
    )


def get_model_evaluations(con: duckdb.DuckDBPyConnection, model_version: str, evaluation_type: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM model_evaluations WHERE model_version = ?"
    params: list = [model_version]
    if evaluation_type:
        query += " AND evaluation_type = ?"
        params.append(evaluation_type)
    return con.execute(query, params).fetchdf()
