"""Typed CRUD access to the DuckDB store.

This is the *only* module that should contain raw SQL for application
tables. Every other layer (features, models, execution, learning, API,
dashboard) should go through these functions so storage can later be
swapped for PostgreSQL/TimescaleDB without touching business logic.
"""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import pandas as pd

from core.schemas import (
    AgentReport,
    ModelPrediction,
    Outcome,
    PaperFill,
    PaperOrder,
    RiskApprovalStatus,
)

# --------------------------------------------------------------------------
# Raw observations (bulk, upsert semantics keyed on natural primary key)
# --------------------------------------------------------------------------


def _upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key_cols: list[str]) -> int:
    if df.empty:
        return 0
    con.register("_incoming_df", df)
    cols = list(df.columns)
    col_list = ", ".join(cols)
    key_pred = " AND ".join(f"t.{k} = s.{k}" for k in key_cols)
    con.execute(f"DELETE FROM {table} t USING _incoming_df s WHERE {key_pred};")
    con.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _incoming_df;")
    con.unregister("_incoming_df")
    return len(df)


def insert_market_observations(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _upsert_df(con, "market_observations", df, ["symbol", "timestamp"])


def insert_fundamental_observations(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _upsert_df(con, "fundamental_observations", df, ["symbol", "publication_timestamp"])


def insert_macro_observations(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _upsert_df(con, "macro_observations", df, ["series_name", "timestamp", "publication_timestamp"])


def insert_news_observations(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _upsert_df(con, "news_observations", df, ["symbol", "timestamp"])


def get_market_observations(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM market_observations WHERE 1=1"
    params: list = []
    if symbols:
        query += f" AND symbol IN ({','.join(['?'] * len(symbols))})"
        params.extend(symbols)
    if start is not None:
        query += " AND timestamp >= ?"
        params.append(start)
    if end is not None:
        query += " AND timestamp <= ?"
        params.append(end)
    query += " ORDER BY symbol, timestamp"
    return con.execute(query, params).fetchdf()


def get_fundamentals_asof(
    con: duckdb.DuckDBPyConnection, symbols: list[str], as_of: datetime
) -> pd.DataFrame:
    """Return the most-recently-*published* fundamental row per symbol as of ``as_of``.

    This is the leakage guard for fundamentals: it filters strictly on
    ``publication_timestamp <= as_of``, never on ``reporting_period_end``.
    """
    query = f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY symbol ORDER BY publication_timestamp DESC
            ) AS rn
            FROM fundamental_observations
            WHERE symbol IN ({",".join(["?"] * len(symbols))}) AND publication_timestamp <= ?
        ) WHERE rn = 1
    """
    df = con.execute(query, [*symbols, as_of]).fetchdf()
    return df.drop(columns=["rn"], errors="ignore")


def get_macro_asof(con: duckdb.DuckDBPyConnection, as_of: datetime) -> pd.DataFrame:
    """Latest published value per macro series as of ``as_of`` (publication-time filtered)."""
    query = """
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY series_name ORDER BY publication_timestamp DESC
            ) AS rn
            FROM macro_observations
            WHERE publication_timestamp <= ?
        ) WHERE rn = 1
    """
    df = con.execute(query, [as_of]).fetchdf()
    return df.drop(columns=["rn"], errors="ignore")


def get_macro_history_asof(con: duckdb.DuckDBPyConnection, as_of: datetime) -> pd.DataFrame:
    """Full macro history (all periods, all series) using the latest vintage
    known as of ``as_of`` for each (series, period). Used to build trailing
    z-score baselines without ever touching a later revision."""
    query = """
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY series_name, timestamp ORDER BY publication_timestamp DESC
            ) AS rn
            FROM macro_observations
            WHERE publication_timestamp <= ?
        ) WHERE rn = 1
    """
    df = con.execute(query, [as_of]).fetchdf()
    return df.drop(columns=["rn"], errors="ignore")


def get_news_observations(
    con: duckdb.DuckDBPyConnection, symbols: list[str], start: datetime, end: datetime
) -> pd.DataFrame:
    query = f"""
        SELECT * FROM news_observations
        WHERE symbol IN ({",".join(["?"] * len(symbols))}) AND timestamp BETWEEN ? AND ?
        ORDER BY symbol, timestamp
    """
    return con.execute(query, [*symbols, start, end]).fetchdf()


# --------------------------------------------------------------------------
# Agent reports
# --------------------------------------------------------------------------


def insert_agent_report(con: duckdb.DuckDBPyConnection, report: AgentReport) -> None:
    insert_agent_reports(con, [report])


def insert_agent_reports(con: duckdb.DuckDBPyConnection, reports: list[AgentReport]) -> int:
    """Bulk-insert agent reports. Prefer this over a report-at-a-time loop
    when persisting thousands of reports (e.g. full historical feature-store
    builds) -- row-at-a-time INSERT round-trips dominate runtime otherwise."""
    if not reports:
        return 0
    df = pd.DataFrame(
        [
            {
                "agent": r.agent,
                "agent_version": r.agent_version,
                "symbol": r.symbol,
                "timestamp": r.timestamp,
                "features_json": json.dumps(r.features),
                "confidence": r.confidence,
                "evidence_refs_json": json.dumps(r.evidence_refs),
                "reasoning_summary": r.reasoning_summary,
            }
            for r in reports
        ]
    )
    # The Market Overview Agent emits one report per date shared across every
    # symbol (symbol="_MARKET_"), so a batch spanning the whole universe
    # naturally contains duplicate (agent, symbol, timestamp) keys -- drop
    # them before the bulk insert (content is identical for a given date).
    df = df.drop_duplicates(subset=["agent", "symbol", "timestamp"], keep="last")
    return _upsert_df(con, "agent_reports", df, ["agent", "symbol", "timestamp"])


def get_agent_reports(
    con: duckdb.DuckDBPyConnection, symbol: str, as_of: datetime, agents: list[str] | None = None
) -> list[AgentReport]:
    query = "SELECT * FROM agent_reports WHERE symbol = ? AND timestamp <= ?"
    params: list = [symbol, as_of]
    if agents:
        query += f" AND agent IN ({','.join(['?'] * len(agents))})"
        params.extend(agents)
    query += " ORDER BY timestamp DESC"
    rows = con.execute(query, params).fetchall()
    cols = [d[0] for d in con.description]
    reports = []
    for row in rows:
        d = dict(zip(cols, row, strict=True))
        reports.append(
            AgentReport(
                agent=d["agent"],
                agent_version=d["agent_version"],
                symbol=d["symbol"],
                timestamp=d["timestamp"],
                features=json.loads(d["features_json"]),
                confidence=d["confidence"],
                evidence_refs=json.loads(d["evidence_refs_json"]),
                reasoning_summary=d["reasoning_summary"],
            )
        )
    return reports


# --------------------------------------------------------------------------
# Predictions / outcomes (immutable)
# --------------------------------------------------------------------------


def insert_prediction(con: duckdb.DuckDBPyConnection, prediction: ModelPrediction) -> None:
    existing = con.execute(
        "SELECT 1 FROM model_predictions WHERE prediction_id = ?", [prediction.prediction_id]
    ).fetchone()
    if existing:
        raise ValueError(
            f"prediction_id={prediction.prediction_id} already recorded; predictions are immutable"
        )
    con.execute(
        """
        INSERT INTO model_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            prediction.prediction_id,
            prediction.model_version,
            prediction.timestamp,
            prediction.symbol,
            prediction.predicted_excess_return_5d,
            prediction.predicted_excess_return_20d,
            prediction.probability_positive_5d,
            prediction.probability_positive_20d,
            prediction.predicted_volatility,
            prediction.confidence,
            prediction.feature_version,
        ],
    )


def get_predictions(
    con: duckdb.DuckDBPyConnection,
    model_version: str | None = None,
    symbol: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM model_predictions WHERE 1=1"
    params: list = []
    if model_version:
        query += " AND model_version = ?"
        params.append(model_version)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if start is not None:
        query += " AND timestamp >= ?"
        params.append(start)
    if end is not None:
        query += " AND timestamp <= ?"
        params.append(end)
    return con.execute(query, params).fetchdf()


def insert_outcome(con: duckdb.DuckDBPyConnection, outcome: Outcome) -> None:
    existing = con.execute(
        "SELECT 1 FROM outcomes WHERE prediction_id = ?", [outcome.prediction_id]
    ).fetchone()
    if existing:
        raise ValueError(f"outcome for prediction_id={outcome.prediction_id} already recorded")
    con.execute(
        "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?)",
        [
            outcome.prediction_id,
            outcome.realised_excess_return_5d,
            outcome.realised_excess_return_20d,
            outcome.realised_volatility,
            outcome.completion_timestamp,
        ],
    )


def get_predictions_without_outcome(
    con: duckdb.DuckDBPyConnection, horizon_days: int, as_of: datetime
) -> pd.DataFrame:
    """Predictions old enough (>= horizon_days before ``as_of``) that lack a recorded outcome."""
    query = """
        SELECT p.* FROM model_predictions p
        LEFT JOIN outcomes o ON p.prediction_id = o.prediction_id
        WHERE o.prediction_id IS NULL
          AND p.timestamp <= ?
    """
    cutoff = as_of - pd.Timedelta(days=horizon_days)
    return con.execute(query, [cutoff]).fetchdf()


# --------------------------------------------------------------------------
# Paper trading
# --------------------------------------------------------------------------


def insert_paper_order(con: duckdb.DuckDBPyConnection, order: PaperOrder) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO paper_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            order.order_id,
            order.symbol,
            order.side.value,
            order.quantity,
            order.order_type.value,
            order.proposed_price,
            order.limit_price,
            order.timestamp,
            order.strategy_version,
            order.risk_approval_status.value,
            json.dumps([c.value for c in order.risk_reason_codes]),
        ],
    )


def get_paper_orders(
    con: duckdb.DuckDBPyConnection, status: RiskApprovalStatus | None = None
) -> pd.DataFrame:
    query = "SELECT * FROM paper_orders WHERE 1=1"
    params: list = []
    if status is not None:
        query += " AND risk_approval_status = ?"
        params.append(status.value)
    query += " ORDER BY timestamp"
    return con.execute(query, params).fetchdf()


def insert_paper_fill(con: duckdb.DuckDBPyConnection, fill: PaperFill) -> None:
    con.execute(
        "INSERT INTO paper_fills VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            fill.fill_id,
            fill.order_id,
            fill.fill_timestamp,
            fill.fill_price,
            fill.quantity,
            fill.slippage,
            fill.commission,
        ],
    )


def get_paper_fills(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM paper_fills ORDER BY fill_timestamp").fetchdf()


def insert_risk_decision(
    con: duckdb.DuckDBPyConnection,
    order_id: str,
    timestamp: datetime,
    symbol: str,
    status: RiskApprovalStatus,
    reason_codes: list[str],
) -> None:
    import uuid

    con.execute(
        "INSERT INTO risk_decisions VALUES (?, ?, ?, ?, ?, ?)",
        [uuid.uuid4().hex, order_id, timestamp, symbol, status.value, json.dumps(reason_codes)],
    )


def get_risk_decisions(con: duckdb.DuckDBPyConnection, status: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM risk_decisions WHERE 1=1"
    params: list = []
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY timestamp"
    return con.execute(query, params).fetchdf()


def insert_portfolio_snapshot(con: duckdb.DuckDBPyConnection, run_id: str, snapshot: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            snapshot["timestamp"],
            run_id,
            snapshot["cash"],
            snapshot["equity"],
            snapshot["realized_pnl"],
            snapshot["unrealized_pnl"],
            snapshot["gross_exposure"],
            snapshot["net_exposure"],
            snapshot["drawdown"],
        ],
    )


def get_portfolio_snapshots(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM portfolio_snapshots WHERE run_id = ? ORDER BY timestamp", [run_id]
    ).fetchdf()


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------


def upsert_model_registry(con: duckdb.DuckDBPyConnection, record: dict) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO model_registry VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record["model_version"],
            record["role"],
            record["created_at"],
            record["feature_version"],
            json.dumps(record["feature_names"]),
            json.dumps(record["hyperparameters"]),
            record["training_period_start"],
            record["training_period_end"],
            record["validation_period_start"],
            record["validation_period_end"],
            record.get("test_period_start"),
            record.get("test_period_end"),
            json.dumps(record["metrics"]),
            record["artifact_path"],
        ],
    )


def get_model_registry(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM model_registry ORDER BY created_at").fetchdf()


def get_champion(con: duckdb.DuckDBPyConnection) -> dict | None:
    row = con.execute(
        "SELECT * FROM model_registry WHERE role = 'CHAMPION' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in con.description]
    return dict(zip(cols, row, strict=True))


def insert_promotion_log(con: duckdb.DuckDBPyConnection, record: dict) -> None:
    con.execute(
        "INSERT INTO promotion_log VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            record["id"],
            record["timestamp"],
            record["challenger_version"],
            record.get("champion_version"),
            record["decision"],
            record["rationale"],
            json.dumps(record["metrics"]),
        ],
    )


def get_promotion_log(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM promotion_log ORDER BY timestamp").fetchdf()


def get_kill_switch(con: duckdb.DuckDBPyConnection) -> tuple[bool, str | None]:
    row = con.execute("SELECT engaged, reason FROM kill_switch_state WHERE id = 1").fetchone()
    if row is None:
        return False, None
    return bool(row[0]), row[1]


def set_kill_switch(con: duckdb.DuckDBPyConnection, engaged: bool, reason: str | None) -> None:
    con.execute(
        "UPDATE kill_switch_state SET engaged = ?, reason = ?, updated_at = now() WHERE id = 1",
        [engaged, reason],
    )
