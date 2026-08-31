"""DuckDB DDL for market-agent-lab's local storage (Version 0.1).

Every table here is designed so the same logical schema could later be
served by PostgreSQL/TimescaleDB instead of DuckDB -- plain columns,
explicit primary keys, no DuckDB-only types. Swapping the storage engine
should only require changing ``database/db.py``.
"""

from __future__ import annotations

DDL_STATEMENTS: list[str] = [
    # --- Raw observations ---------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS market_observations (
        symbol VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        adjusted_close DOUBLE NOT NULL,
        volume BIGINT NOT NULL,
        PRIMARY KEY (symbol, timestamp)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamental_observations (
        symbol VARCHAR NOT NULL,
        publication_timestamp TIMESTAMP NOT NULL,
        reporting_period_end TIMESTAMP NOT NULL,
        revenue DOUBLE, revenue_growth DOUBLE,
        eps DOUBLE, eps_growth DOUBLE,
        gross_margin DOUBLE, operating_margin DOUBLE,
        free_cash_flow DOUBLE, fcf_margin DOUBLE,
        roic DOUBLE, debt DOUBLE, cash DOUBLE,
        pe_ratio DOUBLE, ev_to_ebitda DOUBLE,
        price_to_book DOUBLE, price_to_sales DOUBLE,
        PRIMARY KEY (symbol, publication_timestamp)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS macro_observations (
        series_name VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        value DOUBLE NOT NULL,
        publication_timestamp TIMESTAMP NOT NULL,
        vintage_timestamp TIMESTAMP,
        PRIMARY KEY (series_name, timestamp, publication_timestamp)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_observations (
        symbol VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        news_sentiment DOUBLE NOT NULL,
        event_uncertainty DOUBLE NOT NULL,
        is_earnings_event BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (symbol, timestamp)
    );
    """,
    # --- Agents ---------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS agent_reports (
        agent VARCHAR NOT NULL,
        agent_version VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        features_json VARCHAR NOT NULL,
        confidence DOUBLE NOT NULL,
        evidence_refs_json VARCHAR NOT NULL,
        reasoning_summary VARCHAR,
        PRIMARY KEY (agent, symbol, timestamp)
    );
    """,
    # --- Feature store ----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS feature_snapshots (
        feature_version VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        features_json VARCHAR NOT NULL,
        PRIMARY KEY (feature_version, symbol, timestamp)
    );
    """,
    # --- ML predictions / outcomes (immutable) -----------------------------------
    """
    CREATE TABLE IF NOT EXISTS model_predictions (
        prediction_id VARCHAR PRIMARY KEY,
        model_version VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        symbol VARCHAR NOT NULL,
        predicted_excess_return_5d DOUBLE NOT NULL,
        predicted_excess_return_20d DOUBLE NOT NULL,
        probability_positive_5d DOUBLE NOT NULL,
        probability_positive_20d DOUBLE NOT NULL,
        predicted_volatility DOUBLE NOT NULL,
        confidence DOUBLE NOT NULL,
        feature_version VARCHAR NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        prediction_id VARCHAR PRIMARY KEY,
        realised_excess_return_5d DOUBLE,
        realised_excess_return_20d DOUBLE,
        realised_volatility DOUBLE,
        completion_timestamp TIMESTAMP NOT NULL
    );
    """,
    # --- Paper trading -----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS paper_orders (
        order_id VARCHAR PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        side VARCHAR NOT NULL,
        quantity DOUBLE NOT NULL,
        order_type VARCHAR NOT NULL,
        proposed_price DOUBLE NOT NULL,
        limit_price DOUBLE,
        timestamp TIMESTAMP NOT NULL,
        strategy_version VARCHAR NOT NULL,
        risk_approval_status VARCHAR NOT NULL,
        risk_reason_codes_json VARCHAR NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_fills (
        fill_id VARCHAR PRIMARY KEY,
        order_id VARCHAR NOT NULL,
        fill_timestamp TIMESTAMP NOT NULL,
        fill_price DOUBLE NOT NULL,
        quantity DOUBLE NOT NULL,
        slippage DOUBLE NOT NULL,
        commission DOUBLE NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_decisions (
        id VARCHAR PRIMARY KEY,
        order_id VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        symbol VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        reason_codes_json VARCHAR NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        timestamp TIMESTAMP NOT NULL,
        run_id VARCHAR NOT NULL,
        cash DOUBLE NOT NULL,
        equity DOUBLE NOT NULL,
        realised_pnl DOUBLE NOT NULL,
        unrealised_pnl DOUBLE NOT NULL,
        gross_exposure DOUBLE NOT NULL,
        net_exposure DOUBLE NOT NULL,
        drawdown DOUBLE NOT NULL,
        PRIMARY KEY (run_id, timestamp)
    );
    """,
    # --- Model lifecycle -----------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_version VARCHAR PRIMARY KEY,
        role VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL,
        feature_version VARCHAR NOT NULL,
        feature_names_json VARCHAR NOT NULL,
        hyperparameters_json VARCHAR NOT NULL,
        training_period_start TIMESTAMP NOT NULL,
        training_period_end TIMESTAMP NOT NULL,
        validation_period_start TIMESTAMP NOT NULL,
        validation_period_end TIMESTAMP NOT NULL,
        test_period_start TIMESTAMP,
        test_period_end TIMESTAMP,
        metrics_json VARCHAR NOT NULL,
        artifact_path VARCHAR NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS promotion_log (
        id VARCHAR PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        challenger_version VARCHAR NOT NULL,
        champion_version VARCHAR,
        decision VARCHAR NOT NULL,
        rationale VARCHAR NOT NULL,
        metrics_json VARCHAR NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS kill_switch_state (
        id INTEGER PRIMARY KEY,
        engaged BOOLEAN NOT NULL,
        reason VARCHAR,
        updated_at TIMESTAMP NOT NULL
    );
    """,
]


def init_schema(con) -> None:  # noqa: ANN001 - duckdb.DuckDBPyConnection
    for statement in DDL_STATEMENTS:
        con.execute(statement)
    con.execute(
        "INSERT INTO kill_switch_state (id, engaged, reason, updated_at) "
        "SELECT 1, FALSE, NULL, now() "
        "WHERE NOT EXISTS (SELECT 1 FROM kill_switch_state WHERE id = 1);"
    )
