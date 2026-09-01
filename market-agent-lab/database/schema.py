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

# ==========================================================================
# Version 0.2 additions -- real-data research pipeline.
#
# Purely additive: every V0.1 table above is untouched. Where a V0.1 table's
# shape already fits real data (``market_observations`` for OHLCV,
# ``macro_observations`` for vintage/release-aware macro series --
# it already carries publication_timestamp + vintage_timestamp),
# real providers write into that SAME table with a populated ``source``
# column rather than duplicating it under a new name.
# ==========================================================================
V02_DDL_STATEMENTS: list[str] = [
    # --- Provenance / provider bookkeeping -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS data_sources (
        source_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        tier VARCHAR,
        requires_api_key BOOLEAN NOT NULL DEFAULT FALSE,
        base_url VARCHAR,
        notes VARCHAR,
        is_enabled BOOLEAN NOT NULL DEFAULT TRUE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS data_ingestion_runs (
        run_id VARCHAR PRIMARY KEY,
        source_id VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        started_at TIMESTAMP NOT NULL,
        finished_at TIMESTAMP,
        status VARCHAR NOT NULL,
        records_ingested BIGINT NOT NULL DEFAULT 0,
        error_message VARCHAR,
        symbols_json VARCHAR
    );
    """,
    # --- Corporate actions / point-in-time universe ----------------------------
    """
    CREATE TABLE IF NOT EXISTS corporate_actions (
        id VARCHAR PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        action_type VARCHAR NOT NULL,
        ex_date TIMESTAMP NOT NULL,
        ratio DOUBLE,
        cash_amount DOUBLE,
        new_symbol VARCHAR,
        source VARCHAR NOT NULL,
        retrieved_at TIMESTAMP NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_membership (
        id VARCHAR PRIMARY KEY,
        universe_name VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        start_date TIMESTAMP NOT NULL,
        end_date TIMESTAMP,
        source VARCHAR NOT NULL,
        notes VARCHAR
    );
    """,
    # --- Price reconciliation ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS price_reconciliation (
        id VARCHAR PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        date TIMESTAMP NOT NULL,
        primary_source VARCHAR,
        primary_close DOUBLE,
        secondary_source VARCHAR,
        secondary_close DOUBLE,
        abs_pct_diff DOUBLE,
        status VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """,
    # --- SEC fundamentals ------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sec_filings (
        accession_number VARCHAR PRIMARY KEY,
        cik VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        form_type VARCHAR NOT NULL,
        filing_period_end TIMESTAMP,
        filing_date TIMESTAMP NOT NULL,
        accepted_timestamp TIMESTAMP,
        source_url VARCHAR,
        retrieved_at TIMESTAMP NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fundamental_facts (
        id VARCHAR PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        cik VARCHAR NOT NULL,
        tag VARCHAR NOT NULL,
        unit VARCHAR NOT NULL,
        period_start TIMESTAMP,
        period_end TIMESTAMP NOT NULL,
        value DOUBLE NOT NULL,
        accession_number VARCHAR,
        form_type VARCHAR,
        fiscal_year INTEGER,
        fiscal_period VARCHAR,
        filed_date TIMESTAMP NOT NULL,
        source VARCHAR NOT NULL,
        retrieved_at TIMESTAMP NOT NULL
    );
    """,
    # --- News / events -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS news_articles (
        article_id VARCHAR PRIMARY KEY,
        headline VARCHAR NOT NULL,
        published_at TIMESTAMP,
        retrieved_at TIMESTAMP NOT NULL,
        source VARCHAR NOT NULL,
        publisher VARCHAR,
        tier VARCHAR NOT NULL,
        url VARCHAR,
        event_category VARCHAR,
        language VARCHAR,
        excerpt VARCHAR,
        timestamp_uncertain BOOLEAN NOT NULL DEFAULT FALSE,
        dedupe_key VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_entities (
        article_id VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        relevance DOUBLE NOT NULL DEFAULT 1.0,
        PRIMARY KEY (article_id, symbol)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_agent_features (
        id VARCHAR PRIMARY KEY,
        article_id VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        sentiment DOUBLE,
        impact_magnitude DOUBLE,
        uncertainty DOUBLE,
        novelty DOUBLE,
        event_category VARCHAR,
        expected_horizon VARCHAR,
        llm_model VARCHAR,
        prompt_version VARCHAR,
        generated_at TIMESTAMP NOT NULL
    );
    """,
    # --- Read-only public event-probability research signal ------------------------
    """
    CREATE TABLE IF NOT EXISTS event_probability_observations (
        id VARCHAR PRIMARY KEY,
        event_id VARCHAR NOT NULL,
        question VARCHAR NOT NULL,
        category VARCHAR,
        observed_timestamp TIMESTAMP NOT NULL,
        resolution_date TIMESTAMP,
        public_probability DOUBLE NOT NULL,
        liquidity_json VARCHAR,
        volume_json VARCHAR,
        source VARCHAR NOT NULL,
        retrieved_at TIMESTAMP NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS event_symbol_mappings (
        id VARCHAR PRIMARY KEY,
        event_id VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        relevance DOUBLE NOT NULL,
        rationale_category VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """,
    # --- Data quality / leakage auditing --------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS data_quality_flags (
        id VARCHAR PRIMARY KEY,
        category VARCHAR NOT NULL,
        entity_ref VARCHAR NOT NULL,
        observation_timestamp TIMESTAMP,
        flag_type VARCHAR NOT NULL,
        severity VARCHAR NOT NULL,
        details VARCHAR,
        created_at TIMESTAMP NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS leakage_audit_results (
        id VARCHAR PRIMARY KEY,
        run_id VARCHAR NOT NULL,
        check_type VARCHAR NOT NULL,
        entity_ref VARCHAR,
        prediction_timestamp TIMESTAMP,
        information_timestamp TIMESTAMP,
        passed BOOLEAN NOT NULL,
        details VARCHAR,
        created_at TIMESTAMP NOT NULL
    );
    """,
    # --- Data lineage (traces a feature row back to its source records) ------------
    """
    CREATE TABLE IF NOT EXISTS data_lineage (
        id VARCHAR PRIMARY KEY,
        feature_version VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        source_table VARCHAR NOT NULL,
        source_ref VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """,
    # --- Robustness / evaluation suite outputs --------------------------------------
    """
    CREATE TABLE IF NOT EXISTS model_evaluations (
        id VARCHAR PRIMARY KEY,
        model_version VARCHAR NOT NULL,
        evaluation_type VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """,
    # --- Final holdout access audit trail (Stage 12) --------------------------------
    # Every call to backtesting.holdout.evaluate_on_holdout writes exactly one row
    # here -- the audit trail proving the holdout period was touched only for a
    # final, formal evaluation and never during walk-forward model development.
    """
    CREATE TABLE IF NOT EXISTS holdout_access_log (
        id VARCHAR PRIMARY KEY,
        accessed_at TIMESTAMP NOT NULL,
        purpose VARCHAR NOT NULL,
        model_version VARCHAR NOT NULL,
        holdout_start TIMESTAMP NOT NULL,
        holdout_end TIMESTAMP NOT NULL,
        n_rows INTEGER NOT NULL,
        symbols VARCHAR
    );
    """,
]


def init_schema(con) -> None:  # noqa: ANN001 - duckdb.DuckDBPyConnection
    for statement in DDL_STATEMENTS:
        con.execute(statement)
    for statement in V02_DDL_STATEMENTS:
        con.execute(statement)
    con.execute(
        "INSERT INTO kill_switch_state (id, engaged, reason, updated_at) "
        "SELECT 1, FALSE, NULL, now() "
        "WHERE NOT EXISTS (SELECT 1 FROM kill_switch_state WHERE id = 1);"
    )
