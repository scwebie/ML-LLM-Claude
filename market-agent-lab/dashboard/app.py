"""Streamlit monitoring dashboard for market-agent-lab (Phase 14).

Read-only: this dashboard never places, approves, or modifies an order or
a risk limit. It only visualises what the pipeline (``main.py demo``,
``scripts/run_backtest.py``, ...) has already written to DuckDB.

Run with: ``uv run streamlit run dashboard/app.py`` (or ``uv run python
main.py serve-dashboard``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from backtesting.engine import buy_and_hold_benchmark
from backtesting.metrics import daily_returns
from data import synthetic as synthetic_data
from data.market_data import get_ohlcv
from data.providers import registry as provider_registry
from database import repository as repo
from database import repository_v2 as repo_v2
from database.db import get_connection
from features.feature_store import DEFAULT_FEATURE_VERSION, load_feature_matrix
from models import registry as model_registry
from models.evaluate import calibration_curve, feature_importance

st.set_page_config(page_title="market-agent-lab", layout="wide")
st.title("market-agent-lab -- Version 0.1 + 0.2 (PAPER TRADING / SIMULATION ONLY)")
st.caption(
    "This dashboard visualises a fully simulated research + trading pipeline. "
    "No live brokerage or prediction-market connection exists anywhere in this system."
)

con = get_connection()
SECTOR_MAP = dict(synthetic_data.SYMBOLS)

tab_portfolio, tab_model, tab_agents, tab_backtest, tab_risk, tab_provider_health, tab_data_quality, tab_robustness = st.tabs(
    ["Portfolio", "Model", "Agents", "Backtest", "Risk", "Provider Health", "Data Quality", "Robustness"]
)

# --------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------
with tab_portfolio:
    run_ids = con.execute("SELECT DISTINCT run_id FROM portfolio_snapshots ORDER BY run_id DESC").fetchdf()
    if run_ids.empty:
        st.info("No backtest/demo runs recorded yet. Run `uv run python main.py demo` first.")
    else:
        run_id = st.selectbox("Run", run_ids["run_id"].tolist())
        snapshots = repo.get_portfolio_snapshots(con, run_id)
        if snapshots.empty:
            st.warning("No snapshots for this run.")
        else:
            snapshots = snapshots.sort_values("timestamp")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Equity", f"${snapshots['equity'].iloc[-1]:,.0f}")
            col2.metric("Cash", f"${snapshots['cash'].iloc[-1]:,.0f}")
            col3.metric("Drawdown", f"{snapshots['drawdown'].iloc[-1]:.2%}")
            col4.metric("Gross Exposure", f"{snapshots['gross_exposure'].iloc[-1]:.2%}")

            st.subheader("Equity Curve")
            st.line_chart(snapshots.set_index("timestamp")["equity"])
            st.subheader("Drawdown")
            st.area_chart(snapshots.set_index("timestamp")["drawdown"])
            st.subheader("Exposure")
            st.line_chart(snapshots.set_index("timestamp")[["gross_exposure", "net_exposure"]])

            st.subheader("Positions (from fills)")
            orders = repo.get_paper_orders(con)
            fills = repo.get_paper_fills(con)
            if not orders.empty and not fills.empty:
                merged = fills.merge(orders[["order_id", "symbol", "side"]], on="order_id")
                merged["signed_qty"] = np.where(merged["side"] == "BUY", merged["quantity"], -merged["quantity"])
                positions = merged.groupby("symbol")["signed_qty"].sum().reset_index()
                positions = positions[positions["signed_qty"].abs() > 1e-6]
                st.dataframe(positions, use_container_width=True)
            else:
                st.info("No fills recorded yet.")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
with tab_model:
    champion = model_registry.get_champion(con)
    registry_df = repo.get_model_registry(con)
    if champion is None:
        st.info("No champion model registered yet.")
    else:
        st.subheader(f"Champion: {champion['model_version']}")
        metrics = json.loads(champion["metrics_json"])
        st.json(metrics)

        st.subheader("Feature Importance (excess_return_5d model)")
        try:
            boosters, record = model_registry.load_model(con, champion["model_version"])
            if "excess_return_5d" in boosters:
                importances = feature_importance(boosters["excess_return_5d"], record["feature_names"])
                st.bar_chart(importances.set_index("feature").head(20))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load champion artifacts: {exc}")

        st.subheader("Prediction distribution")
        preds = repo.get_predictions(con, model_version=champion["model_version"])
        if not preds.empty:
            st.bar_chart(np.histogram(preds["predicted_excess_return_5d"].dropna(), bins=30)[0])

            st.subheader("Calibration (5-day positive probability)")
            outcomes_df = con.execute(
                "SELECT p.probability_positive_5d, o.realised_excess_return_5d FROM model_predictions p "
                "JOIN outcomes o ON p.prediction_id = o.prediction_id WHERE p.model_version = ?",
                [champion["model_version"]],
            ).fetchdf()
            if not outcomes_df.empty:
                outcomes_df["positive_5d"] = (outcomes_df["realised_excess_return_5d"] > 0).astype(float)
                curve = calibration_curve(outcomes_df["positive_5d"], outcomes_df["probability_positive_5d"])
                st.dataframe(curve, use_container_width=True)
            else:
                st.info("No labelled outcomes yet for this model -- run outcome labelling first.")

    st.subheader("Model Registry")
    st.dataframe(registry_df, use_container_width=True)

    st.subheader("Promotion Log")
    st.dataframe(repo.get_promotion_log(con), use_container_width=True)

# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------
with tab_agents:
    symbols = con.execute("SELECT DISTINCT symbol FROM agent_reports ORDER BY symbol").fetchdf()
    if symbols.empty:
        st.info("No agent reports recorded yet.")
    else:
        symbol = st.selectbox("Symbol", symbols["symbol"].tolist())
        reports_df = con.execute(
            "SELECT * FROM agent_reports WHERE symbol = ? ORDER BY timestamp DESC LIMIT 5", [symbol]
        ).fetchdf()
        st.subheader(f"Latest reports for {symbol}")
        for _, row in reports_df.iterrows():
            with st.expander(f"{row['agent']} @ {row['timestamp']} (confidence={row['confidence']:.2f})"):
                st.json(json.loads(row["features_json"]))
                if row["reasoning_summary"]:
                    st.caption(row["reasoning_summary"])

        st.subheader("Agent disagreement over time")
        feat_matrix = load_feature_matrix(con, DEFAULT_FEATURE_VERSION, symbols=[symbol])
        if not feat_matrix.empty and "agent_disagreement" in feat_matrix.columns:
            chart_df = feat_matrix.set_index("timestamp")[["agent_disagreement", "agent_composite_confidence"]].tail(500)
            st.line_chart(chart_df)

# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
with tab_backtest:
    if run_ids.empty:
        st.info("No backtest runs yet.")
    else:
        run_id_bt = st.selectbox("Backtest run", run_ids["run_id"].tolist(), key="bt_run")
        snaps = repo.get_portfolio_snapshots(con, run_id_bt).sort_values("timestamp")
        if not snaps.empty:
            equity = snaps.set_index("timestamp")["equity"]
            all_symbols = [s for s, _ in synthetic_data.SYMBOLS]
            period_market = get_ohlcv(con, symbols=all_symbols, start=equity.index.min(), end=equity.index.max())
            try:
                bh = buy_and_hold_benchmark(period_market, all_symbols, equity.iloc[0])
                comparison = pd.DataFrame({"strategy": equity, "buy_and_hold": bh}).dropna()
                st.subheader("Strategy vs. Buy & Hold")
                st.line_chart(comparison)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not compute benchmark comparison: {exc}")

            st.subheader("Rolling 63-day Sharpe")
            rets = daily_returns(equity)
            rolling_sharpe = rets.rolling(63).mean() / rets.rolling(63).std() * np.sqrt(252)
            st.line_chart(rolling_sharpe)

# --------------------------------------------------------------------------
# Risk
# --------------------------------------------------------------------------
with tab_risk:
    engaged, reason = repo.get_kill_switch(con)
    st.metric("Kill switch", "ENGAGED" if engaged else "off")
    if engaged:
        st.error(f"Reason: {reason}")

    st.subheader("Rejected orders by reason")
    decisions = repo.get_risk_decisions(con, status="REJECTED")
    if decisions.empty:
        st.info("No rejected orders recorded yet.")
    else:
        reason_counts: dict[str, int] = {}
        for codes_json in decisions["reason_codes_json"]:
            for code in json.loads(codes_json):
                reason_counts[code] = reason_counts.get(code, 0) + 1
        st.bar_chart(pd.Series(reason_counts, name="count"))
        st.dataframe(decisions.tail(200), use_container_width=True)

    st.subheader("Configured limits (defaults used by the demo pipeline)")
    from portfolio.risk import RiskLimits

    st.json(RiskLimits().__dict__)

# --------------------------------------------------------------------------
# Provider Health (V0.2)
# --------------------------------------------------------------------------
with tab_provider_health:
    st.caption("Real-data provider status, derived from the persisted ingestion-run history (data_ingestion_runs).")
    catalog = provider_registry.get_catalog()
    runs = repo_v2.get_ingestion_runs(con)
    rows = []
    for source in catalog:
        source_runs = runs[runs["source_id"] == source.source_id] if not runs.empty else runs
        row = source.model_dump()
        if source_runs is None or source_runs.empty:
            row.update({"last_status": None, "last_run_at": None, "total_runs": 0, "total_records_ingested": 0})
        else:
            latest = source_runs.sort_values("started_at").iloc[-1]
            row.update(
                {
                    "last_status": latest["status"], "last_run_at": latest["started_at"],
                    "total_runs": len(source_runs), "total_records_ingested": int(source_runs["records_ingested"].sum()),
                }
            )
        rows.append(row)
    health_df = pd.DataFrame(rows)
    if health_df.empty:
        st.info("No providers registered.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Enabled providers", int(health_df["is_enabled"].sum()))
        col2.metric("Providers with a successful ingest", int((health_df["last_status"] == "SUCCESS").sum()))
        col3.metric("Total records ingested", int(health_df["total_records_ingested"].sum()))
        st.dataframe(
            health_df[["source_id", "category", "is_enabled", "requires_api_key", "notes", "last_status", "last_run_at", "total_runs", "total_records_ingested"]],
            use_container_width=True,
        )

    st.subheader("Ingestion run history")
    if runs.empty:
        st.info("No ingestion runs recorded yet -- run `uv run python main.py ingest-prices` (etc.) or `real-demo` first.")
    else:
        st.dataframe(runs.sort_values("started_at", ascending=False).head(200), use_container_width=True)

# --------------------------------------------------------------------------
# Data Quality (V0.2)
# --------------------------------------------------------------------------
with tab_data_quality:
    st.subheader("Data quality flags")
    flags = repo_v2.get_quality_flags(con)
    if flags.empty:
        st.info("No data quality flags recorded yet.")
    else:
        severity_counts = flags["severity"].value_counts()
        st.bar_chart(severity_counts)
        st.dataframe(flags.sort_values("created_at", ascending=False).head(300), use_container_width=True)

    st.subheader("Price reconciliation status (primary vs. secondary source)")
    reconciliations = repo_v2.get_price_reconciliations(con)
    if reconciliations.empty:
        st.info("No price reconciliation records yet -- run `uv run python main.py ingest-prices` first.")
    else:
        status_counts = reconciliations["status"].value_counts()
        st.bar_chart(status_counts)
        st.dataframe(reconciliations.tail(300), use_container_width=True)

    st.subheader("Final holdout access audit trail")
    st.caption("Every recorded access to the final, untouched holdout evaluation period -- should be sparse and deliberate.")
    holdout_log = repo_v2.get_holdout_access_log(con)
    if holdout_log.empty:
        st.info("The holdout period has never been accessed in this database -- as expected before a final evaluation.")
    else:
        st.dataframe(holdout_log, use_container_width=True)

# --------------------------------------------------------------------------
# Robustness (V0.2)
# --------------------------------------------------------------------------
with tab_robustness:
    st.caption("Ablation, bootstrap-CI, permutation-test, and factor-exposure reports written by backtesting/robustness.py.")
    model_versions = con.execute(
        "SELECT DISTINCT model_version FROM model_evaluations ORDER BY model_version"
    ).fetchdf()
    if model_versions.empty:
        st.info("No robustness-suite results recorded yet -- run `uv run python main.py evaluate-real` first.")
    else:
        selected_version = st.selectbox("Model version", model_versions["model_version"].tolist())
        evaluations = repo_v2.get_model_evaluations(con, selected_version)
        if evaluations.empty:
            st.info("No robustness results for this model version.")
        else:
            for eval_type in sorted(evaluations["evaluation_type"].unique()):
                with st.expander(eval_type):
                    for _, row in evaluations[evaluations["evaluation_type"] == eval_type].iterrows():
                        st.json(json.loads(row["payload_json"]))

    st.subheader("Champion/challenger promotion log")
    st.caption("Includes V0.2's stricter initial-qualification gate (learning/champion_challenger_v2.py) for entries with no prior champion.")
    st.dataframe(repo.get_promotion_log(con), use_container_width=True)
