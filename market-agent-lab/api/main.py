"""FastAPI monitoring / read API for market-agent-lab.

Read-mostly by design: this API exposes what the system has already
decided (predictions, risk decisions, fills, model registry, portfolio
snapshots) for the dashboard and external tooling. It does not expose any
endpoint that places, approves, or overrides a trade -- that authority
lives only in ``portfolio/risk.py`` and ``execution/paper.py``, called
from the backtest/demo pipeline, never from an HTTP request body.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException, Query

from core.config import settings
from database import repository as repo
from database.db import get_connection
from models import registry as model_registry

app = FastAPI(
    title="market-agent-lab API",
    description="Paper-trading-only research system monitoring API (Version 0.1)",
    version="0.1.0",
)


def _con():
    return get_connection()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "paper_trading_only": settings.paper_trading_only}


@app.get("/model/champion")
def champion() -> dict:
    con = _con()
    record = model_registry.get_champion(con)
    if record is None:
        raise HTTPException(status_code=404, detail="no champion model registered yet")
    return record


@app.get("/model/registry")
def registry() -> list[dict]:
    con = _con()
    df = repo.get_model_registry(con)
    return df.to_dict(orient="records")


@app.get("/model/promotions")
def promotions() -> list[dict]:
    con = _con()
    return repo.get_promotion_log(con).to_dict(orient="records")


@app.get("/predictions")
def predictions(
    symbol: str | None = None, model_version: str | None = None, limit: int = Query(100, le=5000)
) -> list[dict]:
    con = _con()
    df = repo.get_predictions(con, model_version=model_version, symbol=symbol)
    return df.sort_values("timestamp").tail(limit).to_dict(orient="records")


@app.get("/risk/decisions")
def risk_decisions(status: str | None = None, limit: int = Query(200, le=5000)) -> list[dict]:
    con = _con()
    df = repo.get_risk_decisions(con, status=status)
    return df.sort_values("timestamp").tail(limit).to_dict(orient="records")


@app.get("/risk/rejected-orders")
def rejected_orders(limit: int = Query(200, le=5000)) -> list[dict]:
    con = _con()
    df = repo.get_paper_orders(con)
    rejected = df[df["risk_approval_status"] == "REJECTED"]
    return rejected.sort_values("timestamp").tail(limit).to_dict(orient="records")


@app.get("/execution/fills")
def fills(limit: int = Query(200, le=5000)) -> list[dict]:
    con = _con()
    df = repo.get_paper_fills(con)
    return df.sort_values("fill_timestamp").tail(limit).to_dict(orient="records")


@app.get("/portfolio/snapshots")
def portfolio_snapshots(run_id: str) -> list[dict]:
    con = _con()
    df = repo.get_portfolio_snapshots(con, run_id)
    return df.to_dict(orient="records")


@app.get("/agents/reports")
def agent_reports(symbol: str, as_of: datetime, agents: str | None = None) -> list[dict]:
    con = _con()
    agent_list = agents.split(",") if agents else None
    reports = repo.get_agent_reports(con, symbol, as_of, agents=agent_list)
    return [r.model_dump() for r in reports]
