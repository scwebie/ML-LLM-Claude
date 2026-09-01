"""V0.2 read-only monitoring endpoints (Stage 16): provider health, data
quality, robustness/evaluation results, and the holdout access audit
trail. Mounted under ``/v2`` in ``api/main.py`` (V0.1's endpoints, all at
the top level, are completely untouched).

Same read-only design rule as V0.1's API: nothing here places, approves,
or overrides anything -- it only exposes what ingestion, evaluation, and
promotion runs have already written to DuckDB.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Query

from data.providers import registry as provider_registry
from database import repository as repo
from database import repository_v2 as repo_v2
from database.db import get_connection

router = APIRouter(prefix="/v2", tags=["v2"])


def _con():
    return get_connection()


def _records(df: pd.DataFrame) -> list[dict]:
    """``DataFrame.to_dict(orient="records")`` leaves missing values as
    ``NaN``/``NaT``, neither of which is valid JSON -- normalise to
    ``None`` so every endpoint returns clean, spec-compliant JSON."""
    if df.empty:
        return []
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


# --------------------------------------------------------------------------
# Provider health / ingestion
# --------------------------------------------------------------------------


@router.get("/providers/catalog")
def provider_catalog() -> list[dict]:
    """The full registered provider catalog (enabled or not), from
    ``data/providers/registry.py`` -- static configuration, not live
    status."""
    return [s.model_dump() for s in provider_registry.get_catalog()]


@router.get("/providers/health")
def provider_health() -> list[dict]:
    """Per-source health derived from the PERSISTED ingestion-run history
    (``data_ingestion_runs``) -- unlike the in-memory
    ``ProviderHealthTracker``, this reflects reality across process
    restarts, which is what a dashboard/API running in its own process
    needs. One row per source: its catalog entry plus its most recent
    ingestion attempt's status/timestamp/error, and rolling success/
    failure counts."""
    con = _con()
    catalog = {s.source_id: s.model_dump() for s in provider_registry.get_catalog()}
    runs = repo_v2.get_ingestion_runs(con)
    rows = []
    for source_id, source in catalog.items():
        source_runs = runs[runs["source_id"] == source_id] if not runs.empty else runs
        row = dict(source)
        if source_runs is None or source_runs.empty:
            row.update({"last_status": None, "last_run_at": None, "last_error": None, "total_runs": 0, "total_records_ingested": 0})
        else:
            sorted_runs = source_runs.sort_values("started_at")
            latest = sorted_runs.iloc[-1]
            row.update(
                {
                    "last_status": latest["status"], "last_run_at": latest["started_at"],
                    "last_error": latest["error_message"], "total_runs": len(sorted_runs),
                    "total_records_ingested": int(sorted_runs["records_ingested"].sum()),
                }
            )
        rows.append(row)
    return rows


@router.get("/providers/ingestion-runs")
def ingestion_runs(source_id: str | None = None, limit: int = Query(200, le=5000)) -> list[dict]:
    con = _con()
    df = repo_v2.get_ingestion_runs(con, source_id=source_id)
    if df.empty:
        return []
    return _records(df.sort_values("started_at").tail(limit))


# --------------------------------------------------------------------------
# Data quality
# --------------------------------------------------------------------------


@router.get("/data-quality/flags")
def data_quality_flags(category: str | None = None, severity: str | None = None, limit: int = Query(500, le=5000)) -> list[dict]:
    con = _con()
    df = repo_v2.get_quality_flags(con, category=category, severity=severity)
    if df.empty:
        return []
    return _records(df.sort_values("created_at").tail(limit))


@router.get("/data-quality/price-reconciliations")
def price_reconciliations(symbol: str | None = None, status: str | None = None, limit: int = Query(500, le=5000)) -> list[dict]:
    con = _con()
    df = repo_v2.get_price_reconciliations(con, symbol=symbol, status=status)
    if df.empty:
        return []
    return _records(df.tail(limit))


@router.get("/data-quality/leakage-audits")
def leakage_audits(run_id: str) -> list[dict]:
    con = _con()
    return _records(repo_v2.get_leakage_audit_results(con, run_id))


# --------------------------------------------------------------------------
# Robustness / evaluation suite outputs
# --------------------------------------------------------------------------


@router.get("/robustness/evaluations")
def robustness_evaluations(model_version: str, evaluation_type: str | None = None) -> list[dict]:
    """Rows written by ``backtesting/robustness.py``'s reports via
    ``database.repository_v2.insert_model_evaluation`` (ablation results,
    bootstrap CIs, permutation tests, factor exposure, ...). The payload
    is stored as JSON; this returns it decoded."""
    import json

    con = _con()
    df = repo_v2.get_model_evaluations(con, model_version, evaluation_type)
    records = _records(df)
    for r in records:
        if "payload_json" in r:
            r["payload"] = json.loads(r.pop("payload_json"))
    return records


# --------------------------------------------------------------------------
# Final holdout access audit trail
# --------------------------------------------------------------------------


@router.get("/holdout/access-log")
def holdout_access_log() -> list[dict]:
    """Every recorded access to the final holdout evaluation period
    (``backtesting/holdout.py::evaluate_on_holdout``) -- the audit trail a
    reviewer checks to confirm the holdout was touched only for a final,
    formal evaluation."""
    con = _con()
    return _records(repo_v2.get_holdout_access_log(con))


# --------------------------------------------------------------------------
# Champion/challenger promotion log (V0.2 gate included -- same table as V0.1)
# --------------------------------------------------------------------------


@router.get("/model/promotions")
def promotions_v2(limit: int = Query(200, le=5000)) -> list[dict]:
    """Identical data to V0.1's ``/model/promotions`` -- both V0.1's and
    V0.2's champion/challenger gates write to the same ``promotion_log``
    table. Exposed under ``/v2`` too for discoverability alongside the
    other V0.2 monitoring endpoints."""
    con = _con()
    df = repo.get_promotion_log(con)
    if df.empty:
        return []
    return _records(df.sort_values("timestamp").tail(limit))


# --------------------------------------------------------------------------
# Point-in-time universe membership
# --------------------------------------------------------------------------


@router.get("/universe/{universe_name}")
def universe_membership(universe_name: str, as_of: str | None = None) -> list[str]:
    """The point-in-time universe membership for ``universe_name`` as of
    ``as_of`` (ISO date; default: now) -- see ``data/universe.py``. This
    universe is survivorship-biased (see ``SURVIVORSHIP_BIAS_WARNING``);
    that caveat is documented, not hidden, in ``docs/point_in_time_data.md``."""
    from datetime import UTC, datetime

    con = _con()
    as_of_dt = datetime.fromisoformat(as_of) if as_of else datetime.now(UTC)
    return repo_v2.get_point_in_time_universe(con, universe_name, as_of_dt)
