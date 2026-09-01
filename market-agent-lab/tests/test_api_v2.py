"""Tests for the V0.2 read-only API additions (Stage 16), mounted under
``/v2`` in api/main.py alongside V0.1's untouched top-level endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from core.schemas_v2 import (
    DataIngestionRun,
    DataQualityFlag,
    HoldoutAccessLog,
    IngestionStatus,
    ProviderCategory,
    QualitySeverity,
)
from data.universe import seed_universe_membership
from database import repository_v2 as repo_v2
from database.db import get_connection, reset_connection


@pytest.fixture
def client(tmp_path):
    reset_connection()
    con = get_connection(tmp_path / "test_api_v2.duckdb")
    yield TestClient(_app()), con
    reset_connection()


def _app():
    import api.main as api_main

    return api_main.app


def test_provider_catalog_returns_known_sources(client):
    test_client, _con = client
    resp = test_client.get("/v2/providers/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) > 0
    assert any(s["source_id"] == "polymarket_readonly" for s in body)


def test_provider_health_reflects_persisted_ingestion_runs(client):
    test_client, con = client
    run = DataIngestionRun(
        source_id="sec_edgar", category=ProviderCategory.FUNDAMENTAL, started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC), status=IngestionStatus.SUCCESS, records_ingested=42,
    )
    repo_v2.insert_ingestion_run(con, run)

    resp = test_client.get("/v2/providers/health")
    assert resp.status_code == 200
    body = resp.json()
    sec_edgar = next(s for s in body if s["source_id"] == "sec_edgar")
    assert sec_edgar["last_status"] == "SUCCESS"
    assert sec_edgar["total_records_ingested"] == 42
    assert sec_edgar["total_runs"] == 1

    # A source with no ingestion history yet reports zero, not a crash.
    never_run = next(s for s in body if s["total_runs"] == 0)
    assert never_run["last_status"] is None


def test_ingestion_runs_endpoint_filters_by_source(client):
    test_client, con = client
    repo_v2.insert_ingestion_run(
        con, DataIngestionRun(source_id="fred", category=ProviderCategory.MACRO, started_at=datetime.now(UTC), status=IngestionStatus.SUCCESS)
    )
    repo_v2.insert_ingestion_run(
        con, DataIngestionRun(source_id="bls", category=ProviderCategory.MACRO, started_at=datetime.now(UTC), status=IngestionStatus.FAILED)
    )
    resp = test_client.get("/v2/providers/ingestion-runs", params={"source_id": "fred"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["source_id"] == "fred"


def test_data_quality_flags_endpoint(client):
    test_client, con = client
    repo_v2.insert_quality_flags(
        con,
        [
            DataQualityFlag(
                category="price", entity_ref="AAPL", flag_type="major_price_difference",
                severity=QualitySeverity.WARNING, created_at=datetime.now(UTC),
            )
        ],
    )
    resp = test_client.get("/v2/data-quality/flags", params={"severity": "WARNING"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["entity_ref"] == "AAPL"


def test_holdout_access_log_endpoint_empty_and_populated(client):
    test_client, con = client
    resp = test_client.get("/v2/holdout/access-log")
    assert resp.status_code == 200
    assert resp.json() == []

    repo_v2.insert_holdout_access_log(
        con,
        HoldoutAccessLog(
            accessed_at=datetime.now(UTC), purpose="final report", model_version="m1",
            holdout_start=datetime(2024, 7, 1), holdout_end=datetime(2025, 6, 30), n_rows=100, symbols=["AAPL"],
        ),
    )
    resp = test_client.get("/v2/holdout/access-log")
    body = resp.json()
    assert len(body) == 1
    assert body[0]["model_version"] == "m1"


def test_robustness_evaluations_endpoint_decodes_payload(client):
    test_client, con = client
    repo_v2.insert_model_evaluation(con, "m1", "bootstrap_ci", {"point_estimate": 0.05, "ci_low": 0.01, "ci_high": 0.09})
    resp = test_client.get("/v2/robustness/evaluations", params={"model_version": "m1"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["payload"]["point_estimate"] == pytest.approx(0.05)
    assert "payload_json" not in body[0]


def test_universe_membership_endpoint_is_point_in_time(client):
    test_client, con = client
    seed_universe_membership(con, "test_universe", ["AAPL", "MSFT"], datetime(2023, 1, 1))
    resp = test_client.get("/v2/universe/test_universe", params={"as_of": "2023-06-01T00:00:00"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"AAPL", "MSFT"}

    resp_before = test_client.get("/v2/universe/test_universe", params={"as_of": "2022-01-01T00:00:00"})
    assert resp_before.json() == []


def test_v01_endpoints_still_work_unchanged(client):
    """The V0.2 router must not break V0.1's top-level endpoints."""
    test_client, _con = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["paper_trading_only"] is True

    resp = test_client.get("/model/registry")
    assert resp.status_code == 200
    assert resp.json() == []
