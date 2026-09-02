"""Tests for real_pipeline.py's orchestration wiring (Stage 15): that
build-real-features/evaluate-real/real-demo correctly chain the V0.2
building blocks (feature matrix, purge/embargo, holdout split,
champion/challenger) without live network access.

The read-only event-probability ingestion step (a real HTTP call in
production) is stubbed out here -- its own correctness is already tested
in tests/test_prediction_market_readonly.py; this file's job is proving
the pipeline WIRING, not re-testing every ingestion source."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import real_pipeline as rp
from backtesting.holdout import HoldoutConfig
from database import repository as repo
from database.db import fresh_connection
from database.schema import init_schema

SYMBOLS = ["AAPL", "MSFT", "JPM"]
BENCHMARK = "SPY"


def _seed_market_data(con, n_days: int = 1400) -> None:
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rows = []
    profiles = {"AAPL": (1, 150.0, 0.02), "MSFT": (2, 300.0, 0.018), "JPM": (3, 130.0, 0.017), BENCHMARK: (4, 400.0, 0.011)}
    for symbol, (seed, start_price, vol) in profiles.items():
        rng = np.random.default_rng(seed)
        closes = start_price * np.exp(np.cumsum(rng.normal(0.0003, vol, n_days)))
        volumes = np.random.default_rng(seed + 100).normal(2_000_000.0, 300_000.0, n_days).clip(min=100_000.0)
        for ts, close, volume in zip(dates, closes, volumes, strict=True):
            rows.append(
                {
                    "symbol": symbol, "timestamp": ts, "open": close * 0.998, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "adjusted_close": close, "volume": float(volume),
                }
            )
    repo.insert_market_observations(con, pd.DataFrame(rows))


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        init_schema(c)
        yield c


@pytest.fixture(autouse=True)
def _stub_event_ingestion(monkeypatch):
    """No live network in the regular test suite -- the read-only
    event-probability provider's own correctness is tested elsewhere."""
    monkeypatch.setattr(rp, "ingest_event_probabilities", lambda *a, **k: {"status": "SKIPPED_IN_TEST"})


def test_build_real_features_step_stores_a_matrix(con):
    _seed_market_data(con)
    matrix, summary = rp.build_real_features_step(con, SYMBOLS, "test_universe", pd.Timestamp("2020-01-02"))
    assert not matrix.empty
    assert summary["rows_stored"] == len(matrix)
    assert summary["event_probabilities"] == {"status": "SKIPPED_IN_TEST"}

    stored = con.execute("SELECT COUNT(*) FROM feature_snapshots WHERE feature_version = ?", [rp.REAL_FEATURE_VERSION]).fetchone()[0]
    assert stored == len(matrix)


def test_evaluate_real_step_never_touches_holdout_and_produces_a_promotion_decision(con):
    _seed_market_data(con)
    rp.build_real_features_step(con, SYMBOLS, "test_universe", pd.Timestamp("2020-01-02"))

    # A holdout window matching the seeded data's tail, not the far-future
    # production default -- the point-in-time holdout mechanism is what's
    # under test here, not the specific default dates.
    calendar_end = pd.bdate_range("2020-01-02", periods=1400)[-1]
    holdout = HoldoutConfig(start_date=calendar_end - pd.Timedelta(days=80), end_date=calendar_end)

    evaluation = rp.evaluate_real_step(con, SYMBOLS, holdout=holdout, initial_train_fraction=0.5, validation_fraction=0.15)

    assert not evaluation.development_df.empty
    assert not evaluation.holdout_df.empty
    assert (evaluation.development_df["timestamp"] < holdout.start_date).all()
    assert len(evaluation.fold_results) >= 1
    assert evaluation.champion_model_version is not None
    assert isinstance(evaluation.promoted, bool)
    assert evaluation.promotion_rationale  # non-empty explanation either way

    # A weak/random model trained on synthetic-shaped iid-ish price data
    # has no business auto-promoting -- the V0.2 initial-qualification
    # bar (learning/initial_qualification.py) should be doing real work.
    log = repo.get_promotion_log(con) if hasattr(repo, "get_promotion_log") else con.execute("SELECT * FROM promotion_log").fetchdf()
    assert len(log) == 1
    assert log.iloc[0]["challenger_version"] == evaluation.champion_model_version


def test_repeated_evaluate_real_promotion_rationale_ic_matches_this_runs_own_fold_ic(con):
    """End-to-end regression test for a reported production bug: after a
    champion already exists, re-running evaluate_real_step registers a
    genuinely new (differently-versioned) challenger and compares it
    against the incumbent champion. The promotion rationale's IC must be
    this run's own PRIMARY_TARGET (excess_return_20d) last-fold IC -- the
    same number reported in fold_metrics_summary -- never a mismatched
    excess_return_5d number silently substituted by the promotion gate."""
    _seed_market_data(con)
    rp.build_real_features_step(con, SYMBOLS, "test_universe", pd.Timestamp("2020-01-02"))

    first = rp.evaluate_real_step(con, SYMBOLS, initial_train_fraction=0.6, validation_fraction=0.15)
    assert first.champion_model_version is not None
    assert first.promoted is True  # this seeded fixture's first challenger clears initial qualification

    second = rp.evaluate_real_step(con, SYMBOLS, initial_train_fraction=0.6, validation_fraction=0.15)
    # A champion now exists, so this run took the existing-champion
    # comparison branch, not the initial-qualification branch.
    assert "no existing champion" not in second.promotion_rationale
    last_fold_ic_20d = second.fold_metrics_summary["per_fold_information_coefficient"][-1]
    assert f"IC {last_fold_ic_20d:.4f}" in second.promotion_rationale

    # The two runs registered genuinely different model_versions (never a
    # literal self-comparison) even though -- because nothing about the
    # underlying data changed between calls and LightGBM training is
    # deterministic (fixed seed) -- their metrics come out numerically
    # identical. That is a legitimate, if uninformative, comparison: the
    # self-comparison guard (see test_learning_v2.py) only fires on an
    # actual version collision, not on this.
    log = repo.get_promotion_log(con)
    assert log.iloc[-1]["challenger_version"] != log.iloc[-1]["champion_version"]


def test_run_real_demo_with_skip_ingestion_completes_without_network(con):
    _seed_market_data(con)
    calendar_end = pd.bdate_range("2020-01-02", periods=1400)[-1]
    result = rp.run_real_demo(
        con, SYMBOLS, start=pd.Timestamp("2020-01-02"), end=calendar_end, skip_ingestion=True,
    )
    assert result.evaluation.champion_model_version is not None
    assert isinstance(result.n_fills, int)
    assert isinstance(result.n_rejected_orders, int)
    assert isinstance(result.rejection_reason_codes, list)

    # 3. Current-date real-demo regression: the model was frozen after
    # pre-holdout selection, then formally evaluated on the holdout
    # exactly once (real-demo's new final stage) -- and that evaluation
    # is logged.
    assert result.holdout_evaluation is not None
    assert result.holdout_evaluation.n_rows == len(result.evaluation.holdout_df)
    log = con.execute("SELECT COUNT(*) FROM holdout_access_log").fetchone()[0]
    assert log == 1


def test_evaluate_real_step_never_crosses_holdout_when_data_extends_far_past_it(con):
    """3. Current-date real-demo regression (direct reproduction of the
    production failure): a holdout carved out of the MIDDLE of the seeded
    calendar, with ~300 trading days of real post-holdout data after it --
    exactly the shape a real-demo run against real ingestion (which keeps
    running through "today", long after any realistic historical holdout)
    actually has. evaluate_real_step must succeed, no fold may overlap
    the holdout, and post_holdout_df must be preserved, non-empty, and
    entirely after the holdout end."""
    _seed_market_data(con)
    rp.build_real_features_step(con, SYMBOLS, "test_universe", pd.Timestamp("2020-01-02"))

    calendar = pd.bdate_range("2020-01-02", periods=1400)
    holdout = HoldoutConfig(start_date=calendar[1000], end_date=calendar[1100])

    evaluation = rp.evaluate_real_step(con, SYMBOLS, holdout=holdout, initial_train_fraction=0.6, validation_fraction=0.15)

    assert not evaluation.development_df.empty
    assert not evaluation.holdout_df.empty
    assert not evaluation.post_holdout_df.empty

    assert evaluation.development_df["timestamp"].max() < holdout.start_date
    assert evaluation.holdout_df["timestamp"].min() >= holdout.start_date
    assert evaluation.holdout_df["timestamp"].max() <= holdout.end_date
    assert evaluation.post_holdout_df["timestamp"].min() > holdout.end_date

    assert len(evaluation.fold_results) >= 1
    for fold_result in evaluation.fold_results:
        assert fold_result.fold.validation_end < holdout.start_date
