"""Tests for Stage 14's V0.2 learning additions: extended drift monitoring
(KS test + Wasserstein distance, alongside V0.1's PSI) and the stricter
initial-champion qualification bar + its champion/challenger wiring.
None of this touches learning/champion_challenger.py or learning/drift.py
(V0.1) -- tests/test_learning.py's existing behaviour is unaffected."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from database.db import fresh_connection
from database.schema import init_schema
from learning.champion_challenger import PromotionCriteria
from learning.champion_challenger_v2 import decide_promotion_v2, run_promotion_cycle_v2
from learning.drift_v2 import detect_feature_drift_full, ks_test, wasserstein
from learning.initial_qualification import InitialQualificationBar, evaluate_initial_qualification
from models.registry import ModelPeriods, register_model
from models.train import train_all_targets

# --- drift_v2: KS test ----------------------------------------------------------------------


def test_ks_test_high_p_value_for_identical_distributions():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(0, 1, 500))
    stat, p_value = ks_test(ref, cur)
    assert p_value > 0.05
    assert 0.0 <= stat <= 1.0


def test_ks_test_low_p_value_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(3, 1, 500))
    stat, p_value = ks_test(ref, cur)
    assert p_value < 0.01
    assert stat > 0.5


def test_ks_test_too_few_observations_returns_neutral_result():
    stat, p_value = ks_test(pd.Series([1.0]), pd.Series([2.0]))
    assert stat == 0.0
    assert p_value == 1.0


# --- drift_v2: Wasserstein distance ----------------------------------------------------------


def test_wasserstein_near_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(0, 1, 1000))
    assert wasserstein(ref, cur) < 0.2


def test_wasserstein_tracks_shift_magnitude():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(5, 1, 1000))
    assert wasserstein(ref, cur) == pytest.approx(5.0, abs=0.3)


def test_wasserstein_empty_series_returns_zero():
    assert wasserstein(pd.Series([], dtype=float), pd.Series([1.0])) == 0.0


# --- drift_v2: detect_feature_drift_full ------------------------------------------------------


def test_detect_feature_drift_full_flags_only_the_shifted_feature():
    rng = np.random.default_rng(2)
    reference_df = pd.DataFrame({"stable": rng.normal(0, 1, 500), "drifted": rng.normal(0, 1, 500)})
    current_df = pd.DataFrame({"stable": rng.normal(0, 1, 500), "drifted": rng.normal(4, 1, 500)})
    results = detect_feature_drift_full(reference_df, current_df, ["stable", "drifted"])
    by_feature = {r.feature: r for r in results}
    assert by_feature["stable"].flagged is False
    assert by_feature["drifted"].flagged is True
    assert by_feature["drifted"].reasons  # non-empty, explains why


def test_detect_feature_drift_full_skips_columns_missing_from_either_frame():
    reference_df = pd.DataFrame({"only_in_reference": [1.0, 2.0, 3.0]})
    current_df = pd.DataFrame({"only_in_current": [1.0, 2.0, 3.0]})
    results = detect_feature_drift_full(reference_df, current_df, ["only_in_reference", "only_in_current", "in_neither"])
    assert results == []


# --- initial_qualification --------------------------------------------------------------------


def _strong_predictions_df(n: int = 600, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.normal(0, 0.02, n)
    pred = target * 0.8 + rng.normal(0, 0.005, n)  # strongly, genuinely correlated
    return pd.DataFrame({"excess_return_20d": target, "predicted_excess_return_20d": pred})


def _weak_predictions_df(n: int = 600, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.normal(0, 0.02, n)
    pred = rng.normal(0, 0.02, n)  # unrelated
    return pd.DataFrame({"excess_return_20d": target, "predicted_excess_return_20d": pred})


def test_evaluate_initial_qualification_passes_strong_challenger():
    df = _strong_predictions_df()
    metrics = {"excess_return_20d": {"information_coefficient": 0.6, "sharpe_ratio": 1.2}}
    result = evaluate_initial_qualification(metrics, df)
    assert result.qualified is True
    assert result.reasons == []


def test_evaluate_initial_qualification_fails_for_insufficient_observations():
    df = _strong_predictions_df(n=50)
    metrics = {"excess_return_20d": {"information_coefficient": 0.6, "sharpe_ratio": 1.2}}
    result = evaluate_initial_qualification(metrics, df, bar=InitialQualificationBar(min_out_of_sample_observations=500))
    assert result.qualified is False
    assert any("out-of-sample observations" in r for r in result.reasons)


def test_evaluate_initial_qualification_fails_for_low_information_coefficient():
    df = _strong_predictions_df()
    metrics = {"excess_return_20d": {"information_coefficient": 0.001, "sharpe_ratio": 1.2}}
    result = evaluate_initial_qualification(metrics, df)
    assert result.qualified is False
    assert any("information_coefficient" in r for r in result.reasons)


def test_evaluate_initial_qualification_fails_when_sharpe_missing():
    df = _strong_predictions_df()
    metrics = {"excess_return_20d": {"information_coefficient": 0.6}}  # no sharpe_ratio key
    result = evaluate_initial_qualification(metrics, df)
    assert result.qualified is False
    assert any("sharpe_ratio" in r for r in result.reasons)


def test_evaluate_initial_qualification_fails_permutation_test_for_unrelated_signal():
    df = _weak_predictions_df()
    # Metrics look good on paper (e.g. computed on a lucky slice), but the
    # full prediction stream shows no real relationship -- the permutation
    # test must catch this even when the reported IC/Sharpe look fine.
    metrics = {"excess_return_20d": {"information_coefficient": 0.6, "sharpe_ratio": 1.2}}
    result = evaluate_initial_qualification(metrics, df, bar=InitialQualificationBar(n_permutations=200))
    assert result.qualified is False
    assert any("permutation" in r for r in result.reasons)


def test_evaluate_initial_qualification_reports_every_failure_not_just_the_first():
    df = _weak_predictions_df(n=10)
    metrics = {}  # no metrics at all
    result = evaluate_initial_qualification(metrics, df, bar=InitialQualificationBar(min_out_of_sample_observations=500))
    assert result.qualified is False
    assert len(result.reasons) >= 3  # observations, IC, sharpe (permutation skipped: n>=3 but weak df is fine)


# --- champion_challenger_v2 ---------------------------------------------------------------------


def test_decide_promotion_v2_rejects_weak_first_challenger_no_champion():
    df = _weak_predictions_df()
    metrics = {"excess_return_20d": {"information_coefficient": 0.001, "sharpe_ratio": 0.01}}
    promoted, rationale = decide_promotion_v2(metrics, None, df)
    assert promoted is False
    assert "failed the initial-qualification bar" in rationale


def test_decide_promotion_v2_promotes_strong_first_challenger_no_champion():
    df = _strong_predictions_df()
    metrics = {"excess_return_20d": {"information_coefficient": 0.6, "sharpe_ratio": 1.2}}
    promoted, rationale = decide_promotion_v2(metrics, None, df)
    assert promoted is True
    assert "passed every initial-qualification criterion" in rationale


def test_decide_promotion_v2_delegates_to_v01_gate_when_champion_exists():
    """With an existing champion, v2 must behave exactly like V0.1's
    decide_promotion (the ongoing comparison, not the one-off bar)."""
    challenger = {
        "excess_return_5d": {"information_coefficient": 0.10, "sharpe_ratio": 2.0},
        "positive_5d": {"brier_score": 0.20},
        "backtest": {"max_drawdown": -0.40, "total_return": 0.80},
    }
    champion = {
        "excess_return_5d": {"information_coefficient": 0.06, "sharpe_ratio": 1.0},
        "positive_5d": {"brier_score": 0.20},
        "backtest": {"max_drawdown": -0.10, "total_return": 0.10},
    }
    promoted, rationale = decide_promotion_v2(
        challenger, champion, pd.DataFrame(), target_col="excess_return_5d",
        criteria=PromotionCriteria(max_drawdown_regression_tolerance=0.05),
    )
    assert promoted is False
    assert "drawdown" in rationale


# --- run_promotion_cycle_v2 (full integration, real trained models) -----------------------------


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        init_schema(c)
        yield c


def _synthetic_feature_target_frame(n_days: int = 700, seed: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "symbol": "SYN_X", "timestamp": dates, "f1": rng.normal(0, 1, n_days), "f2": rng.normal(0, 1, n_days),
            "excess_return_5d": rng.normal(0, 0.02, n_days), "excess_return_20d": rng.normal(0, 0.04, n_days),
            "positive_5d": rng.integers(0, 2, n_days).astype(float), "positive_20d": rng.integers(0, 2, n_days).astype(float),
        }
    )


def test_run_promotion_cycle_v2_rejects_weak_initial_model_and_logs_it(con):
    df = _synthetic_feature_target_frame()
    feature_cols = ["f1", "f2"]
    trained = train_all_targets(df.iloc[:500], df.iloc[500:600], feature_cols)
    periods = ModelPeriods(
        training_start=df["timestamp"].iloc[0], training_end=df["timestamp"].iloc[499],
        validation_start=df["timestamp"].iloc[500], validation_end=df["timestamp"].iloc[599],
    )
    # Purely random synthetic data -> genuinely no real skill; IC should be near zero.
    metrics = {"excess_return_20d": {"information_coefficient": 0.001, "sharpe_ratio": 0.01}}
    model_version = register_model(con, trained, "fv1", periods, metrics, role="CHALLENGER")

    weak_predictions = _weak_predictions_df()
    promoted, rationale = run_promotion_cycle_v2(con, model_version, metrics, weak_predictions)
    assert promoted is False

    log = con.execute("SELECT * FROM promotion_log").fetchdf()
    assert len(log) == 1
    assert log.iloc[0]["decision"] == "REJECTED"
    assert log.iloc[0]["challenger_version"] == model_version

    from database import repository as repo

    assert repo.get_champion(con) is None  # rejection must not install a champion


def test_run_promotion_cycle_v2_promotes_strong_initial_model_and_becomes_champion(con):
    df = _synthetic_feature_target_frame()
    feature_cols = ["f1", "f2"]
    trained = train_all_targets(df.iloc[:500], df.iloc[500:600], feature_cols)
    periods = ModelPeriods(
        training_start=df["timestamp"].iloc[0], training_end=df["timestamp"].iloc[499],
        validation_start=df["timestamp"].iloc[500], validation_end=df["timestamp"].iloc[599],
    )
    metrics = {"excess_return_20d": {"information_coefficient": 0.6, "sharpe_ratio": 1.2}}
    model_version = register_model(con, trained, "fv1", periods, metrics, role="CHALLENGER")

    strong_predictions = _strong_predictions_df()
    promoted, rationale = run_promotion_cycle_v2(con, model_version, metrics, strong_predictions)
    assert promoted is True

    from database import repository as repo

    champion = repo.get_champion(con)
    assert champion is not None
    assert champion["model_version"] == model_version
