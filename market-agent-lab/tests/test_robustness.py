"""Tests for the robustness / ablation / regime evaluation suite (Stage 13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.purged_walk_forward import (
    PurgedFold,
    PurgedFoldResult,
    build_trading_calendar,
    generate_purged_folds,
)
from backtesting.robustness import (
    NEGATIVE_CONTROL_FEATURE_NAME,
    add_negative_control_feature,
    block_bootstrap_ci,
    build_quantile_portfolio_returns,
    calibration_report,
    classify_feature_family,
    cost_stress_test,
    evaluate_by_group,
    execution_delay_stress_test,
    factor_exposure_report,
    feature_importance_stability,
    group_features_by_family,
    metrics_by_year,
    negative_control_report,
    permutation_test_ic,
    rank_ic_report,
    run_feature_ablation,
)
from models.evaluate import feature_importance
from models.train import get_feature_columns, train_all_targets


def _synthetic_feature_target_frame(n_days: int = 500, extra_cols: dict | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(0)
    data = {
        "symbol": "SYN_X",
        "timestamp": dates,
        "f1": rng.normal(0, 1, n_days),
        "f2": rng.normal(0, 1, n_days),
        "excess_return_5d": rng.normal(0, 0.02, n_days),
        "excess_return_20d": rng.normal(0, 0.04, n_days),
        "positive_5d": rng.integers(0, 2, n_days).astype(float),
        "positive_20d": rng.integers(0, 2, n_days).astype(float),
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# --- 1. block_bootstrap_ci ------------------------------------------------------------------


def test_block_bootstrap_ci_constant_series_has_zero_width_interval():
    values = np.full(200, 3.0)
    result = block_bootstrap_ci(values, n_boot=200, block_size=10)
    assert result["point_estimate"] == pytest.approx(3.0)
    assert result["ci_low"] == pytest.approx(3.0)
    assert result["ci_high"] == pytest.approx(3.0)


def test_block_bootstrap_ci_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    values = rng.normal(0.001, 0.02, 300)
    result = block_bootstrap_ci(values, n_boot=300, block_size=15)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert result["n_boot"] == 300


def test_block_bootstrap_ci_stationary_variant_also_produces_valid_interval():
    rng = np.random.default_rng(2)
    values = rng.normal(0.0, 0.02, 300)
    result = block_bootstrap_ci(values, n_boot=200, block_size=15, stationary=True)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_block_bootstrap_ci_too_short_series_returns_nan_interval():
    result = block_bootstrap_ci(np.array([1.0, 2.0]), block_size=20)
    assert result["n_boot"] == 0
    assert result["ci_low"] != result["ci_low"]  # NaN


# --- 2. feature-family classification and ablation -----------------------------------------


@pytest.mark.parametrize(
    "col,expected_family",
    [
        ("eventprob_monetary_policy_probability", "eventprob"),
        ("macro_raw_FRED_DGS10_zscore", "macro_raw"),
        ("macro_volatility_regime_code", "macro_regime"),
        ("momentum_percentile", "cross_sectional"),
        ("fund_raw_revenue", "fundamental"),
        ("breadth_pct_above_sma20", "market_breadth"),
        ("agent_composite_confidence", "agent"),
        ("news_sentiment_placeholder", "news"),
        ("is_earnings_event", "news"),
        ("return_5d", "technical"),
        ("dollar_volume", "technical"),
        ("totally_unrecognised_column", "other"),
    ],
)
def test_classify_feature_family(col, expected_family):
    assert classify_feature_family(col) == expected_family


def test_group_features_by_family_partitions_all_columns():
    cols = ["return_5d", "eventprob_x_probability", "macro_raw_y_level", "mystery_col"]
    families = group_features_by_family(cols)
    assert sum(len(v) for v in families.values()) == len(cols)
    assert "eventprob" in families and families["eventprob"] == ["eventprob_x_probability"]
    assert "other" in families and families["other"] == ["mystery_col"]


def test_run_feature_ablation_includes_the_eventprob_family_and_computes_deltas():
    rng = np.random.default_rng(3)
    df = _synthetic_feature_target_frame(n_days=500, extra_cols={"eventprob_test_probability": rng.normal(0, 1, 500)})
    feature_cols = ["f1", "f2", "eventprob_test_probability"]
    calendar = build_trading_calendar(df["timestamp"])
    folds = generate_purged_folds(calendar, initial_train_days=350, validation_days=100, window_mode="expanding")
    assert len(folds) >= 1
    results = run_feature_ablation(df, folds, feature_cols)
    families_seen = {r.family for r in results}
    assert "eventprob" in families_seen
    for r in results:
        assert isinstance(r.baseline_score, float)
        assert isinstance(r.ablated_score, float)
        if r.baseline_score == r.baseline_score and r.ablated_score == r.ablated_score:
            assert r.delta == pytest.approx(r.baseline_score - r.ablated_score)


# --- 3. permutation_test_ic -----------------------------------------------------------------


def test_permutation_test_ic_significant_for_perfectly_correlated_signal():
    rng = np.random.default_rng(4)
    y_true = rng.normal(0, 1, 200)
    y_pred = y_true + rng.normal(0, 0.01, 200)  # near-perfect correlation
    result = permutation_test_ic(pd.Series(y_true), pd.Series(y_pred), n_permutations=200, seed=1)
    assert result["observed_ic"] > 0.9
    assert result["p_value"] < 0.05


def test_permutation_test_ic_not_significant_for_unrelated_signal():
    rng = np.random.default_rng(5)
    y_true = rng.normal(0, 1, 200)
    y_pred = rng.normal(0, 1, 200)  # independent of y_true
    result = permutation_test_ic(pd.Series(y_true), pd.Series(y_pred), n_permutations=200, seed=1)
    assert result["p_value"] > 0.05


def test_permutation_test_ic_insufficient_data_returns_nan():
    result = permutation_test_ic(pd.Series([1.0]), pd.Series([1.0]))
    assert result["n_permutations"] == 0
    assert result["observed_ic"] != result["observed_ic"]


# --- 4. negative control ---------------------------------------------------------------------


def test_add_negative_control_feature_adds_iid_noise_column():
    df = _synthetic_feature_target_frame(n_days=100)
    out = add_negative_control_feature(df, seed=7)
    assert NEGATIVE_CONTROL_FEATURE_NAME in out.columns
    assert len(out) == len(df)


def test_negative_control_report_flags_low_ranked_control_as_passed():
    importances = pd.DataFrame(
        {"feature": ["f1", "f2", "f3", NEGATIVE_CONTROL_FEATURE_NAME], "importance": [100.0, 80.0, 60.0, 1.0]}
    )
    report = negative_control_report(importances, top_fraction=0.25)
    assert report.control_rank == 4
    assert report.passed is True


def test_negative_control_report_flags_top_ranked_control_as_failed():
    importances = pd.DataFrame(
        {"feature": [NEGATIVE_CONTROL_FEATURE_NAME, "f2", "f3", "f4"], "importance": [100.0, 80.0, 60.0, 1.0]}
    )
    report = negative_control_report(importances, top_fraction=0.25)
    assert report.control_rank == 1
    assert report.passed is False


def test_negative_control_report_raises_if_control_column_missing():
    importances = pd.DataFrame({"feature": ["f1", "f2"], "importance": [10.0, 5.0]})
    with pytest.raises(ValueError, match=NEGATIVE_CONTROL_FEATURE_NAME):
        negative_control_report(importances)


# --- 5. regime / year breakdown ---------------------------------------------------------------


def test_evaluate_by_group_skips_small_groups_and_computes_metrics_for_larger_ones():
    df = pd.DataFrame(
        {
            "grp": ["A"] * 10 + ["B"] * 2,
            "excess_return_5d": list(np.linspace(-0.05, 0.05, 10)) + [0.01, 0.02],
            "pred": list(np.linspace(-0.05, 0.05, 10)) + [0.01, 0.02],
        }
    )
    result = evaluate_by_group(df, "grp", "excess_return_5d", "pred")
    assert list(result["grp"]) == ["A"]  # group B has only 2 rows, skipped
    assert result.iloc[0]["n"] == 10


def test_metrics_by_year_splits_on_calendar_year():
    dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2020-09-01", "2021-01-01", "2021-06-01", "2021-09-01"])
    df = pd.DataFrame(
        {"timestamp": dates, "excess_return_5d": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03], "pred": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03]}
    )
    result = metrics_by_year(df, "excess_return_5d", "pred")
    assert set(result["year"]) == {2020, 2021}


# --- 6. rank IC ------------------------------------------------------------------------------


def test_rank_ic_report_near_one_for_identical_series():
    values = pd.Series(np.linspace(-1, 1, 50))
    df = pd.DataFrame({"excess_return_5d": values, "pred": values})
    result = rank_ic_report(df, "excess_return_5d", "pred")
    assert result["rank_ic"] > 0.99
    assert result["n"] == 50


def test_calibration_report_rejects_regression_target():
    df = pd.DataFrame({"excess_return_5d": [0.01, 0.02], "pred": [0.5, 0.6]})
    with pytest.raises(ValueError, match="classification"):
        calibration_report(df, "excess_return_5d", "pred")


def test_calibration_report_returns_bins_for_classification_target():
    rng = np.random.default_rng(10)
    n = 200
    pred = rng.uniform(0, 1, n)
    target = (rng.uniform(0, 1, n) < pred).astype(float)  # well-calibrated by construction
    df = pd.DataFrame({"positive_5d": target, "pred": pred})
    result = calibration_report(df, "positive_5d", "pred", n_bins=5)
    assert not result.empty
    assert {"bin", "mean_predicted", "mean_realised", "count"}.issubset(result.columns)


# --- 7. transaction-cost stress testing ------------------------------------------------------


def test_build_quantile_portfolio_returns_and_cost_stress_test_edge_shrinks_with_cost():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2022-01-01", periods=60)
    symbols = [f"S{i}" for i in range(10)]
    rows = []
    for ts in dates:
        # pred perfectly ranks the realised return within each date -> strong, real signal
        realised = rng.normal(0, 0.02, len(symbols))
        rows.extend(
            {"timestamp": ts, "symbol": sym, "excess_return_5d": ret, "pred": ret}
            for sym, ret in zip(symbols, realised, strict=True)
        )
    df = pd.DataFrame(rows)
    portfolio = build_quantile_portfolio_returns(df, "excess_return_5d", "pred", top_frac=0.2)
    assert not portfolio.empty
    assert (portfolio["gross_return"] > 0).mean() > 0.8  # signal is real -> mostly positive days

    stress = cost_stress_test(portfolio, cost_bps_grid=(0, 500, 2000))
    sharpes = stress.set_index("cost_bps")["net_sharpe"]
    assert sharpes[0] > sharpes[2000]  # higher cost strictly erodes the edge


# --- 8. execution-delay stress testing --------------------------------------------------------


def test_execution_delay_stress_test_signal_decays_with_delay():
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2022-01-01", periods=200)
    target = rng.normal(0, 1, 200)
    df = pd.DataFrame({"symbol": "SYN_X", "timestamp": dates, "excess_return_5d": target, "pred": target})
    result = execution_delay_stress_test(df, "excess_return_5d", "pred", delays_days=(0, 20))
    ic_by_delay = result.set_index("delay_days")["rank_ic"]
    assert ic_by_delay[0] > 0.99  # delay 0: pred == target exactly
    assert abs(ic_by_delay[20]) < 0.5  # shifting decorrelates an iid-random target series


# --- 9. factor exposure -----------------------------------------------------------------------


def test_factor_exposure_report_recovers_known_linear_relationship():
    rng = np.random.default_rng(9)
    n = 300
    factor1 = rng.normal(0, 1, n)
    factor2 = rng.normal(0, 1, n)
    pred = 2.0 * factor1 - 1.0 * factor2 + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"pred": pred, "momentum_percentile": factor1, "liquidity_percentile": factor2})
    report = factor_exposure_report(df, "pred", ["momentum_percentile", "liquidity_percentile"])
    loadings = report.set_index("factor")["loading"]
    assert loadings["momentum_percentile"] > 0
    assert loadings["liquidity_percentile"] < 0
    assert report["r_squared"].iloc[0] > 0.9


def test_factor_exposure_report_insufficient_rows_returns_empty():
    df = pd.DataFrame({"pred": [1.0, 2.0], "f1": [0.1, 0.2]})
    report = factor_exposure_report(df, "pred", ["f1"])
    assert report.empty


# --- 10. feature importance stability -----------------------------------------------------------


def test_feature_importance_stability_reports_valid_correlation_for_real_folds():
    df = _synthetic_feature_target_frame(n_days=500)
    feature_cols = ["f1", "f2"]
    fold_a = PurgedFold(fold_id=0, train_start=df["timestamp"].min(), validation_start=df["timestamp"].iloc[300], validation_end=df["timestamp"].iloc[349], window_mode="expanding")
    fold_b = PurgedFold(fold_id=1, train_start=df["timestamp"].min(), validation_start=df["timestamp"].iloc[400], validation_end=df["timestamp"].iloc[449], window_mode="expanding")

    trained_a = train_all_targets(df.iloc[:300], df.iloc[300:350], feature_cols)
    trained_b = train_all_targets(df.iloc[:400], df.iloc[400:450], feature_cols)
    results = [
        PurgedFoldResult(fold=fold_a, trained=trained_a, metrics={}, predictions=pd.DataFrame(), n_train_rows=300, n_purged_or_embargoed=0),
        PurgedFoldResult(fold=fold_b, trained=trained_b, metrics={}, predictions=pd.DataFrame(), n_train_rows=400, n_purged_or_embargoed=0),
    ]
    stability = feature_importance_stability(results, "excess_return_20d")
    assert stability["n_folds"] == 2
    assert stability["n_pairs"] == 1
    assert -1.0 <= stability["mean_pairwise_spearman"] <= 1.0


def test_feature_importance_stability_insufficient_folds_returns_nan():
    df = _synthetic_feature_target_frame(n_days=500)
    feature_cols = get_feature_columns(df)
    fold = PurgedFold(fold_id=0, train_start=df["timestamp"].min(), validation_start=df["timestamp"].iloc[300], validation_end=df["timestamp"].iloc[349], window_mode="expanding")
    trained = train_all_targets(df.iloc[:300], df.iloc[300:350], feature_cols)
    results = [PurgedFoldResult(fold=fold, trained=trained, metrics={}, predictions=pd.DataFrame(), n_train_rows=300, n_purged_or_embargoed=0)]
    stability = feature_importance_stability(results, "excess_return_20d")
    assert stability["n_folds"] == 1
    assert stability["mean_pairwise_spearman"] != stability["mean_pairwise_spearman"]  # NaN


def test_feature_importance_helper_still_importable_and_returns_sorted_frame():
    df = _synthetic_feature_target_frame(n_days=500)
    feature_cols = ["f1", "f2"]
    trained = train_all_targets(df.iloc[:400], df.iloc[400:450], feature_cols)
    imp = feature_importance(trained.boosters["excess_return_20d"], feature_cols)
    assert list(imp.columns) == ["feature", "importance"]
    assert imp["importance"].is_monotonic_decreasing
