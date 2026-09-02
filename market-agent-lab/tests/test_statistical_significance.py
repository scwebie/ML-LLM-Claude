"""Tests for the V0.3 Stage 8 statistical-significance fixes: the
permutation p-value formula fix in backtesting/robustness.py, and the new
backtesting/statistical_significance.py helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.robustness import permutation_test_ic
from backtesting.statistical_significance import (
    deflated_sharpe_ratio,
    effective_sample_size_for_overlap,
    ic_information_ratio,
    probabilistic_sharpe_ratio,
)

# --- permutation_test_ic p-value fix -------------------------------------------------------------


def test_permutation_test_ic_never_reports_a_literal_zero_p_value():
    """Even for a perfectly correlated signal (every permutation is less
    extreme than observed, so k=0), the reported p-value must be
    1/(N+1), never a literal 0.0 -- a permutation test can never honestly
    claim p=0 from a finite null sample."""
    n = 200
    y_true = pd.Series(np.arange(n, dtype=float))
    y_pred = y_true.copy()  # perfect correlation
    n_permutations = 50
    result = permutation_test_ic(y_true, y_pred, n_permutations=n_permutations, seed=1)
    assert result["p_value"] > 0.0
    assert result["p_value"] == pytest.approx(1.0 / (n_permutations + 1))


def test_permutation_test_ic_p_value_matches_k_plus_1_over_n_plus_1_formula():
    """Directly reproduce the null distribution with the same seed and
    confirm the reported p-value matches (k+1)/(N+1) exactly, not the
    naive k/N."""
    rng_seed = 3
    n_permutations = 100
    y_true = pd.Series(np.random.default_rng(0).normal(size=150))
    y_pred = pd.Series(np.random.default_rng(1).normal(size=150))
    result = permutation_test_ic(y_true, y_pred, n_permutations=n_permutations, seed=rng_seed)

    # Recompute k independently using the exact same permutation procedure.
    from models.evaluate import information_coefficient

    observed = information_coefficient(y_true, y_pred)
    rng = np.random.default_rng(rng_seed)
    null_ics = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_true.to_numpy())
        null_ics.append(information_coefficient(pd.Series(shuffled), y_pred))
    k = int(np.sum(np.abs(np.array(null_ics)) >= abs(observed)))
    expected_p = (k + 1) / (n_permutations + 1)
    assert result["p_value"] == pytest.approx(expected_p)


# --- ic_information_ratio -------------------------------------------------------------------------


def test_ic_information_ratio_higher_for_stable_signal_than_volatile_signal():
    stable = pd.Series([0.05, 0.06, 0.04, 0.05, 0.055, 0.045, 0.05] * 5)
    volatile = pd.Series([0.3, -0.2, 0.4, -0.35, 0.25, -0.3, 0.35] * 5)
    stable_report = ic_information_ratio(stable)
    volatile_report = ic_information_ratio(volatile)
    assert stable_report["information_ratio"] > volatile_report["information_ratio"]


def test_ic_information_ratio_too_few_observations_is_nan():
    result = ic_information_ratio(pd.Series([0.1, 0.2]))
    assert result["information_ratio"] != result["information_ratio"]


# --- effective sample size for overlap -------------------------------------------------------------


def test_effective_sample_size_shrinks_with_larger_horizon():
    short_horizon = effective_sample_size_for_overlap(1000, horizon_days=5)
    long_horizon = effective_sample_size_for_overlap(1000, horizon_days=20)
    assert long_horizon["effective_n"] < short_horizon["effective_n"]
    assert long_horizon["effective_n"] == pytest.approx(50, abs=5)
    assert "note" in long_horizon


def test_effective_sample_size_zero_observations():
    result = effective_sample_size_for_overlap(0, horizon_days=20)
    assert result["n_raw_observations"] == 0


# --- probabilistic / deflated Sharpe ratio -----------------------------------------------------------


def test_probabilistic_sharpe_ratio_high_for_strong_consistent_positive_returns():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.002, 0.005, 500))  # consistently positive, low vol -> high Sharpe
    from backtesting.metrics import sharpe_ratio

    sr = sharpe_ratio(returns, periods_per_year=1)  # per-period Sharpe, not annualized
    psr = probabilistic_sharpe_ratio(sr, 0.0, len(returns), returns)
    assert psr > 0.95


def test_probabilistic_sharpe_ratio_low_for_negative_or_noisy_returns():
    rng = np.random.default_rng(6)
    # A clearly negative drift relative to volatility -- unlike a drift
    # only 1/20th of the volatility, this reliably keeps the REALISED
    # sample Sharpe negative regardless of sampling noise.
    returns = pd.Series(rng.normal(-0.01, 0.02, 200))
    from backtesting.metrics import sharpe_ratio

    sr = sharpe_ratio(returns, periods_per_year=1)
    assert sr < 0  # sanity check on the fixture itself
    psr = probabilistic_sharpe_ratio(sr, 0.0, len(returns), returns)
    assert psr < 0.5


def test_deflated_sharpe_ratio_penalizes_more_trials():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.001, 0.01, 400))
    from backtesting.metrics import sharpe_ratio

    sr = sharpe_ratio(returns, periods_per_year=1)
    few_trials = deflated_sharpe_ratio(sr, [sr * 0.5, sr * 0.7, sr], len(returns), returns)
    many_trials = deflated_sharpe_ratio(sr, [sr * (0.3 + 0.05 * i) for i in range(50)], len(returns), returns)
    assert many_trials["expected_max_sharpe_under_null"] >= few_trials["expected_max_sharpe_under_null"]
    assert many_trials["deflated_sharpe_ratio"] <= few_trials["deflated_sharpe_ratio"]


def test_deflated_sharpe_ratio_too_few_trials_is_nan():
    result = deflated_sharpe_ratio(1.0, [1.0], 100, pd.Series(np.random.default_rng(0).normal(size=100)))
    assert result["deflated_sharpe_ratio"] != result["deflated_sharpe_ratio"]
