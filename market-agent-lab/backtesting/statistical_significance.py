"""V0.3 Stage 8: statistical significance and uncertainty reporting.

``backtesting/robustness.py::permutation_test_ic`` was fixed in place (not
duplicated here) to report ``p = (k + 1) / (N + 1)`` instead of the naive
``k / N``, which could claim a literal p=0.0 from a finite number of
permutations even though the true p-value is only known to be smaller
than ``1 / (N + 1)``.

This module adds what else V0.3 Stage 8 requires: the IC information
ratio, an effective-sample-size discussion for overlapping multi-day
targets (so a pooled significance test never silently assumes far more
independent observations than genuinely exist), and the probabilistic /
deflated Sharpe ratio (Bailey & Lopez de Prado, 2012) for reporting
Sharpe uncertainty and multiple-testing-adjusted significance without
overstating confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _sample_kurtosis
from scipy.stats import norm
from scipy.stats import skew as _sample_skew


def ic_information_ratio(per_date_ic: pd.Series) -> dict:
    """The IC "Sharpe ratio": mean IC over time divided by its own
    volatility across dates. A high mean IC with high date-to-date
    volatility is a much weaker basis for confidence than the same mean
    IC with low volatility -- this ratio makes that distinction explicit
    rather than reporting the mean IC alone."""
    clean = pd.Series(per_date_ic).dropna()
    if len(clean) < 3:
        return {"mean_ic": float("nan"), "std_ic": float("nan"), "information_ratio": float("nan"), "n_dates": len(clean)}
    mean_ic = float(clean.mean())
    std_ic = float(clean.std(ddof=1))
    ir = float(mean_ic / std_ic) if std_ic > 0 else float("nan")
    return {"mean_ic": mean_ic, "std_ic": std_ic, "information_ratio": ir, "n_dates": len(clean)}


def effective_sample_size_for_overlap(n_raw_observations: int, horizon_days: int, trading_days_per_observation: float = 1.0) -> dict:
    """A pooled significance test over (symbol, date) rows built from an
    H-trading-day forward target is NOT n_raw_observations independent
    draws: consecutive dates' target windows overlap by up to
    ``horizon_days - 1`` days, so the number of genuinely independent
    (non-overlapping) observations is much smaller. This reports both the
    raw count and a conservative effective-N estimate
    (``n_raw / horizon_days``, in units of the sampling cadence) so a
    downstream significance claim can be checked against how much real
    independent information actually backs it -- this is a reporting aid,
    not a replacement for the block bootstrap (which already resamples in
    contiguous blocks to respect this dependence structure directly)."""
    if horizon_days <= 0 or n_raw_observations <= 0:
        return {"n_raw_observations": max(0, n_raw_observations), "effective_n": 0, "overlap_ratio": float("nan")}
    non_overlapping_period = max(1.0, horizon_days / max(trading_days_per_observation, 1e-9))
    effective_n = max(1, int(n_raw_observations / non_overlapping_period))
    return {
        "n_raw_observations": n_raw_observations,
        "effective_n": effective_n,
        "overlap_ratio": float(effective_n / n_raw_observations),
        "note": (
            f"{n_raw_observations} pooled rows built from a {horizon_days}-trading-day forward target are "
            f"conservatively treated as approximately {effective_n} independent observations, not "
            f"{n_raw_observations} -- significance tests on the raw pooled count overstate confidence by "
            f"roughly sqrt({non_overlapping_period:.1f})x on the standard-error scale."
        ),
    }


def probabilistic_sharpe_ratio(observed_sharpe: float, benchmark_sharpe: float, n_observations: int, returns: pd.Series | np.ndarray) -> float:
    """Bailey & Lopez de Prado (2012): the probability that the TRUE
    (population) Sharpe ratio exceeds ``benchmark_sharpe``, accounting for
    the sample's own skewness and kurtosis (a non-normal return
    distribution makes the naive Sharpe estimate noisier than a normal-
    returns assumption would suggest). Sharpe values here are in the same
    (per-period, not annualised) units as ``returns``."""
    clean = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(clean)
    if n_observations < 5 or n < 5 or observed_sharpe != observed_sharpe:
        return float("nan")
    gamma3 = float(_sample_skew(clean))
    gamma4 = float(_sample_kurtosis(clean, fisher=False))  # regular (non-excess) kurtosis, normal = 3
    denom = 1.0 - gamma3 * observed_sharpe + (gamma4 - 1.0) / 4.0 * observed_sharpe**2
    if denom <= 0:
        return float("nan")
    z = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_observations - 1) / np.sqrt(denom)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float, sharpe_estimates_across_trials: list[float], n_observations: int, returns: pd.Series | np.ndarray
) -> dict:
    """Bailey & Lopez de Prado (2012): PSR evaluated against the EXPECTED
    MAXIMUM Sharpe ratio one would see from ``len(sharpe_estimates_across_trials)``
    independent trials under the null of no real skill -- i.e., it
    corrects for having tried multiple models/features/hyperparameters
    and reporting only the best one. Pass every Sharpe estimate that was
    actually computed during model selection (every fold, every ablation
    variant, every hyperparameter candidate -- whatever was compared),
    not just the winner."""
    trials = [s for s in sharpe_estimates_across_trials if s == s]
    n_trials = len(trials)
    if n_trials < 2 or n_observations < 5:
        return {"deflated_sharpe_ratio": float("nan"), "expected_max_sharpe_under_null": float("nan"), "n_trials": n_trials}
    var_sharpe = float(np.var(trials, ddof=1))
    euler_mascheroni = 0.5772156649015329
    if var_sharpe <= 0:
        expected_max_sharpe = 0.0
    else:
        expected_max_sharpe = float(
            np.sqrt(var_sharpe)
            * ((1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials) + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e)))
        )
    dsr = probabilistic_sharpe_ratio(observed_sharpe, expected_max_sharpe, n_observations, returns)
    return {"deflated_sharpe_ratio": dsr, "expected_max_sharpe_under_null": expected_max_sharpe, "n_trials": n_trials}
