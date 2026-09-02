"""Robustness / ablation / regime evaluation suite (Stage 13, V0.2).

Every function here operates on genuine out-of-sample predictions --
either the concatenated ``PurgedFoldResult.predictions`` from
``backtesting/purged_walk_forward.py`` (via :func:`build_evaluation_frame`)
or, for the ablation/feature-importance-stability pieces, by re-running
that same purged+embargoed walk-forward machinery with a modified feature
set. Nothing here is ever fit or tuned against the final holdout period
(``backtesting/holdout.py``) -- this suite is strictly a development-set
diagnostic.

Ten pieces, each independently callable:

1. Block/stationary bootstrap confidence intervals for a performance
   statistic on a time-ordered series (``block_bootstrap_ci``).
2. Feature-family ablation, including the read-only event-probability
   family (``run_feature_ablation``, ``classify_feature_family``).
3. A permutation test for a prediction's rank IC (``permutation_test_ic``).
4. A negative-control random feature + report on whether it earns
   suspiciously high importance (``add_negative_control_feature``,
   ``negative_control_report``).
5. Regime-specific and year-by-year performance breakdown
   (``evaluate_by_group``, ``metrics_by_year``).
6. Rank IC / calibration reporting (``rank_ic_report``, reusing
   ``models.evaluate.calibration_curve``).
7. Transaction-cost stress testing on a simple quantile long/short
   portfolio built from the predicted signal (``cost_stress_test``).
8. Execution-delay stress testing -- how fast the signal decays if acted
   on late (``execution_delay_stress_test``).
9. A factor-exposure report regressing the predicted signal on style-
   factor proxies (``factor_exposure_report``).
10. Feature-importance stability across walk-forward folds
    (``feature_importance_stability``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.metrics import sharpe_ratio
from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    TARGET_TO_PRED_COL,
    PurgedFold,
    PurgedFoldResult,
    run_purged_walk_forward,
)
from features.technical import TECHNICAL_FEATURE_COLUMNS
from models.evaluate import (
    calibration_curve,
    evaluate_classification,
    evaluate_regression,
    feature_importance,
    information_coefficient,
)
from models.train import TARGET_KIND

# --------------------------------------------------------------------------
# Evaluation-frame assembly
# --------------------------------------------------------------------------


def build_evaluation_frame(
    fold_results: list[PurgedFoldResult],
    df: pd.DataFrame,
    target_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every fold's out-of-sample predictions and join back to
    ``df`` for the realised target and any extra columns (regime codes,
    factor proxies) the reports below need. Every row here is a genuine
    out-of-sample prediction -- never a training-set fit, since each
    fold's ``predictions`` only ever covers that fold's validation rows."""
    pred_col = TARGET_TO_PRED_COL[target_col]
    extra_cols = extra_cols or []
    frames = [result.predictions[["symbol", "timestamp", pred_col]] for result in fold_results if not result.predictions.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", pred_col, target_col, *extra_cols])
    all_preds = pd.concat(frames, ignore_index=True)
    join_cols = ["symbol", "timestamp", target_col, *extra_cols]
    return all_preds.merge(df[join_cols], on=["symbol", "timestamp"], how="inner")


# --------------------------------------------------------------------------
# 1. Block / stationary bootstrap confidence intervals
# --------------------------------------------------------------------------


def _circular_block_bootstrap_sample(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    pieces = [np.take(values, np.arange(s, s + block_size) % n) for s in starts]
    return np.concatenate(pieces)[:n]


def _stationary_bootstrap_sample(values: np.ndarray, expected_block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap: block lengths are
    themselves random (geometric with mean ``expected_block_size``), so
    unlike a fixed-length block bootstrap the resampled series is itself
    stationary."""
    n = len(values)
    p = 1.0 / expected_block_size
    out = np.empty(n, dtype=values.dtype)
    idx = int(rng.integers(0, n))
    for i in range(n):
        out[i] = values[idx]
        idx = (idx + 1) % n if rng.random() > p else int(rng.integers(0, n))
    return out


def block_bootstrap_ci(
    values: np.ndarray | pd.Series,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    block_size: int = 20,
    ci: float = 0.95,
    seed: int = 42,
    stationary: bool = False,
) -> dict[str, float]:
    """Block (or, with ``stationary=True``, stationary/Politis-Romano)
    bootstrap confidence interval for ``statistic_fn`` applied to a
    time-ordered series (e.g. per-date IC or per-date portfolio return).
    A plain iid bootstrap would understate the CI width for autocorrelated
    financial time series -- resampling contiguous blocks preserves
    short-range serial dependence instead of shuffling it away."""
    clean = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(clean)
    if n < block_size:
        return {
            "point_estimate": float(statistic_fn(clean)) if n else float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0, "n_obs": n,
        }
    rng = np.random.default_rng(seed)
    point_estimate = float(statistic_fn(clean))
    boot_stats = np.empty(n_boot)
    sampler = _stationary_bootstrap_sample if stationary else _circular_block_bootstrap_sample
    for b in range(n_boot):
        boot_stats[b] = statistic_fn(sampler(clean, block_size, rng))
    alpha = 1 - ci
    lo, hi = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    return {"point_estimate": point_estimate, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot, "n_obs": n}


# --------------------------------------------------------------------------
# 2. Feature-family ablation
# --------------------------------------------------------------------------

_FAMILY_PREFIX_RULES: list[tuple[str, str]] = [
    ("eventprob", "eventprob_"),
    ("macro_raw", "macro_raw_"),
    ("news", "news_"),
    ("fundamental", "fund_raw_"),
    ("market_breadth", "breadth_"),
    ("agent", "agent_"),
]
_FAMILY_SUFFIX_RULES: list[tuple[str, str]] = [
    ("macro_regime", "_regime_code"),
    ("cross_sectional", "_percentile"),
]
_FAMILY_EXACT_MEMBERS: dict[str, set[str]] = {"news": {"is_earnings_event", "event_uncertainty"}}


def classify_feature_family(col: str) -> str:
    """Best-effort grouping of a feature column into a named family for
    ablation reporting, by naming convention. Anything unmatched falls
    into ``"other"`` rather than being silently dropped from every
    family's accounting."""
    for family, prefix in _FAMILY_PREFIX_RULES:
        if col.startswith(prefix):
            return family
    for family, suffix in _FAMILY_SUFFIX_RULES:
        if col.endswith(suffix):
            return family
    for family, members in _FAMILY_EXACT_MEMBERS.items():
        if col in members:
            return family
    if col in TECHNICAL_FEATURE_COLUMNS or col == "dollar_volume":
        return "technical"
    return "other"


def group_features_by_family(feature_cols: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for col in feature_cols:
        families.setdefault(classify_feature_family(col), []).append(col)
    return families


def _mean_metric(fold_results: list[PurgedFoldResult], target_col: str, metric: str) -> float:
    values = [r.metrics.get(target_col, {}).get(metric, float("nan")) for r in fold_results]
    values = [v for v in values if v == v]  # noqa: PLR0124 - NaN-safe filter
    return float(np.mean(values)) if values else float("nan")


@dataclass
class AblationResult:
    family: str
    n_features_removed: int
    baseline_score: float
    ablated_score: float
    delta: float  # baseline - ablated; positive means the family HELPED performance


def run_feature_ablation(
    df: pd.DataFrame,
    folds: list[PurgedFold],
    feature_cols: list[str],
    families: dict[str, list[str]] | None = None,
    target_col: str = "excess_return_20d",
    primary_metric: str = "information_coefficient",
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    hyperparameters: dict | None = None,
) -> list[AblationResult]:
    """Baseline: train with every feature. Then, for every family, retrain
    with that family's columns removed and compare mean out-of-fold
    ``primary_metric`` for ``target_col``. Includes the ``eventprob``
    family automatically whenever event-probability features are present
    in ``feature_cols`` -- no separate code path is needed to ablate the
    read-only prediction-market signal."""
    families = families or group_features_by_family(feature_cols)
    baseline_results = run_purged_walk_forward(
        df, folds, feature_cols, horizon_days, embargo_days, hyperparameters, model_version_prefix="ablation_baseline"
    )
    baseline_score = _mean_metric(baseline_results, target_col, primary_metric)

    reports = []
    for family, cols_to_remove in families.items():
        remaining = [c for c in feature_cols if c not in cols_to_remove]
        if not remaining:
            continue
        ablated_results = run_purged_walk_forward(
            df, folds, remaining, horizon_days, embargo_days, hyperparameters, model_version_prefix=f"ablation_{family}"
        )
        ablated_score = _mean_metric(ablated_results, target_col, primary_metric)
        reports.append(
            AblationResult(
                family=family, n_features_removed=len(cols_to_remove),
                baseline_score=baseline_score, ablated_score=ablated_score,
                delta=baseline_score - ablated_score if baseline_score == baseline_score and ablated_score == ablated_score else float("nan"),
            )
        )
    return reports


# --------------------------------------------------------------------------
# 3. Permutation test
# --------------------------------------------------------------------------


def permutation_test_ic(
    y_true: pd.Series, y_pred: pd.Series, n_permutations: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Null hypothesis: the model's predictions carry no real relationship
    to the realised target. Shuffles ``y_true`` (breaking any real
    relationship while preserving both marginal distributions) and
    recomputes the rank IC ``n_permutations`` times.

    The p-value uses the standard finite-permutation correction
    ``p = (k + 1) / (N + 1)`` (Davison & Hinkley 1997; North, Curtis &
    Sham 2002), where ``k`` is the number of permuted statistics at least
    as extreme as the observed one and ``N`` is the permutation count --
    NEVER the naive ``k / N``, which can report a literal p=0.0 with a
    finite number of permutations even though the true p-value is only
    known to be less than ``1 / (N + 1)``. A permutation test can never
    honestly claim p=0 from a finite sample of the null distribution."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    y_true_arr, y_pred_arr = y_true_arr[mask], y_pred_arr[mask]
    if len(y_true_arr) < 3:
        return {"observed_ic": float("nan"), "p_value": float("nan"), "n_permutations": 0, "null_mean": float("nan"), "null_std": float("nan")}

    observed = information_coefficient(pd.Series(y_true_arr), pd.Series(y_pred_arr))
    rng = np.random.default_rng(seed)
    null_ics = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(y_true_arr)
        null_ics[i] = information_coefficient(pd.Series(shuffled), pd.Series(y_pred_arr))
    k = int(np.sum(np.abs(null_ics) >= abs(observed)))
    p_value = float((k + 1) / (n_permutations + 1))
    return {
        "observed_ic": observed, "p_value": p_value, "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_ics)), "null_std": float(np.std(null_ics)),
    }


# --------------------------------------------------------------------------
# 4. Negative-control feature
# --------------------------------------------------------------------------

NEGATIVE_CONTROL_FEATURE_NAME = "negative_control_random"


def add_negative_control_feature(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Returns a copy of ``df`` with one extra column of pure iid noise.
    Include this column's name in ``feature_cols`` when training, then
    call :func:`negative_control_report` on the resulting feature
    importances -- a model correctly ignoring noise should rank it low."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[NEGATIVE_CONTROL_FEATURE_NAME] = rng.normal(0, 1, len(out))
    return out


@dataclass
class NegativeControlReport:
    control_importance: float
    control_rank: int  # 1-based, 1 = most important
    n_features: int
    passed: bool  # True if the control did NOT rank in the top decile


def negative_control_report(
    importances: pd.DataFrame, control_col: str = NEGATIVE_CONTROL_FEATURE_NAME, top_fraction: float = 0.1
) -> NegativeControlReport:
    """``importances``: the output of ``models.evaluate.feature_importance``
    (columns ``feature``, ``importance``, sorted descending)."""
    ranked = importances.reset_index(drop=True)
    if control_col not in ranked["feature"].values:
        raise ValueError(f"{control_col!r} not found in importances -- was it included in feature_cols during training?")
    n = len(ranked)
    rank = int(ranked.index[ranked["feature"] == control_col][0]) + 1
    control_importance = float(ranked.loc[ranked["feature"] == control_col, "importance"].iloc[0])
    passed = rank > max(1, int(np.ceil(n * top_fraction)))
    return NegativeControlReport(control_importance=control_importance, control_rank=rank, n_features=n, passed=passed)


# --------------------------------------------------------------------------
# 5. Regime-specific / year-by-year performance
# --------------------------------------------------------------------------


def evaluate_by_group(df: pd.DataFrame, group_col: str, target_col: str, pred_col: str) -> pd.DataFrame:
    """Breaks out ``evaluate_regression``/``evaluate_classification`` per
    distinct value of ``group_col`` (a regime code, a calendar year, ...).
    Groups with fewer than 3 usable rows are skipped rather than reporting
    a meaningless metric."""
    kind = TARGET_KIND[target_col]
    rows = []
    for group_value, sub in df.groupby(group_col):
        clean = sub.dropna(subset=[target_col, pred_col])
        if len(clean) < 3:
            continue
        metrics = (
            evaluate_regression(clean[target_col], clean[pred_col])
            if kind == "regression"
            else evaluate_classification(clean[target_col], clean[pred_col])
        )
        rows.append({group_col: group_value, "n": len(clean), **metrics})
    return pd.DataFrame(rows)


def metrics_by_year(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    tagged = df.assign(year=pd.to_datetime(df["timestamp"]).dt.year)
    return evaluate_by_group(tagged, "year", target_col, pred_col)


# --------------------------------------------------------------------------
# 6. Rank IC / calibration
# --------------------------------------------------------------------------


def rank_ic_report(df: pd.DataFrame, target_col: str, pred_col: str) -> dict[str, float]:
    sub = df.dropna(subset=[target_col, pred_col])
    return {"rank_ic": information_coefficient(sub[target_col], sub[pred_col]), "n": len(sub)}


def calibration_report(df: pd.DataFrame, target_col: str, pred_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Thin wrapper around ``models.evaluate.calibration_curve`` for a
    classification target (``positive_5d``/``positive_20d``): buckets
    predicted probabilities into ``n_bins`` and compares the mean
    predicted probability to the realised positive rate per bin -- a
    well-calibrated model's two columns should track closely."""
    if TARGET_KIND.get(target_col) != "classification":
        raise ValueError(f"calibration_report is only meaningful for a classification target, got {target_col!r}")
    return calibration_curve(df[target_col], df[pred_col], n_bins=n_bins)


# --------------------------------------------------------------------------
# 7. Transaction-cost stress testing
# --------------------------------------------------------------------------


def build_quantile_portfolio_returns(df: pd.DataFrame, target_col: str, pred_col: str, top_frac: float = 0.2) -> pd.DataFrame:
    """For each date, ranks symbols by the predicted signal, goes long the
    top ``top_frac`` and short the bottom ``top_frac`` (equal-weighted),
    and realises ``target_col`` (the actual forward excess return).
    Returns one row per date with ``gross_return`` and ``turnover``
    (the fraction of long+short names that changed since the prior
    rebalance -- the input transaction-cost stress testing scales against).

    **Do not feed this into ``sharpe_ratio()`` for a reported annualised
    Sharpe** (V0.3 Stage 2 audit finding). When ``target_col`` is a
    multi-day forward target (``excess_return_5d``/``excess_return_20d``,
    the only targets this pipeline has), consecutive rows here overlap --
    each shares nearly all of its underlying price window with its
    neighbours -- so the series is heavily autocorrelated, understating
    its true sample volatility, and ``sqrt(252)`` annualisation assumes
    252 INDEPENDENT one-day periods a year, which this series does not
    have. Both effects inflate an annualised Sharpe computed this way; use
    ``backtesting.daily_portfolio.build_daily_rebalanced_portfolio_returns``
    + ``sharpe_audit_report`` for a genuine one, and keep this function
    only for the relative-degradation diagnostics it was built for
    (``cost_stress_test``, ``execution_delay_stress_test``)."""
    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    for ts, group in df.sort_values("timestamp").groupby("timestamp"):
        sub = group.dropna(subset=[target_col, pred_col])
        n = len(sub)
        k = max(1, int(np.floor(n * top_frac)))
        if n < 2 * k:
            continue
        ranked = sub.sort_values(pred_col, ascending=False)
        long_syms = set(ranked.iloc[:k]["symbol"])
        short_syms = set(ranked.iloc[-k:]["symbol"])
        long_ret = float(ranked.iloc[:k][target_col].mean())
        short_ret = float(ranked.iloc[-k:][target_col].mean())
        gross_return = 0.5 * long_ret - 0.5 * short_ret
        book_size = max(1, 2 * k)
        turnover = 1.0 if not rows else (len(long_syms ^ prev_long) + len(short_syms ^ prev_short)) / book_size
        rows.append({"timestamp": ts, "gross_return": gross_return, "turnover": turnover})
        prev_long, prev_short = long_syms, short_syms
    return pd.DataFrame(rows)


def cost_stress_test(portfolio_returns: pd.DataFrame, cost_bps_grid: tuple[float, ...] = (0, 5, 10, 20, 30, 50)) -> pd.DataFrame:
    """Sweeps round-trip cost assumptions (bps of turned-over notional) and
    reports the resulting net Sharpe -- shows at what cost level the
    signal's edge disappears."""
    rows = []
    for bps in cost_bps_grid:
        cost = portfolio_returns["turnover"] * (bps / 10_000.0)
        net = portfolio_returns["gross_return"] - cost
        rows.append({"cost_bps": bps, "net_sharpe": sharpe_ratio(net), "mean_net_return": float(net.mean()) if len(net) else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 8. Execution-delay stress testing
# --------------------------------------------------------------------------


def execution_delay_stress_test(
    df: pd.DataFrame, target_col: str, pred_col: str, delays_days: tuple[int, ...] = (0, 1, 2, 3, 5, 10)
) -> pd.DataFrame:
    """Simulates acting on a prediction ``delay`` trading days late: within
    each symbol's own timeline, shifts the prediction column forward by
    ``delay`` rows (so the "prediction used" at row t is what was actually
    predicted ``delay`` days earlier) and recomputes rank IC against the
    same realised target -- showing how fast the signal's predictive
    value decays with execution delay."""
    rows = []
    for delay in delays_days:
        shifted_frames = []
        for _symbol, group in df.sort_values("timestamp").groupby("symbol"):
            g = group.copy()
            g["_delayed_pred"] = g[pred_col].shift(delay)
            shifted_frames.append(g)
        shifted = pd.concat(shifted_frames, ignore_index=True).dropna(subset=[target_col, "_delayed_pred"])
        ic = information_coefficient(shifted[target_col], shifted["_delayed_pred"]) if len(shifted) >= 3 else float("nan")
        rows.append({"delay_days": delay, "rank_ic": ic, "n": len(shifted)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 9. Factor exposure report
# --------------------------------------------------------------------------


def factor_exposure_report(df: pd.DataFrame, pred_col: str, factor_cols: list[str]) -> pd.DataFrame:
    """OLS-regresses the model's predicted signal on a set of style-factor
    proxies (e.g. momentum, size/liquidity, volatility percentiles) so a
    reviewer can tell whether the "alpha" is simply a repackaging of a
    known factor rather than novel signal. Returns one row per factor with
    its standardized loading, plus an ``intercept`` row; every row carries
    the regression's overall R-squared for context."""
    sub = df.dropna(subset=[pred_col, *factor_cols])
    if len(sub) < len(factor_cols) + 2:
        return pd.DataFrame(columns=["factor", "loading", "r_squared"])

    y = sub[pred_col].to_numpy(dtype=float)
    y_std = y.std() or 1.0
    y = (y - y.mean()) / y_std

    x_raw = sub[factor_cols].to_numpy(dtype=float)
    x_std = np.where(x_raw.std(axis=0) == 0, 1.0, x_raw.std(axis=0))
    x = (x_raw - x_raw.mean(axis=0)) / x_std
    x_design = np.column_stack([np.ones(len(x)), x])

    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(x_design, y, rcond=None)
    y_hat = x_design @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")

    rows = [{"factor": "intercept", "loading": float(coeffs[0]), "r_squared": r_squared}]
    rows.extend({"factor": name, "loading": float(coef), "r_squared": r_squared} for name, coef in zip(factor_cols, coeffs[1:], strict=True))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 10. Feature-importance stability
# --------------------------------------------------------------------------


def feature_importance_stability(fold_results: list[PurgedFoldResult], target_col: str) -> dict[str, float]:
    """Mean pairwise Spearman rank correlation of feature importances
    across every pair of folds' models for ``target_col``. A low mean
    correlation means the model picks different features as important
    fold to fold -- a red flag for overfitting to fold-specific noise
    rather than learning a stable signal."""
    importance_series = []
    for result in fold_results:
        booster = result.trained.boosters.get(target_col)
        if booster is None:
            continue
        imp = feature_importance(booster, result.trained.feature_names)
        importance_series.append(imp.set_index("feature")["importance"])
    if len(importance_series) < 2:
        return {"mean_pairwise_spearman": float("nan"), "n_folds": len(importance_series), "n_pairs": 0}

    combined = pd.concat(importance_series, axis=1, keys=range(len(importance_series))).fillna(0.0)
    corr_matrix = combined.corr(method="spearman")
    n = len(combined.columns)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    values = [corr_matrix.iloc[i, j] for i, j in pairs]
    return {"mean_pairwise_spearman": float(np.mean(values)) if values else float("nan"), "n_folds": n, "n_pairs": len(pairs)}
