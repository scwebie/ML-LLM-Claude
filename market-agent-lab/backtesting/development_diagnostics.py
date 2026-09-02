"""V0.3 Stage 3: development-only signal diagnostics.

Every function here takes a DEVELOPMENT-region evaluation frame (the
concatenated out-of-sample walk-forward fold predictions joined back to
realised targets and context columns, e.g.
``backtesting.robustness.build_evaluation_frame``'s output over
``development_df``) and NEVER the final holdout or post-holdout regions.
Nothing here is used to select, tune, or qualify a model -- it exists to
diagnose WHY development performance may not generalise: whether the
signal is concentrated in a few years, a few symbols, or a particular
market regime, and how quickly it decays.

**Regime labels.** ``agents/market_overview.py``'s regime classifier
reads macro z-scores under V0.1's synthetic ``SYN_*`` key names
(``SYN_VOL_INDEX_zscore``, ``SYN_GROWTH_INDEX_zscore``, ...); the real
feature pipeline (``data/real_features.py``) never populates those keys
(real macro series are ingested and z-scored under ``FRED_*``/``BLS_*``
names, e.g. ``macro_raw_FRED_VIXCLS_zscore``), so on real data every row
silently gets the SAME constant regime code (macro_feats.get(...) always
falls back to its 0.0 default) -- an audit finding of this stage, not
something fixed here (fixing the live feature pipeline is out of scope
for a diagnostics-only stage and would perturb the trained feature
matrix). Instead, this module reclassifies volatility/rate regimes
directly from the real z-scored macro columns already present in the
development matrix, using the SAME fixed thresholds
``agents/market_overview.py`` already uses -- "already implemented", just
correctly wired for real data, and computed with a strictly backward-
looking window (point-in-time safe, no future information).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.evaluate import information_coefficient

# Same fixed thresholds as agents/market_overview.py's _vol_regime, ported
# to the real (FRED-sourced) volatility z-score column.
_VOL_ELEVATED_Z = 1.0
_VOL_CRISIS_Z = 2.0
_VOL_LOW_Z = -1.0


def pearson_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Raw (non-ranked) Pearson correlation between predicted and
    realised values -- reported alongside rank IC
    (``models.evaluate.information_coefficient``, Spearman) since the two
    can diverge meaningfully when the relationship is monotonic but
    non-linear, or when outliers dominate one but not the other."""
    if len(y_true) < 3:
        return float("nan")
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    if y_true.std(ddof=0) == 0 or y_pred.std(ddof=0) == 0:
        return float("nan")
    return float(y_true.corr(y_pred, method="pearson"))


def _fisher_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Fisher z-transform confidence interval for a correlation
    coefficient -- a standard, practical approximation, used here for
    rank IC as well as Pearson (common practice for Spearman's rho at
    moderate-to-large n)."""
    if n < 4 or r != r or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm

    crit = norm.ppf(0.5 + confidence / 2.0)
    lo, hi = z - crit * se, z + crit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def _ic_stats(sub: pd.DataFrame, target_col: str, pred_col: str) -> dict:
    clean = sub.dropna(subset=[target_col, pred_col])
    n = len(clean)
    rank_ic = information_coefficient(clean[target_col], clean[pred_col]) if n >= 3 else float("nan")
    p_ic = pearson_ic(clean[target_col], clean[pred_col]) if n >= 3 else float("nan")
    ci_low, ci_high = _fisher_ci(rank_ic, n) if n >= 3 else (float("nan"), float("nan"))
    return {"n": n, "pearson_ic": p_ic, "rank_ic": rank_ic, "rank_ic_ci_low": ci_low, "rank_ic_ci_high": ci_high}


# --------------------------------------------------------------------------
# 1. IC by calendar year
# --------------------------------------------------------------------------


def ic_by_year(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for year, sub in df.assign(year=pd.to_datetime(df["timestamp"]).dt.year).groupby("year"):
        stats = _ic_stats(sub, target_col, pred_col)
        if stats["n"] < 3:
            continue
        rows.append({"year": int(year), **stats})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. IC by market regime
# --------------------------------------------------------------------------


def classify_volatility_regime(vol_zscore: pd.Series) -> pd.Series:
    """HIGH / NORMAL / LOW, same thresholds as
    agents/market_overview.py::_vol_regime (CRISIS folded into HIGH --
    too rare in practice to support as its own IC group)."""
    z = vol_zscore.astype(float)
    out = pd.Series("NORMAL", index=z.index, dtype=object)
    out[z >= _VOL_ELEVATED_Z] = "HIGH"
    out[z <= _VOL_LOW_Z] = "LOW"
    return out


def classify_rate_regime(rate_zscore_by_date: pd.Series, lookback_days: int = 20) -> pd.Series:
    """RISING / FALLING, from the trailing change in a rate z-score over
    ``lookback_days`` -- strictly backward-looking (today vs. N days ago),
    so this never uses information not yet available as of each date.
    ``rate_zscore_by_date`` must be indexed by date, one row per date,
    already sorted ascending."""
    diff = rate_zscore_by_date.astype(float).diff(lookback_days)
    return pd.Series(np.where(diff > 0, "RISING", "FALLING"), index=rate_zscore_by_date.index)


def classify_risk_regime(breadth_advance_decline: pd.Series, vol_regime: pd.Series) -> pd.Series:
    """RISK_ON / RISK_OFF: positive market breadth (more advancers than
    decliners cross-sectionally) AND volatility not HIGH; RISK_OFF
    otherwise. A simple, defensible composite of two already-computed,
    point-in-time-safe signals -- not a new data source."""
    is_risk_on = (breadth_advance_decline.astype(float) > 0.0) & (vol_regime != "HIGH")
    return pd.Series(np.where(is_risk_on, "RISK_ON", "RISK_OFF"), index=breadth_advance_decline.index)


def ic_by_regime(df: pd.DataFrame, target_col: str, pred_col: str, regime_col: str) -> pd.DataFrame:
    rows = []
    for regime, sub in df.groupby(regime_col):
        stats = _ic_stats(sub, target_col, pred_col)
        if stats["n"] < 3:
            continue
        rows.append({"regime": regime, **stats})
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. IC by symbol / breadth
# --------------------------------------------------------------------------


def ic_by_symbol(df: pd.DataFrame, target_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for symbol, sub in df.groupby("symbol"):
        stats = _ic_stats(sub, target_col, pred_col)
        if stats["n"] < 3:
            continue
        rows.append({"symbol": symbol, **stats})
    return pd.DataFrame(rows).sort_values("rank_ic", ascending=False).reset_index(drop=True)


def signal_breadth_report(ic_by_symbol_df: pd.DataFrame) -> dict:
    """Whether the signal is broad (many symbols individually show a
    positive IC) or dominated by a handful of names."""
    if ic_by_symbol_df.empty:
        return {
            "n_symbols": 0, "pct_positive_ic": float("nan"), "median_symbol_ic": float("nan"),
            "iqr_symbol_ic": float("nan"),
        }
    ic = ic_by_symbol_df["rank_ic"].dropna()
    if ic.empty:
        return {"n_symbols": len(ic_by_symbol_df), "pct_positive_ic": float("nan"), "median_symbol_ic": float("nan"), "iqr_symbol_ic": float("nan")}
    q1, q3 = ic.quantile(0.25), ic.quantile(0.75)
    return {
        "n_symbols": int(len(ic)),
        "pct_positive_ic": float((ic > 0).mean()),
        "median_symbol_ic": float(ic.median()),
        "iqr_symbol_ic": float(q3 - q1),
    }


# --------------------------------------------------------------------------
# 4. IC decay across realised-return horizons
# --------------------------------------------------------------------------


def build_forward_returns(market_df: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20), price_col: str = "adjusted_close") -> pd.DataFrame:
    """Point-in-time-safe forward realised returns at each horizon,
    computed purely from already-ingested historical prices (never a
    model output): for each (symbol, date), the actual price change from
    that date to `horizon` TRADING DAYS later. No look-ahead is used to
    construct predictions from this -- it exists only to measure how a
    prediction correlates with what actually happened at different
    horizons than the one it was trained on."""
    wide = market_df.dropna(subset=[price_col]).pivot_table(index="timestamp", columns="symbol", values=price_col).sort_index()
    rows = []
    for h in horizons:
        fwd = wide.shift(-h) / wide - 1.0
        for symbol in wide.columns:
            series = fwd[symbol].dropna()
            for ts, val in series.items():
                rows.append({"symbol": symbol, "timestamp": ts, "horizon_days": h, "forward_return": float(val)})
    return pd.DataFrame(rows)


def ic_decay_report(predictions_df: pd.DataFrame, market_df: pd.DataFrame, pred_col: str, horizons: tuple[int, ...] = (1, 5, 10, 20)) -> pd.DataFrame:
    """Correlates a single fixed prediction column against ACTUAL forward
    returns realised at each horizon -- shows whether the signal's
    predictive power decays (or grows) at horizons other than the one the
    model was trained/scored on."""
    forward = build_forward_returns(market_df, horizons)
    rows = []
    for h in horizons:
        sub = forward[forward["horizon_days"] == h].merge(
            predictions_df[["symbol", "timestamp", pred_col]], on=["symbol", "timestamp"], how="inner"
        )
        stats = _ic_stats(sub, "forward_return", pred_col)
        rows.append({"horizon_days": h, **stats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Assembled report (used by the V0.3 CLI/reporting layer)
# --------------------------------------------------------------------------


def build_development_diagnostics_report(
    fold_results: list, development_df: pd.DataFrame, market_df: pd.DataFrame, target_col: str = "excess_return_20d"
) -> dict:
    """Assembles every Stage 3 diagnostic from DEVELOPMENT-region walk-
    forward fold predictions only. Never touches the holdout or
    post-holdout regions -- callers must pass ``development_df`` and its
    own ``fold_results``, never anything derived from the fixed holdout."""
    from backtesting.purged_walk_forward import TARGET_TO_PRED_COL
    from backtesting.robustness import build_evaluation_frame

    if not fold_results:
        return {"status": "no walk-forward fold results available"}

    pred_col = TARGET_TO_PRED_COL[target_col]
    report: dict = {}

    for tc in ("excess_return_5d", "excess_return_20d"):
        pc = TARGET_TO_PRED_COL[tc]
        ef = build_evaluation_frame(fold_results, development_df, tc)
        if ef.empty:
            continue
        report[f"ic_by_year_{tc}"] = ic_by_year(ef, tc, pc).to_dict(orient="records")

    context_cols = [
        c for c in ("macro_raw_FRED_VIXCLS_zscore", "macro_raw_FRED_DGS10_zscore", "breadth_advance_decline_proxy")
        if c in development_df.columns
    ]
    ef_ctx = build_evaluation_frame(fold_results, development_df, target_col, extra_cols=context_cols)
    if ef_ctx.empty:
        report["regime_and_symbol_diagnostics"] = "no out-of-sample development predictions available"
        return report

    if "macro_raw_FRED_VIXCLS_zscore" in ef_ctx.columns:
        ef_ctx["vol_regime"] = classify_volatility_regime(ef_ctx["macro_raw_FRED_VIXCLS_zscore"])
        report["ic_by_volatility_regime"] = ic_by_regime(ef_ctx, target_col, pred_col, "vol_regime").to_dict(orient="records")

    if "macro_raw_FRED_DGS10_zscore" in ef_ctx.columns:
        rate_by_date = ef_ctx.drop_duplicates("timestamp").set_index("timestamp")["macro_raw_FRED_DGS10_zscore"].sort_index()
        ef_ctx["rate_regime"] = ef_ctx["timestamp"].map(classify_rate_regime(rate_by_date))
        report["ic_by_rate_regime"] = ic_by_regime(ef_ctx, target_col, pred_col, "rate_regime").to_dict(orient="records")

    if "vol_regime" in ef_ctx.columns and "breadth_advance_decline_proxy" in ef_ctx.columns:
        ef_ctx["risk_regime"] = classify_risk_regime(ef_ctx["breadth_advance_decline_proxy"], ef_ctx["vol_regime"])
        report["ic_by_risk_regime"] = ic_by_regime(ef_ctx, target_col, pred_col, "risk_regime").to_dict(orient="records")

    by_symbol = ic_by_symbol(ef_ctx, target_col, pred_col)
    report["ic_by_symbol"] = by_symbol.to_dict(orient="records")
    report["breadth"] = signal_breadth_report(by_symbol)

    last_fold = fold_results[-1]
    report["ic_decay"] = ic_decay_report(last_fold.predictions, market_df, pred_col).to_dict(orient="records")

    return report
