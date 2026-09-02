"""Tests for backtesting/development_diagnostics.py (V0.3 Stage 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.development_diagnostics import (
    build_development_diagnostics_report,
    build_forward_returns,
    classify_rate_regime,
    classify_risk_regime,
    classify_volatility_regime,
    ic_by_regime,
    ic_by_symbol,
    ic_by_year,
    ic_decay_report,
    pearson_ic,
    signal_breadth_report,
)


def _synthetic_eval_frame(n_symbols=10, n_years=3, seed=1, edge_by_year=None):
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    rows = []
    for year_idx in range(n_years):
        year = 2020 + year_idx
        edge = 1.0 if edge_by_year is None else edge_by_year[year_idx]
        dates = pd.bdate_range(f"{year}-01-02", periods=60)
        for d in dates:
            for sym in symbols:
                target = rng.normal(0, 0.02)
                pred = edge * target * 0.7 + rng.normal(0, 0.01)
                rows.append({"symbol": sym, "timestamp": d, "target": target, "pred": pred})
    return pd.DataFrame(rows)


# --- pearson_ic -------------------------------------------------------------------------------


def test_pearson_ic_perfect_linear_relationship_is_one():
    x = pd.Series(np.arange(50, dtype=float))
    assert pearson_ic(x, x * 2 + 1) == pytest.approx(1.0)


def test_pearson_ic_too_few_observations_is_nan():
    result = pearson_ic(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.0]))
    assert result != result  # NaN


# --- IC by year ---------------------------------------------------------------------------------


def test_ic_by_year_detects_a_year_with_no_real_edge():
    """A signal with a real edge in years 0/2 but pure noise in year 1
    must show a materially lower IC for year 1 -- proving the per-year
    breakdown actually separates strong from weak periods, not just
    reporting one blended number."""
    df = _synthetic_eval_frame(n_years=3, edge_by_year=[1.0, 0.0, 1.0])
    report = ic_by_year(df, "target", "pred")
    assert list(report["year"]) == [2020, 2021, 2022]
    assert report.loc[report["year"] == 2021, "rank_ic"].iloc[0] < report.loc[report["year"] == 2020, "rank_ic"].iloc[0]
    assert (report["n"] > 0).all()
    assert {"pearson_ic", "rank_ic", "rank_ic_ci_low", "rank_ic_ci_high"} <= set(report.columns)


# --- regime classification ----------------------------------------------------------------------


def test_classify_volatility_regime_thresholds():
    z = pd.Series([-2.0, -0.5, 0.5, 1.5, 3.0])
    regimes = classify_volatility_regime(z)
    assert list(regimes) == ["LOW", "NORMAL", "NORMAL", "HIGH", "HIGH"]


def test_classify_rate_regime_uses_only_backward_looking_window():
    dates = pd.bdate_range("2021-01-04", periods=30)
    # Monotonically rising z-score -- every date's 20-day-back diff must
    # be positive from day 20 onward, using ONLY past values.
    z = pd.Series(np.linspace(-1, 1, 30), index=dates)
    regimes = classify_rate_regime(z, lookback_days=20)
    assert regimes.iloc[25] == "RISING"
    # The classification at date i must be identical whether or not
    # future dates exist in the series -- proving no look-ahead.
    truncated_regimes = classify_rate_regime(z.iloc[:26], lookback_days=20)
    assert regimes.iloc[25] == truncated_regimes.iloc[25]


def test_classify_risk_regime_combines_breadth_and_volatility():
    breadth = pd.Series([0.3, -0.2, 0.5])
    vol = pd.Series(["NORMAL", "NORMAL", "HIGH"])
    result = classify_risk_regime(breadth, vol)
    assert list(result) == ["RISK_ON", "RISK_OFF", "RISK_OFF"]  # third: positive breadth but HIGH vol -> risk off


def test_ic_by_regime_groups_correctly():
    df = _synthetic_eval_frame(n_symbols=8, n_years=1, edge_by_year=[1.0])
    df["regime"] = np.where(df["timestamp"].dt.day % 2 == 0, "REGIME_A", "REGIME_B")
    report = ic_by_regime(df, "target", "pred", "regime")
    assert set(report["regime"]) <= {"REGIME_A", "REGIME_B"}
    assert (report["n"] > 0).all()


# --- IC by symbol / breadth ----------------------------------------------------------------------


def test_ic_by_symbol_and_breadth_report():
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2021-01-04", periods=100)
    rows = []
    # SYM0 has a real edge; every other symbol is pure noise.
    for d in dates:
        target0 = rng.normal(0, 0.02)
        rows.append({"symbol": "SYM0", "timestamp": d, "target": target0, "pred": target0 * 0.8 + rng.normal(0, 0.005)})
        for i in range(1, 6):
            rows.append({"symbol": f"SYM{i}", "timestamp": d, "target": rng.normal(0, 0.02), "pred": rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)
    by_symbol = ic_by_symbol(df, "target", "pred")
    assert len(by_symbol) == 6
    assert by_symbol.iloc[0]["symbol"] == "SYM0"  # sorted descending by rank_ic -- the real signal on top

    breadth = signal_breadth_report(by_symbol)
    assert breadth["n_symbols"] == 6
    assert 0.0 <= breadth["pct_positive_ic"] <= 1.0


def test_signal_breadth_report_empty_input():
    result = signal_breadth_report(pd.DataFrame())
    assert result["n_symbols"] == 0


# --- IC decay -------------------------------------------------------------------------------------


def test_build_forward_returns_is_point_in_time_and_one_row_per_symbol_date_horizon():
    dates = pd.bdate_range("2021-01-04", periods=30)
    prices = pd.DataFrame({"AAA": np.linspace(100, 130, 30), "BBB": np.linspace(50, 50, 30)}, index=dates)
    market_df = pd.DataFrame(
        [{"symbol": s, "timestamp": d, "adjusted_close": prices.loc[d, s]} for s in prices.columns for d in dates]
    )
    result = build_forward_returns(market_df, horizons=(1, 5))
    assert set(result["horizon_days"]) == {1, 5}
    # AAA rises linearly -- every forward return must be positive.
    aaa_1d = result[(result["symbol"] == "AAA") & (result["horizon_days"] == 1)]
    assert (aaa_1d["forward_return"] > 0).all()
    # BBB is flat -- forward returns must be ~0.
    bbb_1d = result[(result["symbol"] == "BBB") & (result["horizon_days"] == 1)]
    assert bbb_1d["forward_return"].abs().max() < 1e-9


def test_ic_decay_report_shows_high_ic_at_the_horizon_the_signal_was_built_for():
    """A prediction that is literally the 5-day-forward return (plus a
    little noise) must show its strongest IC at horizon_days=5, not at 1,
    10, or 20 -- proving the decay report actually measures horizon-
    specific alignment rather than a constant/meaningless number."""
    rng = np.random.default_rng(11)
    n_days = 120
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    symbols = [f"SYM{i}" for i in range(6)]
    daily_rets = pd.DataFrame({s: rng.normal(0, 0.01, n_days) for s in symbols}, index=dates)
    prices = (1 + daily_rets).cumprod() * 100.0
    market_df = pd.DataFrame(
        [{"symbol": s, "timestamp": d, "adjusted_close": prices.loc[d, s]} for s in symbols for d in dates]
    )
    fwd5 = (prices.shift(-5) / prices - 1.0)
    pred_rows = []
    for s in symbols:
        for d in dates:
            v = fwd5.loc[d, s]
            if pd.isna(v):
                continue
            pred_rows.append({"symbol": s, "timestamp": d, "pred": v + rng.normal(0, 0.001)})
    predictions_df = pd.DataFrame(pred_rows)

    report = ic_decay_report(predictions_df, market_df, "pred", horizons=(1, 5, 10, 20))
    assert list(report["horizon_days"]) == [1, 5, 10, 20]
    best_horizon = report.loc[report["rank_ic"].idxmax(), "horizon_days"]
    assert best_horizon == 5


# --- build_development_diagnostics_report: end-to-end integration on the real pipeline -----------


def test_build_development_diagnostics_report_end_to_end(monkeypatch):
    import pandas as pd

    import real_pipeline as rp
    from database.db import fresh_connection
    from database.schema import init_schema
    from tests.test_real_pipeline import SYMBOLS, _seed_market_data

    monkeypatch.setattr(rp, "ingest_event_probabilities", lambda *a, **k: {"status": "SKIPPED_IN_TEST"})
    with fresh_connection(":memory:") as con:
        init_schema(con)
        _seed_market_data(con)
        rp.build_real_features_step(con, SYMBOLS, "test_universe", pd.Timestamp("2020-01-02"))
        evaluation = rp.evaluate_real_step(con, SYMBOLS, initial_train_fraction=0.6, validation_fraction=0.15)

        from database import repository as repo

        market = repo.get_market_observations(con, symbols=SYMBOLS)
        report = build_development_diagnostics_report(evaluation.fold_results, evaluation.development_df, market)

    assert "ic_by_year_excess_return_20d" in report
    assert "ic_by_symbol" in report
    assert "breadth" in report
    assert "ic_decay" in report
    assert len(report["ic_decay"]) == 4  # horizons 1, 5, 10, 20
    assert len(report["ic_by_symbol"]) == len(SYMBOLS)
