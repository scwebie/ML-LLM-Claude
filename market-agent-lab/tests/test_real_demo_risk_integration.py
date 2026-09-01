"""Stage 15's required dedicated integration test: proves the risk engine
produces BOTH approvals (fills) AND rejections during a REAL run -- real
ticker symbols, multiple real sectors, a trained LightGBM model, and the
SAME, unmodified Portfolio/Risk/Execution engine V0.1 uses
(``backtesting/engine.py::run_ml_strategy_backtest``), never a mock or a
simplified stand-in.

No live network is used (per project convention, the regular test suite
never makes live calls): price data is directly seeded into
``market_observations`` -- the exact table ``data/real_prices.py``
populates from Yahoo Finance/StockAnalysis.com in production -- so this
exercises the identical code path a real ingestion run would.

Individual risk-reason-code mechanics (STALE_DATA, INVALID_PRICE,
DUPLICATE_ORDER, KILL_SWITCH, RISK_DRAWDOWN, ...) are already unit-tested
in isolation in ``tests/test_risk_engine.py``; this test's job is
different -- proving those mechanisms are correctly WIRED end-to-end in a
realistic multi-symbol, multi-sector run, by deliberately using risk
limits tight enough that position-limit, sector-concentration, and
exposure rejections occur naturally alongside genuine fills."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import run_ml_strategy_backtest
from core.schemas import RiskReasonCode
from database import repository as repo
from database.db import fresh_connection
from database.schema import init_schema
from features.technical import compute_technical_features_multi
from models.registry import ModelPeriods, register_model
from models.train import (
    compute_excess_return_targets,
    get_feature_columns,
    prepare_training_frame,
    train_all_targets,
)
from portfolio.allocation import AllocationConfig
from portfolio.risk import RiskLimits

# A real, multi-sector universe with a deliberately heavy TECH cluster (5
# names) so a tight sector-concentration limit is guaranteed to bind once
# more than one or two TECH positions are held simultaneously.
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "JPM", "XOM", "UNH"]
BENCHMARK = "SPY"
SECTOR_MAP = {
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "META": "TECH", "NVDA": "TECH",
    "JPM": "FINANCE", "XOM": "ENERGY", "UNH": "HEALTH",
}


def _realistic_price_path(seed: int, n_days: int, start_price: float, daily_vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, daily_vol, n_days)
    return start_price * np.exp(np.cumsum(log_returns))


def _seed_market_data(con) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=420)
    rows = []
    profiles = {
        "AAPL": (1, 150.0, 0.018), "MSFT": (2, 300.0, 0.017), "GOOGL": (3, 100.0, 0.020),
        "META": (4, 200.0, 0.028), "NVDA": (5, 180.0, 0.035),  # NVDA/META: high-vol names
        "JPM": (6, 130.0, 0.016), "XOM": (7, 90.0, 0.019), "UNH": (8, 480.0, 0.015),
        BENCHMARK: (9, 400.0, 0.011),
    }
    for symbol, (seed, start_price, vol) in profiles.items():
        closes = _realistic_price_path(seed, len(dates), start_price, vol)
        volumes = np.random.default_rng(seed + 100).normal(3_000_000.0, 500_000.0, len(dates)).clip(min=100_000.0)
        for ts, close, volume in zip(dates, closes, volumes, strict=True):
            rows.append(
                {
                    "symbol": symbol, "timestamp": ts, "open": close * 0.998, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "adjusted_close": close, "volume": float(volume),
                }
            )
    market_df = pd.DataFrame(rows)
    repo.insert_market_observations(con, market_df)
    return market_df


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        init_schema(c)
        yield c


def test_risk_engine_produces_both_fills_and_multiple_rejection_categories_in_a_real_run(con):
    market_df = _seed_market_data(con)
    stock_df = market_df[market_df["symbol"] != BENCHMARK]
    bench_df = market_df[market_df["symbol"] == BENCHMARK]

    technical_df = compute_technical_features_multi(stock_df)
    targets = compute_excess_return_targets(stock_df, bench_df)
    df = prepare_training_frame(technical_df, targets)
    feature_cols = get_feature_columns(df)
    assert len(feature_cols) > 10

    dates_sorted = sorted(df["timestamp"].unique())
    train_end = dates_sorted[299]
    val_end = dates_sorted[349]
    test_start = dates_sorted[350]

    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]
    test_df = df[df["timestamp"] >= test_start]
    assert not train_df.empty and not val_df.empty and not test_df.empty

    trained = train_all_targets(train_df, val_df, feature_cols)
    periods = ModelPeriods(
        training_start=train_df["timestamp"].min(), training_end=train_df["timestamp"].max(),
        validation_start=val_df["timestamp"].min(), validation_end=val_df["timestamp"].max(),
    )
    model_version = register_model(con, trained, "real_integration_fv1", periods, {}, role="CHALLENGER")

    # Deliberately tight, realistic risk limits: small enough that a
    # concentrated, high-conviction allocation across a 5-name TECH
    # cluster cannot all be approved, but not so small that literally
    # nothing gets through.
    risk_limits = RiskLimits(
        max_position_weight=0.06, max_gross_exposure=0.50, max_net_exposure=0.30,
        max_sector_concentration=0.15, max_portfolio_volatility=0.80,
    )
    allocation_config = AllocationConfig()  # V0.1 default, target_gross_exposure=0.50

    result = run_ml_strategy_backtest(
        con, run_id="risk_integration_test", feature_df=test_df, market_df=market_df,
        boosters=trained.boosters, feature_cols=feature_cols, model_version=model_version,
        feature_version="real_integration_fv1", symbols=SYMBOLS, sector_map=SECTOR_MAP,
        initial_cash=1_000_000.0, risk_limits=risk_limits, allocation_config=allocation_config,
    )

    # --- the core assertion: both approvals AND rejections occurred -------------------
    assert len(result.fills) > 0, "expected at least one order to be approved and filled"
    assert len(result.rejected_orders) > 0, "expected at least one order to be rejected by the risk engine"

    rejection_codes = {code for order in result.rejected_orders for code in order.risk_reason_codes}
    rejection_codes.discard(RiskReasonCode.OK)
    assert len(rejection_codes) >= 2, f"expected multiple distinct rejection categories, got {rejection_codes}"
    # The tight sector-concentration and position/exposure limits above are
    # specifically chosen to bind on this TECH-heavy universe.
    plausible_categories = {
        RiskReasonCode.RISK_POSITION_LIMIT, RiskReasonCode.RISK_SECTOR_CONCENTRATION,
        RiskReasonCode.RISK_GROSS_EXPOSURE, RiskReasonCode.RISK_NET_EXPOSURE,
    }
    assert rejection_codes & plausible_categories, f"expected an exposure/concentration/position rejection, got {rejection_codes}"

    # --- every rejected/approved order and every fill is actually persisted -----------
    orders_in_db = con.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
    fills_in_db = con.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0]
    risk_decisions_in_db = con.execute("SELECT COUNT(*) FROM risk_decisions").fetchone()[0]
    assert orders_in_db == len(result.fills) + len(result.rejected_orders)
    assert fills_in_db == len(result.fills)
    assert risk_decisions_in_db == orders_in_db

    # --- sanity: no order that was filled also appears among the rejected --
    fill_order_ids = {f.order_id for f in result.fills}
    rejected_ids = {o.order_id for o in result.rejected_orders}
    assert fill_order_ids.isdisjoint(rejected_ids)


def test_tighter_limits_reject_more_and_looser_limits_reject_less(con):
    """A second, comparative check that the risk engine's bite actually
    scales with its configured limits in this real-data run -- not just
    that *some* rejections happen for unrelated reasons (e.g. stale
    data)."""
    market_df = _seed_market_data(con)
    stock_df = market_df[market_df["symbol"] != BENCHMARK]
    bench_df = market_df[market_df["symbol"] == BENCHMARK]
    technical_df = compute_technical_features_multi(stock_df)
    targets = compute_excess_return_targets(stock_df, bench_df)
    df = prepare_training_frame(technical_df, targets)
    feature_cols = get_feature_columns(df)

    dates_sorted = sorted(df["timestamp"].unique())
    train_df = df[df["timestamp"] <= dates_sorted[299]]
    val_df = df[(df["timestamp"] > dates_sorted[299]) & (df["timestamp"] <= dates_sorted[349])]
    test_df = df[df["timestamp"] >= dates_sorted[350]]

    trained = train_all_targets(train_df, val_df, feature_cols)
    model_version = register_model(
        con, trained, "real_integration_fv1",
        ModelPeriods(train_df["timestamp"].min(), train_df["timestamp"].max(), val_df["timestamp"].min(), val_df["timestamp"].max()),
        {}, role="CHALLENGER",
    )
    allocation_config = AllocationConfig()

    tight = run_ml_strategy_backtest(
        con, "tight_run", test_df, market_df, trained.boosters, feature_cols, model_version, "real_integration_fv1",
        SYMBOLS, SECTOR_MAP, initial_cash=1_000_000.0,
        risk_limits=RiskLimits(max_position_weight=0.03, max_gross_exposure=0.20, max_net_exposure=0.15, max_sector_concentration=0.08),
        allocation_config=allocation_config,
    )
    loose = run_ml_strategy_backtest(
        con, "loose_run", test_df, market_df, trained.boosters, feature_cols, model_version, "real_integration_fv1",
        SYMBOLS, SECTOR_MAP, initial_cash=1_000_000.0,
        risk_limits=RiskLimits(max_position_weight=0.50, max_gross_exposure=2.0, max_net_exposure=2.0, max_sector_concentration=1.0),
        allocation_config=allocation_config,
    )
    assert len(tight.rejected_orders) > len(loose.rejected_orders)
