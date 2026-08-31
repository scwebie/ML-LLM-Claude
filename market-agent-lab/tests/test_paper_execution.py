"""Tests for the Paper Execution Engine: fills, partial fills, rejections,
and account/portfolio accounting."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.schemas import OrderSide, OrderType, PaperOrder
from execution.fills import (
    REJECT_INSUFFICIENT_BUYING_POWER,
    REJECT_NO_VOLUME,
    REJECT_NOT_MARKETABLE,
    FillConfig,
    simulate_fill,
)
from execution.paper import PaperBroker
from portfolio.risk import RiskEngine, RiskLimits


def _order(symbol="SYN_A", side=OrderSide.BUY, quantity=10.0, order_type=OrderType.MARKET, price=100.0, limit=None, ts=None):
    return PaperOrder(
        symbol=symbol, side=side, quantity=quantity, order_type=order_type,
        proposed_price=price, limit_price=limit, timestamp=ts or datetime(2023, 1, 1), strategy_version="v1",
    )


# --- fills.py -------------------------------------------------------------------


def test_buy_fill_applies_spread_and_slippage_against_trader():
    order = _order(side=OrderSide.BUY, quantity=100)
    fill, reason = simulate_fill(order, execution_price=100.0, bar_volume=100_000, fill_timestamp=datetime(2023, 1, 2))
    assert reason is None
    assert fill.fill_price > 100.0  # buys pay up
    assert fill.quantity == 100
    assert fill.commission >= FillConfig().commission_min


def test_sell_fill_applies_spread_against_trader():
    order = _order(side=OrderSide.SELL, quantity=100)
    fill, reason = simulate_fill(order, execution_price=100.0, bar_volume=100_000, fill_timestamp=datetime(2023, 1, 2))
    assert reason is None
    assert fill.fill_price < 100.0  # sells receive less


def test_partial_fill_when_participation_exceeds_cap():
    config = FillConfig(max_participation_rate=0.1)
    order = _order(quantity=5000)
    fill, reason = simulate_fill(order, execution_price=100.0, bar_volume=10_000, fill_timestamp=datetime(2023, 1, 2), config=config)
    assert reason is None
    assert fill.quantity == pytest.approx(1000.0)  # 10% of 10,000 volume


def test_no_volume_rejects():
    order = _order(quantity=10)
    fill, reason = simulate_fill(order, execution_price=100.0, bar_volume=0, fill_timestamp=datetime(2023, 1, 2))
    assert fill is None
    assert reason == REJECT_NO_VOLUME


def test_unmarketable_limit_order_rejected():
    order = _order(side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT, limit=95.0)
    fill, reason = simulate_fill(order, execution_price=100.0, bar_volume=10_000, fill_timestamp=datetime(2023, 1, 2))
    assert fill is None
    assert reason == REJECT_NOT_MARKETABLE


def test_marketable_limit_order_fills():
    order = _order(side=OrderSide.BUY, quantity=10, order_type=OrderType.LIMIT, limit=105.0)
    fill, reason = simulate_fill(order, execution_price=100.0, bar_volume=10_000, fill_timestamp=datetime(2023, 1, 2))
    assert reason is None
    assert fill is not None


def test_insufficient_buying_power_rejects_when_cannot_afford_even_commission():
    order = _order(side=OrderSide.BUY, quantity=1000)
    fill, reason = simulate_fill(
        order, execution_price=100.0, bar_volume=100_000, fill_timestamp=datetime(2023, 1, 2), available_buying_power=0.5
    )
    assert fill is None
    assert reason == REJECT_INSUFFICIENT_BUYING_POWER


def test_buying_power_caps_partial_quantity():
    order = _order(side=OrderSide.BUY, quantity=100)
    fill, reason = simulate_fill(
        order, execution_price=100.0, bar_volume=100_000, fill_timestamp=datetime(2023, 1, 2), available_buying_power=1000.0
    )
    assert reason is None
    assert fill.quantity < 100
    assert fill.fill_price * fill.quantity + fill.commission <= 1000.0 + 1e-6


# --- paper.py: account bookkeeping -----------------------------------------------


def test_buy_then_sell_realizes_pnl_correctly():
    broker = PaperBroker(initial_cash=100_000.0)
    buy_order = _order(side=OrderSide.BUY, quantity=100)
    buy_fill, _ = simulate_fill(buy_order, execution_price=100.0, bar_volume=1_000_000, fill_timestamp=datetime(2023, 1, 2), config=FillConfig(half_spread_bps=0, slippage_coefficient=0, commission_per_share=0, commission_min=0))
    broker.apply_fill(buy_fill, buy_order)

    assert broker.account.positions["SYN_A"] == pytest.approx(100.0)
    assert broker.account.cash == pytest.approx(100_000.0 - 100 * buy_fill.fill_price)

    sell_order = _order(side=OrderSide.SELL, quantity=100)
    sell_fill, _ = simulate_fill(sell_order, execution_price=110.0, bar_volume=1_000_000, fill_timestamp=datetime(2023, 1, 3), config=FillConfig(half_spread_bps=0, slippage_coefficient=0, commission_per_share=0, commission_min=0))
    broker.apply_fill(sell_fill, sell_order)

    assert "SYN_A" not in broker.account.positions
    expected_pnl = (sell_fill.fill_price - buy_fill.fill_price) * 100
    assert broker.account.realized_pnl == pytest.approx(expected_pnl)


def test_equity_reflects_cash_plus_market_value():
    broker = PaperBroker(initial_cash=100_000.0)
    order = _order(side=OrderSide.BUY, quantity=100)
    fill, _ = simulate_fill(order, execution_price=100.0, bar_volume=1_000_000, fill_timestamp=datetime(2023, 1, 2), config=FillConfig(half_spread_bps=0, slippage_coefficient=0, commission_per_share=0, commission_min=0))
    broker.apply_fill(fill, order)

    equity_at_cost = broker.equity({"SYN_A": 100.0})
    assert equity_at_cost == pytest.approx(100_000.0)

    equity_after_gain = broker.equity({"SYN_A": 120.0})
    assert equity_after_gain == pytest.approx(100_000.0 + 100 * 20.0)


def test_submit_and_fill_end_to_end_respects_risk_engine():
    broker = PaperBroker(initial_cash=100_000.0)
    risk_engine = RiskEngine(RiskLimits(max_position_weight=0.01))  # tiny limit -> forces rejection
    order = _order(side=OrderSide.BUY, quantity=100)
    result = broker.submit_and_fill(
        orders=[order],
        risk_engine=risk_engine,
        decision_prices={"SYN_A": 100.0},
        execution_prices={"SYN_A": 100.0},
        bar_volumes={"SYN_A": 1_000_000},
        fill_timestamp=datetime(2023, 1, 2),
        sector_map={"SYN_A": "TECH"},
        position_volatility={"SYN_A": 0.2},
        market_data_timestamp=datetime(2023, 1, 1),
    )
    assert result.approved_orders == []
    assert len(result.rejected_orders) == 1
    assert result.fills == []
    assert "SYN_A" not in broker.account.positions
