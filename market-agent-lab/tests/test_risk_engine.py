"""Tests for the deterministic Risk Engine (Phase 8)."""

from __future__ import annotations

from datetime import datetime

from core.schemas import OrderSide, OrderType, PaperOrder, RiskApprovalStatus, RiskReasonCode
from portfolio.risk import PortfolioState, RiskEngine, RiskLimits


def _order(symbol="SYN_A", side=OrderSide.BUY, quantity=10.0, price=100.0, ts=None, strategy="v1") -> PaperOrder:
    return PaperOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        proposed_price=price,
        timestamp=ts or datetime(2023, 1, 1),
        strategy_version=strategy,
    )


def _state(**overrides) -> PortfolioState:
    defaults = dict(
        positions={},
        prices={"SYN_A": 100.0, "SYN_B": 50.0},
        position_volatility={"SYN_A": 0.2, "SYN_B": 0.2},
        equity=100_000.0,
        peak_equity=100_000.0,
        daily_start_equity=100_000.0,
        sector_map={"SYN_A": "TECH", "SYN_B": "TECH"},
        market_data_timestamp=datetime(2023, 1, 1),
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def test_approves_a_reasonable_order():
    engine = RiskEngine(RiskLimits(max_position_weight=0.5, max_gross_exposure=1.0))
    result = engine.evaluate_order(_order(quantity=10), _state())
    assert result.risk_approval_status == RiskApprovalStatus.APPROVED
    assert result.risk_reason_codes == [RiskReasonCode.OK]


def test_rejects_position_limit_breach():
    engine = RiskEngine(RiskLimits(max_position_weight=0.05))
    # 500 shares @ 100 = 50,000 = 50% of 100k equity, way over 5% limit.
    result = engine.evaluate_order(_order(quantity=500), _state())
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_POSITION_LIMIT in result.risk_reason_codes


def test_rejects_gross_exposure_breach():
    engine = RiskEngine(RiskLimits(max_position_weight=1.0, max_gross_exposure=0.05))
    result = engine.evaluate_order(_order(quantity=100), _state())  # 10,000 / 100,000 = 10% > 5%
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_GROSS_EXPOSURE in result.risk_reason_codes


def test_rejects_net_exposure_breach():
    engine = RiskEngine(RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=0.05))
    result = engine.evaluate_order(_order(side=OrderSide.BUY, quantity=100), _state())
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_NET_EXPOSURE in result.risk_reason_codes


def test_rejects_sector_concentration_breach():
    engine = RiskEngine(
        RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=1.0, max_sector_concentration=0.05)
    )
    result = engine.evaluate_order(_order(quantity=100), _state())
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_SECTOR_CONCENTRATION in result.risk_reason_codes


def test_rejects_portfolio_volatility_breach():
    engine = RiskEngine(
        RiskLimits(
            max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=1.0,
            max_sector_concentration=1.0, max_portfolio_volatility=0.01,
        )
    )
    result = engine.evaluate_order(_order(quantity=100), _state())
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_PORTFOLIO_VOLATILITY in result.risk_reason_codes


def test_daily_loss_limit_rejects_and_engages_kill_switch():
    engine = RiskEngine(RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=1.0, max_daily_loss_pct=0.02))
    state = _state(equity=97_000.0, daily_start_equity=100_000.0)  # -3% today
    result = engine.evaluate_order(_order(quantity=1), state)
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_DAILY_LOSS_LIMIT in result.risk_reason_codes
    assert engine.kill_switch_engaged is True

    # Kill switch now blocks ALL subsequent orders, even harmless ones.
    result2 = engine.evaluate_order(_order(quantity=1), _state())
    assert result2.risk_approval_status == RiskApprovalStatus.REJECTED
    assert result2.risk_reason_codes == [RiskReasonCode.KILL_SWITCH]


def test_drawdown_limit_rejects_and_engages_kill_switch():
    engine = RiskEngine(RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=1.0, max_drawdown_pct=0.10))
    state = _state(equity=85_000.0, peak_equity=100_000.0, daily_start_equity=85_000.0)  # -15% drawdown
    result = engine.evaluate_order(_order(quantity=1), state)
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.RISK_DRAWDOWN in result.risk_reason_codes
    assert engine.kill_switch_engaged is True


def test_kill_switch_can_be_reset_by_operator():
    engine = RiskEngine()
    engine.engage_kill_switch("manual test")
    assert engine.kill_switch_engaged is True
    engine.reset_kill_switch()
    assert engine.kill_switch_engaged is False
    result = engine.evaluate_order(_order(quantity=1), _state())
    assert result.risk_approval_status == RiskApprovalStatus.APPROVED


def test_stale_data_rejected():
    engine = RiskEngine(RiskLimits(stale_data_max_age_days=1))
    state = _state(market_data_timestamp=datetime(2022, 1, 1))
    result = engine.evaluate_order(_order(ts=datetime(2023, 1, 1)), state)
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.STALE_DATA in result.risk_reason_codes


def test_invalid_price_rejected():
    engine = RiskEngine()
    state = _state(prices={"SYN_A": 0.0, "SYN_B": 50.0})
    result = engine.evaluate_order(_order(symbol="SYN_A", quantity=1), state)
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.INVALID_PRICE in result.risk_reason_codes


def test_missing_price_is_invalid_price():
    engine = RiskEngine()
    state = _state(prices={"SYN_B": 50.0})
    result = engine.evaluate_order(_order(symbol="SYN_A", quantity=1), state)
    assert result.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.INVALID_PRICE in result.risk_reason_codes


def test_duplicate_order_rejected_after_approval():
    engine = RiskEngine(RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=1.0))
    ts = datetime(2023, 1, 1)
    order = _order(quantity=1, ts=ts, strategy="stratA")
    first = engine.evaluate_order(order, _state())
    assert first.risk_approval_status == RiskApprovalStatus.APPROVED

    duplicate = _order(quantity=1, ts=ts, strategy="stratA")
    second = engine.evaluate_order(duplicate, _state())
    assert second.risk_approval_status == RiskApprovalStatus.REJECTED
    assert RiskReasonCode.DUPLICATE_ORDER in second.risk_reason_codes


def test_reset_duplicate_ledger_allows_resubmission():
    engine = RiskEngine(RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, max_net_exposure=1.0))
    ts = datetime(2023, 1, 1)
    order = _order(quantity=1, ts=ts, strategy="stratA")
    engine.evaluate_order(order, _state())
    engine.reset_duplicate_ledger()
    result = engine.evaluate_order(order, _state())
    assert result.risk_approval_status == RiskApprovalStatus.APPROVED


def test_original_order_object_is_never_mutated():
    engine = RiskEngine()
    order = _order(quantity=1)
    assert order.risk_approval_status == RiskApprovalStatus.PENDING
    engine.evaluate_order(order, _state())
    assert order.risk_approval_status == RiskApprovalStatus.PENDING  # unchanged
