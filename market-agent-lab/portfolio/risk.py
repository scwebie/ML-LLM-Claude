"""Deterministic Risk Engine (Phase 8).

This is the single hard gate between the Portfolio Decision Engine and the
Paper Execution Engine. It is 100% deterministic arithmetic -- no LLM, no
agent, and no model output can bypass or override it. Every proposed
order returns either APPROVED or REJECTED with one or more explicit
:class:`~core.schemas.RiskReasonCode` values; there is no silent partial
approval.

Order of checks (first failing check wins, short-circuiting the rest):

1. ``KILL_SWITCH``            -- trading halted entirely
2. ``STALE_DATA``              -- market data older than allowed
3. ``INVALID_PRICE``           -- non-finite / non-positive reference price
4. ``DUPLICATE_ORDER``         -- an identical order was already approved
5. ``RISK_POSITION_LIMIT``     -- resulting position weight too large
6. ``RISK_GROSS_EXPOSURE``     -- resulting gross exposure too large
7. ``RISK_NET_EXPOSURE``       -- resulting net exposure too large
8. ``RISK_SECTOR_CONCENTRATION`` -- resulting sector weight too large
9. ``RISK_PORTFOLIO_VOLATILITY`` -- estimated portfolio vol too large
10. ``RISK_DAILY_LOSS_LIMIT``   -- today's realised+unrealised loss too large
11. ``RISK_DRAWDOWN``           -- drawdown from peak equity too large

A daily-loss or drawdown breach also engages the kill switch automatically
(defense in depth): once either fires, every subsequent order is rejected
with ``KILL_SWITCH`` until a human/operator explicitly resets it via
:meth:`RiskEngine.reset_kill_switch`. Agents and the ML model have no
access to that method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.logging import get_logger
from core.schemas import OrderSide, PaperOrder, RiskApprovalStatus, RiskReasonCode

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskLimits:
    max_position_weight: float = 0.10
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 0.60
    max_sector_concentration: float = 0.35
    max_portfolio_volatility: float = 0.30
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    stale_data_max_age_days: int = 3


@dataclass
class PortfolioState:
    """Minimal snapshot of account/portfolio state the risk engine needs.

    Built fresh by the caller (the Paper Execution Engine) from its own
    account bookkeeping before every risk check -- the risk engine itself
    holds no portfolio state, only kill-switch state and the duplicate-order
    ledger, which keeps it trivially testable in isolation.
    """

    positions: dict[str, float]  # symbol -> signed quantity
    prices: dict[str, float]  # symbol -> current reference price
    position_volatility: dict[str, float]  # symbol -> predicted/realised annualised vol
    equity: float
    peak_equity: float
    daily_start_equity: float
    sector_map: dict[str, str] = field(default_factory=dict)
    market_data_timestamp: datetime | None = None


def _position_value(state: PortfolioState, symbol: str, extra_qty: float = 0.0, side: OrderSide | None = None) -> float:
    qty = state.positions.get(symbol, 0.0)
    if side == OrderSide.SELL:
        qty -= extra_qty
    elif side == OrderSide.BUY:
        qty += extra_qty
    price = state.prices.get(symbol, 0.0)
    return qty * price


@dataclass(frozen=True)
class _HypotheticalExposures:
    gross: float
    net: float
    sector_exposure: dict[str, float]
    position_values: dict[str, float]


def _portfolio_exposures(state: PortfolioState, order: PaperOrder) -> _HypotheticalExposures:
    """Gross/net exposure and per-sector exposure AFTER hypothetically applying ``order``."""
    hypothetical_values: dict[str, float] = {}
    for symbol in set(state.positions) | {order.symbol}:
        extra_qty = order.quantity if symbol == order.symbol else 0.0
        side = order.side if symbol == order.symbol else None
        hypothetical_values[symbol] = _position_value(state, symbol, extra_qty, side)

    equity = state.equity if state.equity > 0 else 1e-9
    gross = sum(abs(v) for v in hypothetical_values.values()) / equity
    net = sum(hypothetical_values.values()) / equity

    sector_exposure: dict[str, float] = {}
    for symbol, value in hypothetical_values.items():
        sector = state.sector_map.get(symbol, "UNKNOWN")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(value) / equity

    return _HypotheticalExposures(gross=gross, net=net, sector_exposure=sector_exposure, position_values=hypothetical_values)


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._kill_switch_engaged = False
        self._kill_switch_reason: str | None = None
        self._approved_order_keys: set[tuple] = set()

    # --- kill switch -------------------------------------------------------
    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch_engaged

    def engage_kill_switch(self, reason: str) -> None:
        self._kill_switch_engaged = True
        self._kill_switch_reason = reason
        logger.warning("kill_switch_engaged", reason=reason)

    def reset_kill_switch(self) -> None:
        """Explicit operator action. Never called by agents, the model, or
        the portfolio engine -- only by a human operator / the API's admin
        endpoint."""
        self._kill_switch_engaged = False
        self._kill_switch_reason = None
        logger.info("kill_switch_reset")

    def reset_duplicate_ledger(self) -> None:
        self._approved_order_keys = set()

    @staticmethod
    def _order_key(order: PaperOrder) -> tuple:
        return (order.symbol, order.side.value, order.timestamp, order.strategy_version)

    # --- main entry point ----------------------------------------------------
    def evaluate_order(self, order: PaperOrder, state: PortfolioState) -> PaperOrder:
        reasons = self._evaluate_reasons(order, state)
        status = RiskApprovalStatus.APPROVED if not reasons else RiskApprovalStatus.REJECTED
        if status is RiskApprovalStatus.APPROVED:
            self._approved_order_keys.add(self._order_key(order))
        return order.model_copy(
            update={
                "risk_approval_status": status,
                "risk_reason_codes": reasons or [RiskReasonCode.OK],
            }
        )

    def _evaluate_reasons(self, order: PaperOrder, state: PortfolioState) -> list[RiskReasonCode]:
        if self._kill_switch_engaged:
            return [RiskReasonCode.KILL_SWITCH]

        if state.market_data_timestamp is not None:
            age = order.timestamp - state.market_data_timestamp
            if age > timedelta(days=self.limits.stale_data_max_age_days):
                return [RiskReasonCode.STALE_DATA]

        price = state.prices.get(order.symbol)
        if price is None or not math.isfinite(price) or price <= 0:
            return [RiskReasonCode.INVALID_PRICE]

        if self._order_key(order) in self._approved_order_keys:
            return [RiskReasonCode.DUPLICATE_ORDER]

        exposures = _portfolio_exposures(state, order)
        equity = state.equity if state.equity > 0 else 1e-9

        position_weight = abs(exposures.position_values.get(order.symbol, 0.0)) / equity
        if position_weight > self.limits.max_position_weight:
            return [RiskReasonCode.RISK_POSITION_LIMIT]

        if exposures.gross > self.limits.max_gross_exposure:
            return [RiskReasonCode.RISK_GROSS_EXPOSURE]

        if abs(exposures.net) > self.limits.max_net_exposure:
            return [RiskReasonCode.RISK_NET_EXPOSURE]

        sector = state.sector_map.get(order.symbol, "UNKNOWN")
        if exposures.sector_exposure.get(sector, 0.0) > self.limits.max_sector_concentration:
            return [RiskReasonCode.RISK_SECTOR_CONCENTRATION]

        est_portfolio_vol = self._estimate_portfolio_volatility(state, exposures, equity)
        if est_portfolio_vol > self.limits.max_portfolio_volatility:
            return [RiskReasonCode.RISK_PORTFOLIO_VOLATILITY]

        if state.daily_start_equity > 0:
            daily_pnl_pct = (state.equity - state.daily_start_equity) / state.daily_start_equity
            if daily_pnl_pct <= -self.limits.max_daily_loss_pct:
                self.engage_kill_switch(f"daily loss limit breached ({daily_pnl_pct:.2%})")
                return [RiskReasonCode.RISK_DAILY_LOSS_LIMIT]

        if state.peak_equity > 0:
            drawdown_pct = (state.peak_equity - state.equity) / state.peak_equity
            if drawdown_pct >= self.limits.max_drawdown_pct:
                self.engage_kill_switch(f"drawdown limit breached ({drawdown_pct:.2%})")
                return [RiskReasonCode.RISK_DRAWDOWN]

        return []

    @staticmethod
    def _estimate_portfolio_volatility(state: PortfolioState, exposures: _HypotheticalExposures, equity: float) -> float:
        """Simple, deliberately conservative estimate: weight-squared sum of
        per-position volatilities assuming zero cross-correlation benefit is
        NOT assumed -- i.e. we use a straight weighted-average vol (an upper
        bound relative to a diversified portfolio), which is the appropriately
        conservative simplification for a v0.1 deterministic risk gate."""
        total_weighted_vol = 0.0
        for symbol, value in exposures.position_values.items():
            weight = abs(value) / equity
            vol = state.position_volatility.get(symbol, 0.20)
            total_weighted_vol += weight * vol
        return total_weighted_vol
