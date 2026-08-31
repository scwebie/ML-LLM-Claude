"""Paper Execution Engine (Phase 9): an entirely internal simulated broker.

DOES NOT connect to any live brokerage, exchange, or prediction-market /
betting platform -- there is no network client in this module at all.
It owns its own account state (cash, positions, realised/unrealised P&L)
and is the only place order fills are ever applied.

Decision vs. execution timing: risk evaluation uses prices/timestamps
known *at decision time* (e.g. today's close); the resulting approved
orders are then filled against a *separate*, later execution price (e.g.
next bar's open) via :mod:`execution.fills`, which is what gives Version
0.1 its "delayed execution" simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from core.schemas import OrderSide, PaperFill, PaperOrder
from execution.fills import FillConfig, simulate_fill
from execution.orders import evaluate_and_route
from portfolio.risk import PortfolioState, RiskEngine


@dataclass
class AccountState:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    avg_cost: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0


@dataclass
class ExecutionBatchResult:
    approved_orders: list[PaperOrder]
    rejected_orders: list[PaperOrder]
    fills: list[PaperFill]
    broker_rejections: list[tuple[PaperOrder, str]]


class PaperBroker:
    """Simulated broker: market/limit orders, partial fills, spreads,
    slippage, commissions, and rejected orders -- all in-memory, no real
    money and no live market connectivity anywhere in this class."""

    def __init__(self, initial_cash: float = 1_000_000.0, fill_config: FillConfig | None = None) -> None:
        self.account = AccountState(cash=initial_cash)
        self.fill_config = fill_config or FillConfig()
        self.initial_cash = initial_cash
        self.peak_equity = initial_cash
        self.daily_start_equity = initial_cash
        self._last_known_prices: dict[str, float] = {}

    # --- accounting ------------------------------------------------------------
    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        market_value = 0.0
        for symbol, qty in self.account.positions.items():
            price = prices.get(symbol, self._last_known_prices.get(symbol))
            if price is not None:
                market_value += qty * price
        return self.account.cash + market_value

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, qty in self.account.positions.items():
            price = prices.get(symbol, self._last_known_prices.get(symbol))
            cost = self.account.avg_cost.get(symbol, 0.0)
            if price is not None:
                total += (price - cost) * qty
        return total

    def gross_exposure(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        return sum(abs(qty) * prices.get(symbol, self._last_known_prices.get(symbol, 0.0)) for symbol, qty in self.account.positions.items()) / eq

    def net_exposure(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        return sum(qty * prices.get(symbol, self._last_known_prices.get(symbol, 0.0)) for symbol, qty in self.account.positions.items()) / eq

    def start_new_day(self, prices: dict[str, float]) -> None:
        self.daily_start_equity = self.equity(prices)
        self.peak_equity = max(self.peak_equity, self.daily_start_equity)

    def apply_fill(self, fill: PaperFill, order: PaperOrder) -> None:
        symbol = order.symbol
        side_sign = 1.0 if order.side == OrderSide.BUY else -1.0
        trade_qty = side_sign * fill.quantity

        prev_qty = self.account.positions.get(symbol, 0.0)
        prev_cost = self.account.avg_cost.get(symbol, 0.0)
        new_qty = prev_qty + trade_qty

        if prev_qty == 0.0 or (prev_qty > 0) == (trade_qty > 0):
            if new_qty != 0.0:
                self.account.avg_cost[symbol] = (
                    prev_cost * abs(prev_qty) + fill.fill_price * abs(trade_qty)
                ) / abs(new_qty)
        else:
            closing_qty = min(abs(trade_qty), abs(prev_qty))
            direction = 1.0 if prev_qty > 0 else -1.0
            self.account.realized_pnl += (fill.fill_price - prev_cost) * closing_qty * direction
            if abs(new_qty) < 1e-9:
                new_qty = 0.0
                self.account.avg_cost[symbol] = 0.0
            elif (new_qty > 0) != (prev_qty > 0):
                self.account.avg_cost[symbol] = fill.fill_price

        if new_qty == 0.0:
            self.account.positions.pop(symbol, None)
        else:
            self.account.positions[symbol] = new_qty

        if order.side == OrderSide.BUY:
            self.account.cash -= fill.fill_price * fill.quantity + fill.commission
        else:
            self.account.cash += fill.fill_price * fill.quantity - fill.commission
        self.account.realized_pnl -= fill.commission

    # --- the actual pipeline step ------------------------------------------------
    def submit_and_fill(
        self,
        orders: list[PaperOrder],
        risk_engine: RiskEngine,
        decision_prices: dict[str, float],
        execution_prices: dict[str, float],
        bar_volumes: dict[str, float],
        fill_timestamp: datetime,
        sector_map: dict[str, str],
        position_volatility: dict[str, float],
        market_data_timestamp: datetime,
    ) -> ExecutionBatchResult:
        state = PortfolioState(
            positions=dict(self.account.positions),
            prices=decision_prices,
            position_volatility=position_volatility,
            equity=self.equity(decision_prices),
            peak_equity=self.peak_equity,
            daily_start_equity=self.daily_start_equity,
            sector_map=sector_map,
            market_data_timestamp=market_data_timestamp,
        )
        approved, rejected = evaluate_and_route(orders, risk_engine, state)

        fills: list[PaperFill] = []
        broker_rejections: list[tuple[PaperOrder, str]] = []
        for order in approved:
            execution_price = execution_prices.get(order.symbol)
            if execution_price is None:
                broker_rejections.append((order, "NO_EXECUTION_PRICE"))
                continue
            available_bp = self.account.cash if order.side == OrderSide.BUY else None
            fill, reason = simulate_fill(
                order,
                execution_price=execution_price,
                bar_volume=bar_volumes.get(order.symbol, 0.0),
                fill_timestamp=fill_timestamp,
                config=self.fill_config,
                available_buying_power=available_bp,
            )
            if fill is None:
                broker_rejections.append((order, reason or "UNKNOWN"))
                continue
            self.apply_fill(fill, order)
            fills.append(fill)

        self._last_known_prices.update(decision_prices)
        self._last_known_prices.update(execution_prices)
        self.peak_equity = max(self.peak_equity, self.equity(execution_prices))

        return ExecutionBatchResult(
            approved_orders=approved, rejected_orders=rejected, fills=fills, broker_rejections=broker_rejections
        )
