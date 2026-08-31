"""Daily-frequency backtesting engine (Phase 6).

The primary ML-driven strategy is simulated *event-by-event*: for every
trading day, predictions -> Portfolio Decision Engine -> Risk Engine ->
Paper Execution Engine, in that exact order, exercising the real
production code paths (not a shortcut vectorised approximation). Fills
happen with a one-day delay against the next bar's open, and commissions
/ spreads / slippage / partial fills / rejections all come from
``execution/fills.py``.

Three simpler *vectorised* comparison benchmarks (buy-and-hold,
equal-weight rebalanced, and a simple momentum baseline) are computed
separately in this module for context -- they don't need agents, a model,
or risk gating, only a consistent transaction-cost assumption (see
``backtesting/costs.py``) so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import pandas as pd

from backtesting.costs import apply_cost_drag
from core.schemas import ModelPrediction, OrderSide, PaperFill, PaperOrder
from database import repository as repo
from execution.fills import FillConfig
from execution.paper import PaperBroker
from models.predict import predict_one
from portfolio.allocation import AllocationConfig, build_paper_orders, compute_target_weights
from portfolio.risk import RiskEngine, RiskLimits


@dataclass
class BacktestResult:
    run_id: str
    equity_curve: pd.Series
    snapshots: pd.DataFrame
    trade_pnls: list[float]
    traded_notional: pd.Series
    gross_exposure_series: pd.Series
    holding_periods: list[float]
    predictions: list[ModelPrediction]
    fills: list[PaperFill]
    rejected_orders: list[PaperOrder]
    broker: PaperBroker = field(repr=False)


def _fill_pnl_from_snapshot(
    prev_qty: float, prev_cost: float, fill: PaperFill, order: PaperOrder
) -> float | None:
    """Realised P&L delta attributable to this single fill, given the
    position/avg-cost snapshot from *before* the fill was applied (None if
    it was purely opening/adding to a position, i.e. no round-trip yet).

    ``PaperBroker.submit_and_fill`` already applies every fill internally
    (see ``execution/paper.py``), so this must be computed from a snapshot
    captured *before* that call -- reading ``broker.account`` afterwards
    would see already-updated state and silently double-count.
    """
    side_sign = 1.0 if order.side == OrderSide.BUY else -1.0
    trade_qty = side_sign * fill.quantity
    if prev_qty == 0.0 or (prev_qty > 0) == (trade_qty > 0):
        return None
    closing_qty = min(abs(trade_qty), abs(prev_qty))
    direction = 1.0 if prev_qty > 0 else -1.0
    return (fill.fill_price - prev_cost) * closing_qty * direction - fill.commission


def run_ml_strategy_backtest(
    con,
    run_id: str,
    feature_df: pd.DataFrame,
    market_df: pd.DataFrame,
    boosters: dict[str, lgb.Booster],
    feature_cols: list[str],
    model_version: str,
    feature_version: str,
    symbols: list[str],
    sector_map: dict[str, str],
    initial_cash: float = 1_000_000.0,
    risk_limits: RiskLimits | None = None,
    allocation_config: AllocationConfig | None = None,
    fill_config: FillConfig | None = None,
    persist_predictions: bool = True,
) -> BacktestResult:
    """Run the full Research -> Model -> Portfolio -> Risk -> Execution loop, one day at a time."""
    broker = PaperBroker(initial_cash=initial_cash, fill_config=fill_config)
    risk_engine = RiskEngine(risk_limits)

    market_by_symbol = {s: market_df[market_df["symbol"] == s].set_index("timestamp").sort_index() for s in symbols}
    dates = sorted(feature_df["timestamp"].unique())

    equity_records = []
    all_predictions: list[ModelPrediction] = []
    all_fills: list[PaperFill] = []
    all_rejected: list[PaperOrder] = []
    trade_pnls: list[float] = []
    traded_notional_by_date: dict[pd.Timestamp, float] = {}
    gross_exposure_by_date: dict[pd.Timestamp, float] = {}
    position_open_date: dict[str, pd.Timestamp] = {}
    holding_periods: list[float] = []

    for i, date in enumerate(dates[:-1]):  # need a next bar to execute against
        next_date = dates[i + 1]
        day_features = feature_df[feature_df["timestamp"] == date]

        decision_prices, position_vol, execution_prices, bar_volumes = {}, {}, {}, {}
        for symbol in symbols:
            hist = market_by_symbol[symbol]
            if date in hist.index:
                decision_prices[symbol] = float(hist.loc[date, "close"])
            if next_date in hist.index:
                execution_prices[symbol] = float(hist.loc[next_date, "open"])
                bar_volumes[symbol] = float(hist.loc[next_date, "volume"])
            row = day_features[day_features["symbol"] == symbol]
            if not row.empty:
                vol = row.iloc[0].get("raw_realised_vol_20d")
                if vol == vol and vol is not None:
                    position_vol[symbol] = float(vol)

        broker.start_new_day(decision_prices)

        predictions = []
        for _, row in day_features.iterrows():
            if row[feature_cols].isna().any():
                continue
            pred = predict_one(boosters, row, feature_cols, row["symbol"], date, model_version, feature_version)
            predictions.append(pred)
        all_predictions.extend(predictions)

        weights = compute_target_weights(predictions, allocation_config)
        orders = build_paper_orders(
            weights, broker.account.positions, decision_prices, broker.equity(decision_prices),
            timestamp=pd.Timestamp(date).to_pydatetime(), strategy_version=model_version, config=allocation_config,
        )

        # Snapshot positions/avg-cost BEFORE submit_and_fill -- it applies
        # every fill internally, so this is the only correct "previous
        # state" to diff P&L and holding-period transitions against.
        pre_batch_positions = dict(broker.account.positions)
        pre_batch_avg_cost = dict(broker.account.avg_cost)

        result = broker.submit_and_fill(
            orders=orders,
            risk_engine=risk_engine,
            decision_prices=decision_prices,
            execution_prices=execution_prices,
            bar_volumes=bar_volumes,
            fill_timestamp=pd.Timestamp(next_date).to_pydatetime(),
            sector_map=sector_map,
            position_volatility=position_vol,
            market_data_timestamp=pd.Timestamp(date).to_pydatetime(),
        )
        all_rejected.extend(result.rejected_orders)

        day_notional = 0.0
        for order in result.approved_orders:
            matching_fill = next((f for f in result.fills if f.order_id == order.order_id), None)
            if matching_fill is None:
                continue
            prev_qty = pre_batch_positions.get(order.symbol, 0.0)
            prev_cost = pre_batch_avg_cost.get(order.symbol, 0.0)
            pnl = _fill_pnl_from_snapshot(prev_qty, prev_cost, matching_fill, order)
            if pnl is not None:
                trade_pnls.append(pnl)
            day_notional += matching_fill.fill_price * matching_fill.quantity
            all_fills.append(matching_fill)

            was_flat = prev_qty == 0.0
            now_flat = order.symbol not in broker.account.positions
            if was_flat and not now_flat:
                position_open_date[order.symbol] = pd.Timestamp(next_date)
            elif now_flat and order.symbol in position_open_date:
                held = (pd.Timestamp(next_date) - position_open_date.pop(order.symbol)).days
                holding_periods.append(float(held))

        traded_notional_by_date[next_date] = traded_notional_by_date.get(next_date, 0.0) + day_notional

        mark_prices = {**decision_prices, **execution_prices}
        equity_now = broker.equity(mark_prices)
        gross_exposure_by_date[next_date] = broker.gross_exposure(mark_prices)
        equity_records.append(
            {
                "timestamp": next_date,
                "cash": broker.account.cash,
                "equity": equity_now,
                "realized_pnl": broker.account.realized_pnl,
                "unrealized_pnl": broker.unrealized_pnl(mark_prices),
                "gross_exposure": gross_exposure_by_date[next_date],
                "net_exposure": broker.net_exposure(mark_prices),
                "drawdown": (equity_now - broker.peak_equity) / broker.peak_equity if broker.peak_equity else 0.0,
            }
        )

        if persist_predictions:
            for pred in predictions:
                try:
                    repo.insert_prediction(con, pred)
                except ValueError:
                    pass  # already recorded (e.g. re-run of the same demo) -- immutability preserved
        for order in result.approved_orders:
            repo.insert_paper_order(con, order)
            repo.insert_risk_decision(con, order.order_id, order.timestamp, order.symbol, order.risk_approval_status, [c.value for c in order.risk_reason_codes])
        for order in result.rejected_orders:
            repo.insert_paper_order(con, order)
            repo.insert_risk_decision(con, order.order_id, order.timestamp, order.symbol, order.risk_approval_status, [c.value for c in order.risk_reason_codes])
        for fill in result.fills:
            repo.insert_paper_fill(con, fill)

    snapshots = pd.DataFrame(equity_records)
    for _, snap in snapshots.iterrows():
        repo.insert_portfolio_snapshot(con, run_id, snap.to_dict())

    equity_curve = snapshots.set_index("timestamp")["equity"] if not snapshots.empty else pd.Series(dtype=float)
    traded_notional = pd.Series(traded_notional_by_date).sort_index()
    gross_exposure_series = pd.Series(gross_exposure_by_date).sort_index()

    return BacktestResult(
        run_id=run_id,
        equity_curve=equity_curve,
        snapshots=snapshots,
        trade_pnls=trade_pnls,
        traded_notional=traded_notional,
        gross_exposure_series=gross_exposure_series,
        holding_periods=holding_periods,
        predictions=all_predictions,
        fills=all_fills,
        rejected_orders=all_rejected,
        broker=broker,
    )


# --------------------------------------------------------------------------
# Vectorised comparison benchmarks
# --------------------------------------------------------------------------


def _wide_close(market_df: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    wide = market_df[market_df["symbol"].isin(symbols)].pivot(index="timestamp", columns="symbol", values="close")
    return wide.sort_index()


def buy_and_hold_benchmark(market_df: pd.DataFrame, symbols: list[str], initial_cash: float = 1_000_000.0) -> pd.Series:
    """Equal-dollar allocation at t0, held with no rebalancing."""
    close = _wide_close(market_df, symbols).dropna(how="all")
    weights0 = pd.Series(1.0 / len(symbols), index=close.columns)
    shares = (initial_cash * weights0) / close.iloc[0]
    equity = (close * shares).sum(axis=1)
    return equity


def equal_weight_benchmark(
    market_df: pd.DataFrame, symbols: list[str], initial_cash: float = 1_000_000.0, rebalance_freq: str = "W-FRI"
) -> pd.Series:
    """Equal-weight portfolio, periodically rebalanced back to equal weights."""
    close = _wide_close(market_df, symbols).dropna(how="all")
    returns = close.pct_change().fillna(0.0)
    rebalance_dates = set(close.resample(rebalance_freq).last().index)

    equity = initial_cash
    weights = pd.Series(1.0 / len(symbols), index=close.columns)
    equity_curve = []
    turnover_series = []
    prev_weights = weights.copy()
    for date in close.index:
        day_return = (weights * returns.loc[date]).sum()
        equity *= 1 + day_return
        equity_curve.append(equity)
        if date in rebalance_dates:
            turnover_series.append(float((weights - prev_weights).abs().sum()))
            prev_weights = weights.copy()
        else:
            turnover_series.append(0.0)
    equity_series = pd.Series(equity_curve, index=close.index)
    gross_returns = equity_series.pct_change().fillna(0.0)
    net_returns = apply_cost_drag(gross_returns, pd.Series(turnover_series, index=close.index))
    return initial_cash * (1 + net_returns).cumprod()


def momentum_benchmark(
    market_df: pd.DataFrame,
    symbols: list[str],
    initial_cash: float = 1_000_000.0,
    lookback: int = 20,
    top_n: int = 3,
    rebalance_freq: str = "W-FRI",
) -> pd.Series:
    """Simple cross-sectional momentum: go long the top-N trailing-return
    symbols, equal-weighted, rebalanced periodically."""
    close = _wide_close(market_df, symbols).dropna(how="all")
    trailing_return = close.pct_change(lookback)
    returns = close.pct_change().fillna(0.0)
    rebalance_dates = set(close.resample(rebalance_freq).last().index)

    equity = initial_cash
    weights = pd.Series(0.0, index=close.columns)
    equity_curve = []
    turnover_series = []
    for date in close.index:
        day_return = (weights * returns.loc[date]).sum()
        equity *= 1 + day_return
        equity_curve.append(equity)
        if date in rebalance_dates and trailing_return.loc[date].notna().sum() >= top_n:
            top = trailing_return.loc[date].nlargest(top_n).index
            new_weights = pd.Series(0.0, index=close.columns)
            new_weights[top] = 1.0 / top_n
            turnover_series.append(float((new_weights - weights).abs().sum()))
            weights = new_weights
        else:
            turnover_series.append(0.0)
    equity_series = pd.Series(equity_curve, index=close.index)
    gross_returns = equity_series.pct_change().fillna(0.0)
    net_returns = apply_cost_drag(gross_returns, pd.Series(turnover_series, index=close.index))
    return initial_cash * (1 + net_returns).cumprod()
