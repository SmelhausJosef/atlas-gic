from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import copysign

from atlas.backtest.costs import apply_execution_price
from atlas.common.models import TradeRecord


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass(slots=True)
class PendingOrder:
    execute_index: int
    target_position: float
    confidence: float
    reason: str
    tags: list[str]


class Portfolio:
    def __init__(
        self,
        initial_equity: float,
        max_leverage: float,
        liquidation_buffer_pct: float,
    ):
        self.cash = initial_equity
        self.quantity = 0.0
        self.max_leverage = max_leverage
        self.liquidation_buffer_pct = liquidation_buffer_pct
        self.completed_trades: list[TradeRecord] = []
        self.entry_price: float | None = None
        self.entry_timestamp: datetime | None = None
        self.entry_bar_index: int | None = None
        self.entry_confidence = 0.0
        self.entry_reason = ""
        self.entry_tags: list[str] = []
        self.last_exit_bar_index: int | None = None
        self.liquidated = False

    def equity_at(self, price: float) -> float:
        return self.cash + self.quantity * price

    def gross_exposure(self, price: float) -> float:
        equity = max(self.equity_at(price), 1e-9)
        return abs(self.quantity * price) / equity

    def bars_since_entry(self, current_index: int) -> int | None:
        if self.entry_bar_index is None:
            return None
        return current_index - self.entry_bar_index

    def bars_since_exit(self, current_index: int) -> int | None:
        if self.last_exit_bar_index is None:
            return None
        return current_index - self.last_exit_bar_index

    @property
    def current_position(self) -> float:
        return float(max(-1.0, min(1.0, _sign(self.quantity))))

    def apply_funding(self, price: float, funding_rate: float) -> None:
        if self.quantity == 0:
            return
        payment = self.quantity * price * funding_rate
        self.cash -= payment
        self._check_liquidation(price)

    def execute_target(
        self,
        target_position: float,
        price: float,
        timestamp: datetime,
        bar_index: int,
        total_cost_bps: float,
        confidence: float,
        reason: str,
        tags: list[str],
    ) -> None:
        if self.liquidated:
            return
        current_equity = max(self.equity_at(price), 1e-9)
        desired_notional = target_position * self.max_leverage * current_equity
        desired_quantity = desired_notional / price if price else 0.0
        current_sign = _sign(self.quantity)
        desired_sign = _sign(desired_quantity)

        if current_sign != 0 and current_sign == desired_sign:
            return

        if current_sign != 0 and current_sign != desired_sign:
            self._close_position(price, timestamp, bar_index, total_cost_bps)

        if desired_sign == 0:
            return

        if self.quantity != 0:
            self._close_position(price, timestamp, bar_index, total_cost_bps)

        trade_qty = desired_quantity - self.quantity
        if abs(trade_qty) < 1e-12:
            return
        side = _sign(trade_qty)
        execution_price = apply_execution_price(price, side, total_cost_bps)
        self.cash -= trade_qty * execution_price
        self.quantity = desired_quantity
        self.entry_price = execution_price
        self.entry_timestamp = timestamp
        self.entry_bar_index = bar_index
        self.entry_confidence = confidence
        self.entry_reason = reason
        self.entry_tags = tags
        self._check_liquidation(price)

    def force_close(self, price: float, timestamp: datetime, bar_index: int, total_cost_bps: float) -> None:
        self._close_position(price, timestamp, bar_index, total_cost_bps)

    def _close_position(self, price: float, timestamp: datetime, bar_index: int, total_cost_bps: float) -> None:
        if self.quantity == 0 or self.entry_price is None or self.entry_timestamp is None or self.entry_bar_index is None:
            return

        close_qty = -self.quantity
        side = _sign(close_qty)
        execution_price = apply_execution_price(price, side, total_cost_bps)
        self.cash -= close_qty * execution_price

        pnl = self.quantity * (execution_price - self.entry_price)
        notional = abs(self.entry_price * self.quantity)
        pnl_pct = pnl / notional if notional else 0.0
        direction = "LONG" if self.quantity > 0 else "SHORT"

        self.completed_trades.append(
            TradeRecord(
                entry_timestamp=self.entry_timestamp,
                exit_timestamp=timestamp,
                direction=direction,
                quantity=abs(self.quantity),
                entry_price=self.entry_price,
                exit_price=execution_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_bars=bar_index - self.entry_bar_index,
                confidence=self.entry_confidence,
                reason=self.entry_reason,
                tags=list(self.entry_tags),
            )
        )
        self.quantity = 0.0
        self.entry_price = None
        self.entry_timestamp = None
        self.entry_bar_index = None
        self.entry_confidence = 0.0
        self.entry_reason = ""
        self.entry_tags = []
        self.last_exit_bar_index = bar_index
        self._check_liquidation(price)

    def _check_liquidation(self, price: float) -> None:
        equity = self.equity_at(price)
        gross_notional = abs(self.quantity * price)
        if gross_notional == 0:
            self.liquidated = False
            return
        margin_ratio = equity / gross_notional
        if margin_ratio <= self.liquidation_buffer_pct:
            self.liquidated = True
