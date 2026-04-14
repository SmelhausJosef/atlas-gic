from __future__ import annotations

from copy import deepcopy

import pandas as pd

from atlas.backtest.portfolio import PendingOrder, Portfolio
from atlas.common.artifacts import build_run_id
from atlas.common.config import AppConfig
from atlas.common.models import BacktestResult, DatasetManifest, StrategyContext
from atlas.evaluation.metrics import calculate_metrics
from atlas.strategies.base import BaseStrategy


class BacktestEngine:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(
        self,
        frame: pd.DataFrame,
        manifest: DatasetManifest,
        strategy: BaseStrategy,
        strategy_name: str,
    ) -> BacktestResult:
        portfolio = Portfolio(
            initial_equity=self.config.backtest.initial_equity,
            max_leverage=self.config.risk.max_leverage,
            liquidation_buffer_pct=self.config.risk.liquidation_buffer_pct,
        )
        total_cost_bps = self.config.costs.total_bps
        pending_order: PendingOrder | None = None
        equity_rows: list[dict[str, float | str | int]] = []
        start_index = max(
            self.config.strategy.lookback_bars,
            self.config.strategy.mean_window,
            self.config.strategy.std_window,
            self.config.strategy.vol_window,
        )

        for index in range(len(frame)):
            bar = frame.iloc[index]
            if pending_order and pending_order.execute_index == index:
                portfolio.execute_target(
                    target_position=pending_order.target_position,
                    price=float(bar["close"]),
                    timestamp=bar["timestamp"].to_pydatetime(),
                    bar_index=index,
                    total_cost_bps=total_cost_bps,
                    confidence=pending_order.confidence,
                    reason=pending_order.reason,
                    tags=pending_order.tags,
                )
                pending_order = None

            portfolio.apply_funding(float(bar["close"]), float(bar["funding_rate"]))
            equity = portfolio.equity_at(float(bar["close"]))
            equity_rows.append(
                {
                    "timestamp": bar["timestamp"],
                    "close": float(bar["close"]),
                    "equity": equity,
                    "position": portfolio.current_position,
                    "gross_exposure": portfolio.gross_exposure(float(bar["close"])) if equity > 0 else 0.0,
                    "liquidated": int(portfolio.liquidated),
                }
            )
            if portfolio.liquidated:
                break

            if index >= start_index and index < len(frame) - 1:
                context = StrategyContext(
                    bars=frame.iloc[max(0, index - self.config.strategy.lookback_bars + 1) : index + 1].copy(),
                    current_position=portfolio.current_position,
                    gross_exposure=portfolio.gross_exposure(float(bar["close"])) if equity > 0 else 0.0,
                    equity=equity,
                    bars_since_entry=portfolio.bars_since_entry(index),
                    bars_since_exit=portfolio.bars_since_exit(index),
                    expected_bar_seconds=self.config.data.expected_bar_seconds,
                )
                decision = strategy.generate(context)
                pending_order = PendingOrder(
                    execute_index=index + 1,
                    target_position=decision.target_position,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    tags=decision.tags,
                )

        final_bar = frame.iloc[min(len(equity_rows) - 1, len(frame) - 1)]
        portfolio.force_close(
            price=float(final_bar["close"]),
            timestamp=final_bar["timestamp"].to_pydatetime(),
            bar_index=min(len(equity_rows) - 1, len(frame) - 1),
            total_cost_bps=total_cost_bps,
        )

        equity_curve = pd.DataFrame(equity_rows)
        metrics = calculate_metrics(
            equity_curve=equity_curve,
            trades=portfolio.completed_trades,
            periods_per_year=self.config.backtest.periods_per_year,
            expected_bar_seconds=self.config.data.expected_bar_seconds,
        )
        return BacktestResult(
            run_id=build_run_id(strategy_name, manifest.sha256),
            strategy_name=strategy_name,
            manifest=manifest,
            metrics=metrics,
            trades=deepcopy(portfolio.completed_trades),
            equity_curve=equity_curve,
            execution_assumptions="signal on bar N, execution at close of bar N+1, conservative next-bar-close approximation",
        )

