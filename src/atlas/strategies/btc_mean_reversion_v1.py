from __future__ import annotations

import math

import pandas as pd

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.base import BaseStrategy


class BTCMeanReversionV1(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self._active_target_position = 0.0

    def _flat_decision(self, reason: str, tags: list[str]) -> StrategyDecision:
        self._active_target_position = 0.0
        return StrategyDecision(target_position=0.0, confidence=0.0, reason=reason, tags=tags)

    def _position_size(self) -> float:
        position_size = float(getattr(self.config, "position_size", 1.0))
        return max(0.0, min(1.0, position_size))

    def generate(self, context: StrategyContext) -> StrategyDecision:
        bars = context.bars
        trend_window = int(getattr(self.config, "trend_window", self.config.lookback_bars))
        momentum_window = int(getattr(self.config, "momentum_window", 12))
        min_holding_bars = int(getattr(self.config, "min_holding_bars", 0))
        max_volatility = float(getattr(self.config, "max_volatility", 1.0))
        max_regime_distance = float(getattr(self.config, "max_regime_distance", 1.0))
        max_trend_deviation = float(getattr(self.config, "max_trend_deviation", 1.0))
        entry_momentum_threshold = float(getattr(self.config, "entry_momentum_threshold", 0.0))
        stop_zscore = float(getattr(self.config, "stop_zscore", self.config.entry_zscore * 1.75))

        required_bars = max(
            self.config.lookback_bars,
            self.config.mean_window,
            self.config.std_window,
            self.config.vol_window,
            trend_window,
            momentum_window + 1,
        )
        if len(bars) < required_bars:
            return self._flat_decision("warmup", ["warmup"])

        close = bars["close"].astype(float)
        returns = close.pct_change().fillna(0.0)
        current_price = float(close.iloc[-1])
        fast_mean = float(close.tail(self.config.mean_window).mean())
        slow_mean = float(close.tail(trend_window).mean())
        std = float(close.tail(self.config.std_window).std(ddof=0))
        realized_vol = float(returns.tail(self.config.vol_window).std(ddof=0))
        if std == 0 or math.isnan(std):
            return self._flat_decision("invalid_dispersion", ["filter"])
        if slow_mean == 0:
            return self._flat_decision("invalid_slow_mean", ["filter"])

        zscore = (current_price - fast_mean) / std
        trend_deviation = (current_price - slow_mean) / slow_mean
        regime_distance = (fast_mean - slow_mean) / slow_mean
        anchor_price = float(close.iloc[-(momentum_window + 1)])
        recent_return = current_price / anchor_price - 1.0 if anchor_price else 0.0
        position_size = self._position_size()
        if context.current_position == 0:
            self._active_target_position = 0.0
            if realized_vol < self.config.min_volatility or realized_vol > max_volatility:
                return self._flat_decision("volatility_regime", ["filter", "volatility"])
            if abs(trend_deviation) > max_trend_deviation or abs(regime_distance) > max_regime_distance:
                return self._flat_decision("trend_regime_filter", ["filter", "trend"])
            cooldown_ok = context.bars_since_exit is None or context.bars_since_exit >= self.config.cooldown_bars
            if not cooldown_ok:
                return self._flat_decision("cooldown", ["cooldown"])
            if zscore >= self.config.entry_zscore and recent_return >= entry_momentum_threshold:
                confidence = min(1.0, abs(zscore) / (self.config.entry_zscore * 2))
                self._active_target_position = -position_size
                return StrategyDecision(
                    target_position=self._active_target_position,
                    confidence=confidence,
                    reason="short_reversion",
                    tags=["short", "mean_reversion", "impulse"],
                )
            if zscore <= -self.config.entry_zscore and recent_return <= -entry_momentum_threshold:
                confidence = min(1.0, abs(zscore) / (self.config.entry_zscore * 2))
                self._active_target_position = position_size
                return StrategyDecision(
                    target_position=self._active_target_position,
                    confidence=confidence,
                    reason="long_reversion",
                    tags=["long", "mean_reversion", "impulse"],
                )
            return self._flat_decision("no_entry", ["flat"])

        bars_since_entry = context.bars_since_entry or 0
        if self._active_target_position == 0.0:
            self._active_target_position = self._position_size() * context.current_position

        adverse_stop = (
            context.current_position > 0 and zscore <= -stop_zscore
        ) or (
            context.current_position < 0 and zscore >= stop_zscore
        )
        regime_break = abs(trend_deviation) > (max_trend_deviation * 1.35) or abs(regime_distance) > (max_regime_distance * 1.35)
        hit_time_stop = bars_since_entry >= self.config.max_holding_bars
        if adverse_stop or regime_break:
            self._active_target_position = 0.0
            return StrategyDecision(target_position=0.0, confidence=0.7, reason="protective_exit", tags=["exit", "risk"])
        if hit_time_stop:
            self._active_target_position = 0.0
            return StrategyDecision(target_position=0.0, confidence=0.5, reason="time_exit", tags=["exit", "time"])
        if bars_since_entry < min_holding_bars:
            return StrategyDecision(
                target_position=self._active_target_position,
                confidence=0.55,
                reason="minimum_hold",
                tags=["hold", "minimum_hold"],
            )

        should_exit = abs(zscore) <= self.config.exit_zscore
        if should_exit:
            self._active_target_position = 0.0
            return StrategyDecision(target_position=0.0, confidence=0.4, reason="mean_reversion_exit", tags=["exit"])
        return StrategyDecision(
            target_position=self._active_target_position,
            confidence=0.6,
            reason="hold_position",
            tags=["hold"],
        )
