from __future__ import annotations

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.btc_mean_reversion_v1 import BTCMeanReversionV1


class BTCMeanReversionTightV1(BTCMeanReversionV1):
    def generate(self, context: StrategyContext) -> StrategyDecision:
        decision = super().generate(context)
        if decision.reason in {"hold_position", "minimum_hold"} and context.bars_since_entry is not None and context.bars_since_entry >= 72:
            self._active_target_position = 0.0
            return StrategyDecision(target_position=0.0, confidence=0.5, reason="tight_time_exit", tags=["exit", "candidate"])
        return decision
