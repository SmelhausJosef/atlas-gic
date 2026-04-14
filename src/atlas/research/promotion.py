from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PromotionDecision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reasons": self.reasons}


def evaluate_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    minimum_trade_count: int,
    max_drawdown_pct: float,
    improvement_epsilon: float,
) -> PromotionDecision:
    reasons: list[str] = []
    if candidate["full"]["metrics"]["trade_count"] < minimum_trade_count:
        reasons.append("candidate failed minimum trade count")
    if candidate["oos"]["metrics"]["trade_count"] < max(1, minimum_trade_count // 2):
        reasons.append("candidate failed OOS trade count")
    if candidate["full"]["metrics"]["max_drawdown"] > max_drawdown_pct:
        reasons.append("candidate exceeded max drawdown threshold")
    if candidate["overall_score"] <= baseline["overall_score"] + improvement_epsilon:
        reasons.append("candidate did not improve overall composite score")
    if candidate["oos_score"] <= baseline["oos_score"]:
        reasons.append("candidate did not improve out-of-sample score")
    if candidate["stressed_score"] < baseline["stressed_score"]:
        reasons.append("candidate regressed under stressed costs")
    return PromotionDecision(accepted=not reasons, reasons=reasons)

