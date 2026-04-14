from __future__ import annotations

import pytest

from atlas.common.models import StrategyDecision


def test_strategy_decision_rejects_invalid_position():
    with pytest.raises(ValueError):
        StrategyDecision(target_position=1.1, confidence=0.5, reason="bad")


def test_strategy_decision_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        StrategyDecision(target_position=0.0, confidence=1.5, reason="bad")


def test_strategy_decision_requires_reason():
    with pytest.raises(ValueError):
        StrategyDecision(target_position=0.0, confidence=0.1, reason="")

