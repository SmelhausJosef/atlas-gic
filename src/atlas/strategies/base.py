from __future__ import annotations

from abc import ABC, abstractmethod

from atlas.common.config import StrategyConfig
from atlas.common.models import StrategyContext, StrategyDecision


class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    def generate(self, context: StrategyContext) -> StrategyDecision:
        """Return a bounded target position for the next bar."""

