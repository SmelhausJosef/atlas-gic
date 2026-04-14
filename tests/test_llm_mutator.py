from __future__ import annotations

from pathlib import Path

from atlas.common.config import load_config
from atlas.common.models import StrategyContext
from atlas.research.candidate_strategies import validate_candidate_module
from atlas.research.llm_mutator import LLMMutator


def test_llm_mutator_returns_empty_when_disabled(config_path):
    config = load_config(config_path)
    mutator = LLMMutator(config)
    assert mutator.generate("btc_mean_reversion_v1", "class Stub: pass", {"overall_score": 0.0}) == []


def test_llm_mutator_parses_generated_strategy_candidates(config_path):
    config = load_config(config_path)
    mutator = LLMMutator(config)
    candidates = mutator._parse_candidates(  # noqa: SLF001 - intentional unit test of structured payload parsing
        {
            "candidates": [
                {
                    "candidate_name": "Codex Candidate",
                    "family_name": "Idea Family",
                    "notes": "generated strategy module",
                    "parent_candidate_name": "baseline_parent",
                    "class_name": "GeneratedStrategy",
                    "strategy_code": """
from __future__ import annotations

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.base import BaseStrategy


class GeneratedStrategy(BaseStrategy):
    def generate(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(target_position=0.0, confidence=0.0, reason="flat", tags=["flat"])
""",
                    "mutation_spec": {
                        "objective": "reduce turnover",
                        "hypothesis": "new hold logic should filter noise",
                        "risk_notes": "trade count may fall",
                    },
                }
            ]
        },
        "btc_mean_reversion_v1",
        default_family_name=None,
        default_iteration_index=0,
    )
    assert len(candidates) == 1
    assert candidates[0].candidate_name == "codex_candidate"
    assert candidates[0].family_name == "idea_family"
    assert candidates[0].parent_candidate_name == "baseline_parent"
    assert candidates[0].strategy_code is not None
    assert candidates[0].class_name == "GeneratedStrategy"
    assert candidates[0].mutation_spec["objective"] == "reduce turnover"


def test_generated_candidate_validation_rejects_forbidden_import(config_path, market_frame, tmp_path: Path):
    config = load_config(config_path)
    module_path = tmp_path / "strategy.py"
    module_path.write_text(
        """
from __future__ import annotations

import os

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.base import BaseStrategy


class GeneratedStrategy(BaseStrategy):
    def generate(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(target_position=0.0, confidence=0.0, reason="flat", tags=["flat"])
""".strip()
        + "\n",
        encoding="utf-8",
    )
    context = StrategyContext(
        bars=market_frame.iloc[:240].copy(),
        current_position=0.0,
        gross_exposure=0.0,
        equity=config.backtest.initial_equity,
        bars_since_entry=None,
        bars_since_exit=None,
        expected_bar_seconds=config.data.expected_bar_seconds,
    )
    validation, strategy_cls = validate_candidate_module(
        candidate_name="bad_import",
        module_path=module_path,
        validation_context=context,
        strategy_config=config.strategy,
    )
    assert not validation.valid
    assert strategy_cls is None
    assert "import 'os' is not allowed" in validation.issues


def test_generated_candidate_validation_accepts_valid_strategy(config_path, market_frame, tmp_path: Path):
    config = load_config(config_path)
    module_path = tmp_path / "strategy.py"
    module_path.write_text(
        """
from __future__ import annotations

import math

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.base import BaseStrategy


class GeneratedStrategy(BaseStrategy):
    def generate(self, context: StrategyContext) -> StrategyDecision:
        price = float(context.bars["close"].iloc[-1])
        anchor = float(context.bars["close"].iloc[-10])
        edge = (price / anchor - 1.0) if anchor else 0.0
        if math.fabs(edge) < 0.0001:
            return StrategyDecision(target_position=0.0, confidence=0.0, reason="flat", tags=["flat"])
        target = -0.2 if edge > 0 else 0.2
        return StrategyDecision(target_position=target, confidence=0.4, reason="edge", tags=["generated"])
""".strip()
        + "\n",
        encoding="utf-8",
    )
    context = StrategyContext(
        bars=market_frame.iloc[:240].copy(),
        current_position=0.0,
        gross_exposure=0.0,
        equity=config.backtest.initial_equity,
        bars_since_entry=None,
        bars_since_exit=None,
        expected_bar_seconds=config.data.expected_bar_seconds,
    )
    validation, strategy_cls = validate_candidate_module(
        candidate_name="generated_ok",
        module_path=module_path,
        validation_context=context,
        strategy_config=config.strategy,
    )
    assert validation.valid
    assert strategy_cls is not None
    assert validation.class_name == "GeneratedStrategy"
