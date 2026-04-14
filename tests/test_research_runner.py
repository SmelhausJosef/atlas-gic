from __future__ import annotations

import json
from pathlib import Path

from atlas.common.config import load_config
from atlas.research.experiment_runner import ExperimentRunner
from atlas.research.llm_mutator import MutationCandidate


def test_research_runner_returns_candidate_summary(config_path):
    config = load_config(config_path)
    runner = ExperimentRunner(config)
    summary = runner.run("btc_mean_reversion_v1")
    assert summary["baseline"]["strategy"] == "btc_mean_reversion_v1"
    assert "families" in summary
    assert len(summary["families"]) == 1
    assert "verdict" in summary["families"][0]
    assert "iterations" in summary["families"][0]
    run_dir = Path(summary["artifact_dir"])
    assert (run_dir / "heartbeat.json").exists()
    assert (run_dir / "leaderboard.json").exists()
    assert (run_dir / "lineage.json").exists()
    assert (run_dir / "research_summary.json").exists()
    assert (run_dir / "families" / "btc_mean_reversion_tight_v1" / "verdict.json").exists()
    heartbeat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "completed"
    leaderboard = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert leaderboard[0]["rank_candidate_name"] == "btc_mean_reversion_v1"


def test_research_runner_persists_generated_candidate_bundle(config_path, monkeypatch):
    config = load_config(config_path)
    config.research.candidate_strategies = []
    runner = ExperimentRunner(config)

    monkeypatch.setattr(
        runner.mutator,
        "propose_strategy_families",
        lambda parent_name, parent_source, summary, remaining_budget=None: [
            MutationCandidate(
                candidate_name="generated_strategy_alpha",
                family_name="generated_strategy_alpha",
                strategy_name="generated_strategy_alpha",
                source="codex_sdk",
                notes="generated test strategy",
                parent_candidate_name=parent_name,
                class_name="GeneratedStrategy",
                strategy_code="""
from __future__ import annotations

import math

from atlas.common.models import StrategyContext, StrategyDecision
from atlas.strategies.base import BaseStrategy


class GeneratedStrategy(BaseStrategy):
    def generate(self, context: StrategyContext) -> StrategyDecision:
        close = context.bars["close"].astype(float)
        edge = float(close.iloc[-1] / close.iloc[-20] - 1.0)
        if math.fabs(edge) < 0.001:
            return StrategyDecision(target_position=0.0, confidence=0.0, reason="flat", tags=["flat"])
        target = -0.15 if edge > 0 else 0.15
        return StrategyDecision(target_position=target, confidence=0.35, reason="generated_edge", tags=["generated"])
""".strip()
                + "\n",
                mutation_spec={
                    "objective": "discover a different hold logic",
                    "hypothesis": "small mean reversion edge may survive with lighter sizing",
                    "risk_notes": "could under-trade",
                },
            )
        ],
    )

    summary = runner.run("btc_mean_reversion_v1")
    run_dir = Path(summary["artifact_dir"])
    candidate_dir = run_dir / "families" / "generated_strategy_alpha" / "iterations" / "00_generated_strategy_alpha"
    assert candidate_dir.exists()
    assert (candidate_dir / "strategy.py").exists()
    validation = json.loads((candidate_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "passed"
    leaderboard = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
    generated_row = next(row for row in leaderboard if row["rank_candidate_name"] == "generated_strategy_alpha")
    assert generated_row["validation_status"] == "passed"
    assert generated_row["family_name"] == "generated_strategy_alpha"
    assert generated_row["strategy_source_path"].endswith("00_generated_strategy_alpha/strategy.py")
    summary_payload = json.loads((run_dir / "research_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["families"][0]["family_name"] == "generated_strategy_alpha"
    assert summary_payload["families"][0]["iterations"][0]["candidate_name"] == "generated_strategy_alpha"


def test_research_runner_marks_session_failed_on_exception(config_path, monkeypatch):
    config = load_config(config_path)
    config.research.use_llm = True
    config.research.candidate_strategies = []
    runner = ExperimentRunner(config)

    def _raise(*args, **kwargs):
        raise RuntimeError("codex sdk missing")

    monkeypatch.setattr(runner.mutator, "propose_strategy_families", _raise)

    try:
        runner.run("btc_mean_reversion_v1")
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("runner.run should have raised RuntimeError")

    failed_runs = sorted(config.app.artifacts_dir.glob("*-research_btc_mean_reversion_v1-*/heartbeat.json"))
    assert failed_runs
    heartbeat = json.loads(failed_runs[-1].read_text(encoding="utf-8"))
    assert heartbeat["status"] == "failed"
    assert heartbeat["stage"] == "failed"
