from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.common.config import AppConfig
from atlas.common.models import BacktestResult


class ArtifactManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.base_dir = config.app.artifacts_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = config.research.strategy_memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def write_backtest_run(self, result: BacktestResult, config: AppConfig) -> Path:
        run_dir = self.base_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "config.json", config.model_dump(mode="json"))
        self._write_json(run_dir / "dataset_manifest.json", result.manifest.to_dict())
        self._write_json(run_dir / "metrics.json", result.metrics)
        self._write_csv(run_dir / "trades.csv", [trade.to_dict() for trade in result.trades])
        result.equity_curve.to_csv(run_dir / "equity.csv", index=False)
        (run_dir / "summary.md").write_text(self._build_summary(result), encoding="utf-8")
        return run_dir

    def create_research_session(
        self,
        experiment_id: str,
        *,
        baseline_name: str,
        config: AppConfig,
        manifest: dict[str, Any],
    ) -> Path:
        run_dir = self.base_dir / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "config.json", config.model_dump(mode="json"))
        self._write_json(run_dir / "dataset_manifest.json", manifest)
        self._write_json(
            run_dir / "session.json",
            {
                "experiment_id": experiment_id,
                "baseline_name": baseline_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return run_dir

    def write_research_heartbeat(self, run_dir: Path, payload: dict[str, Any]) -> Path:
        heartbeat_path = run_dir / "heartbeat.json"
        body = {"updated_at": datetime.now(timezone.utc).isoformat(), **payload}
        self._write_json(heartbeat_path, body)
        return heartbeat_path

    def write_research_leaderboard(self, run_dir: Path, rows: list[dict[str, Any]]) -> Path:
        leaderboard_path = run_dir / "leaderboard.json"
        ordered = sorted(
            rows,
            key=lambda row: row["overall_score"] if isinstance(row.get("overall_score"), (int, float)) else float("-inf"),
            reverse=True,
        )
        self._write_json(leaderboard_path, ordered)
        self._write_csv(run_dir / "leaderboard.csv", ordered)
        return leaderboard_path

    def write_research_lineage(self, run_dir: Path, rows: list[dict[str, Any]]) -> Path:
        lineage_path = run_dir / "lineage.json"
        self._write_json(lineage_path, rows)
        self._write_csv(run_dir / "lineage.csv", rows)
        return lineage_path

    def write_research_summary(self, run_dir: Path, payload: dict[str, Any]) -> Path:
        summary_path = run_dir / "research_summary.json"
        self._write_json(summary_path, payload)
        return summary_path

    def create_candidate_bundle(
        self,
        run_dir: Path,
        *,
        candidate_name: str,
        metadata: dict[str, Any],
        strategy_code: str | None = None,
    ) -> Path:
        candidate_dir = run_dir / "candidates" / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(candidate_dir / "candidate.json", metadata)
        if strategy_code is not None:
            (candidate_dir / "strategy.py").write_text(strategy_code, encoding="utf-8")
        return candidate_dir

    def write_candidate_validation(self, candidate_dir: Path, payload: dict[str, Any]) -> Path:
        validation_path = candidate_dir / "validation.json"
        self._write_json(validation_path, payload)
        return validation_path

    def create_family_bundle(self, run_dir: Path, *, family_name: str, metadata: dict[str, Any]) -> Path:
        family_dir = run_dir / "families" / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(family_dir / "family.json", metadata)
        return family_dir

    def create_family_iteration_bundle(
        self,
        family_dir: Path,
        *,
        family_name: str,
        candidate_name: str,
        iteration_index: int,
        metadata: dict[str, Any],
        strategy_code: str | None = None,
    ) -> Path:
        iteration_dir = family_dir / "iterations" / f"{iteration_index:02d}_{candidate_name}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            iteration_dir / "iteration.json",
            {"family_name": family_name, "candidate_name": candidate_name, "iteration_index": iteration_index, **metadata},
        )
        if strategy_code is not None:
            (iteration_dir / "strategy.py").write_text(strategy_code, encoding="utf-8")
        return iteration_dir

    def write_family_verdict(self, family_dir: Path, payload: dict[str, Any]) -> Path:
        verdict_path = family_dir / "verdict.json"
        self._write_json(verdict_path, payload)
        return verdict_path

    def append_strategy_memory(self, category: str, payload: dict[str, Any]) -> Path:
        category_path = self.memory_dir / f"{category}_families.jsonl"
        with category_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
        return category_path

    def _build_summary(self, result: BacktestResult) -> str:
        lines = [
            f"# Run {result.run_id}",
            "",
            f"- Strategy: `{result.strategy_name}`",
            f"- Instrument: `{result.manifest.instrument}`",
            f"- Timeframe: `{result.manifest.timeframe}`",
            f"- Date range: `{result.manifest.date_start.isoformat()}` to `{result.manifest.date_end.isoformat()}`",
            f"- Execution: {result.execution_assumptions}",
            "",
            "## Metrics",
        ]
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(result.metrics.items()))
        return "\n".join(lines) + "\n"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)


def build_run_id(strategy_name: str, dataset_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = _git_sha()
    material = f"{timestamp}:{strategy_name}:{dataset_hash}:{git_sha}"
    digest = sha1(material.encode("utf-8")).hexdigest()[:8]
    return f"{timestamp}-{strategy_name}-{digest}"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"
