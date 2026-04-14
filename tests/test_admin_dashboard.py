from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from atlas.cli import main
from atlas_admin.app import create_app
from atlas_admin.models import JobSnapshot
from atlas_admin.services import DashboardService


class FakeRegistry:
    def __init__(self) -> None:
        self.backtest_calls: list[tuple[Path, str | None]] = []
        self.research_calls: list[tuple[Path, str | None, int | None, int | None, bool | None]] = []
        self.stop_calls: list[str] = []
        self.jobs: dict[str, JobSnapshot] = {}

    def launch_backtest(self, config_path: Path, strategy: str | None = None) -> JobSnapshot:
        self.backtest_calls.append((config_path, strategy))
        snapshot = self._build_snapshot("backtest", strategy)
        self.jobs[snapshot.dashboard_job_id] = snapshot
        return snapshot

    def launch_research(
        self,
        config_path: Path,
        baseline: str | None = None,
        *,
        candidate_budget: int | None = None,
        max_generations: int | None = None,
        use_llm: bool | None = None,
    ) -> JobSnapshot:
        self.research_calls.append((config_path, baseline, candidate_budget, max_generations, use_llm))
        snapshot = self._build_snapshot("research", baseline)
        self.jobs[snapshot.dashboard_job_id] = snapshot
        return snapshot

    def stop(self, target_id: str) -> bool:
        self.stop_calls.append(target_id)
        return True

    def list_jobs(self) -> list[JobSnapshot]:
        return list(self.jobs.values())

    def get_job(self, target_id: str) -> JobSnapshot | None:
        return self.jobs.get(target_id)

    def _build_snapshot(self, command_type: str, label: str | None) -> JobSnapshot:
        now = datetime.now(timezone.utc)
        return JobSnapshot(
            dashboard_job_id=f"{command_type}-job",
            command_type=command_type,
            status="running",
            config_path=Path("/tmp/config.yaml"),
            started_at=now,
            last_seen_at=now,
            atlas_run_id=None,
            strategy_or_baseline=label,
            stdout_log_path=Path(f"/tmp/{command_type}.log"),
        )


def _seed_runs(config_path: Path) -> None:
    assert main(["backtest", "run", "--config", str(config_path), "--strategy", "btc_mean_reversion_v1"]) == 0
    assert main(["research", "run", "--config", str(config_path), "--baseline", "btc_mean_reversion_v1"]) == 0


def test_dashboard_pages_render(config_path: Path) -> None:
    _seed_runs(config_path)
    service = DashboardService(project_root=config_path.parent, config_path=config_path)
    registry = FakeRegistry()
    client = TestClient(create_app(service=service, job_registry=registry))

    overview = client.get("/")
    assert overview.status_code == 200
    assert "ATLAS Admin" in overview.text
    assert "BTCUSDT-PERP" in overview.text

    runs = client.get("/runs")
    assert runs.status_code == 200
    assert "Research btc_mean_reversion_v1" in runs.text
    assert "Backtest btc_mean_reversion_v1" in runs.text

    run_items = service.list_runs(registry)
    research_run = next(item for item in run_items if item.run_type == "research")
    backtest_run = next(item for item in run_items if item.run_type == "backtest")

    research_detail = client.get(f"/runs/{research_run.run_id}")
    assert research_detail.status_code == 200
    assert "Leaderboard" in research_detail.text
    assert "Lineage" in research_detail.text
    assert "Strategy families" in research_detail.text
    assert "Progress log" in research_detail.text

    backtest_detail = client.get(f"/runs/{backtest_run.run_id}")
    assert backtest_detail.status_code == 200
    assert "Metrics" in backtest_detail.text
    assert "Equity curve" in backtest_detail.text

    heartbeat = client.get(f"/api/heartbeat/{research_run.run_id}")
    assert heartbeat.status_code == 200
    assert heartbeat.json()["status"] == "completed"

    log_preview = client.get(f"/partials/log-preview/{research_run.run_id}")
    assert log_preview.status_code == 200


def test_dashboard_actions_delegate_to_registry(config_path: Path) -> None:
    service = DashboardService(project_root=config_path.parent, config_path=config_path)
    registry = FakeRegistry()
    client = TestClient(create_app(service=service, job_registry=registry))

    backtest_response = client.post(
        "/actions/backtest",
        data={"config_path_value": str(config_path), "strategy": "btc_mean_reversion_v1"},
        follow_redirects=False,
    )
    assert backtest_response.status_code == 303
    assert registry.backtest_calls == [(config_path, "btc_mean_reversion_v1")]

    research_response = client.post(
        "/actions/research",
        data={
            "config_path_value": str(config_path),
            "baseline": "btc_mean_reversion_v1",
            "candidate_budget": "3",
            "max_generations": "2",
        },
        follow_redirects=False,
    )
    assert research_response.status_code == 303
    assert registry.research_calls == [(config_path, "btc_mean_reversion_v1", 3, 2, True)]

    stop_response = client.post("/actions/stop/backtest-job", follow_redirects=False)
    assert stop_response.status_code == 303
    assert registry.stop_calls == ["backtest-job"]


def test_dashboard_service_handles_sparse_research_artifacts(tmp_path: Path, config_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    run_dir = artifacts_dir / "20260414T000000Z-research_sparse-abcdef12"
    run_dir.mkdir(parents=True)
    (run_dir / "session.json").write_text(
        json.dumps({"experiment_id": run_dir.name, "baseline_name": "btc_mean_reversion_v1", "created_at": "2026-04-14T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (run_dir / "heartbeat.json").write_text(
        json.dumps({"status": "running", "stage": "baseline", "updated_at": "2026-04-14T00:00:01+00:00"}),
        encoding="utf-8",
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["app"]["artifacts_dir"] = str(artifacts_dir)
    adjusted_config = tmp_path / "dashboard.yaml"
    adjusted_config.write_text(json.dumps(payload), encoding="utf-8")

    service = DashboardService(project_root=tmp_path, config_path=adjusted_config)
    detail = service.get_run_detail(run_dir.name, FakeRegistry())
    assert detail is not None
    assert detail.run_type == "research"
    assert detail.leaderboard == []
    assert detail.lineage == []


def test_dashboard_service_marks_stale_research_as_failed(tmp_path: Path, config_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    run_dir = artifacts_dir / "20260414T000000Z-research_stale-abcdef12"
    run_dir.mkdir(parents=True)
    (run_dir / "session.json").write_text(
        json.dumps({"experiment_id": run_dir.name, "baseline_name": "btc_mean_reversion_v1", "created_at": "2026-04-14T00:00:00+00:00"}),
        encoding="utf-8",
    )
    (run_dir / "heartbeat.json").write_text(
        json.dumps({"status": "running", "stage": "baseline", "updated_at": "2026-04-14T00:00:01+00:00"}),
        encoding="utf-8",
    )
    (run_dir / "research_summary.json").write_text(
        json.dumps({"error": "Codex SDK is not installed", "stop_reason": "failed:RuntimeError"}),
        encoding="utf-8",
    )
    (run_dir / "dashboard_job.json").write_text(
        json.dumps({"pid": 999999, "stdout_log_path": str(run_dir / "stale.log")}),
        encoding="utf-8",
    )
    (run_dir / "stale.log").write_text("Traceback\nRuntimeError: codex sdk missing\n", encoding="utf-8")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["app"]["artifacts_dir"] = str(artifacts_dir)
    adjusted_config = tmp_path / "dashboard.yaml"
    adjusted_config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    service = DashboardService(project_root=tmp_path, config_path=adjusted_config)
    detail = service.get_run_detail(run_dir.name, FakeRegistry())
    assert detail is not None
    assert detail.status == "failed"
    assert detail.stop_reason == "failed:RuntimeError"
    assert detail.error_message == "Codex SDK is not installed"


def test_dashboard_service_returns_log_preview(tmp_path: Path, config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adjusted_config = tmp_path / "dashboard.yaml"
    adjusted_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    service = DashboardService(project_root=tmp_path, config_path=adjusted_config)
    log_path = tmp_path / "research.log"
    log_path.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    registry = FakeRegistry()
    snapshot = registry._build_snapshot("research", "btc_mean_reversion_v1")
    snapshot.stdout_log_path = log_path
    registry.jobs[snapshot.dashboard_job_id] = snapshot

    preview = service.get_log_preview(snapshot.dashboard_job_id, registry, line_limit=2)
    assert preview == "line 2\nline 3"
