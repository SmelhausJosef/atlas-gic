from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.common.config import AppConfig, load_config
from atlas.data.loaders import load_market_data
from atlas_admin.job_registry import JobRegistry
from atlas_admin.models import DatasetOverview, JobSnapshot, RunDetail, RunListItem

STALE_HEARTBEAT_SECONDS = 120


class DashboardService:
    def __init__(self, project_root: Path, config_path: Path):
        self.project_root = project_root
        self.config_path = config_path

    @property
    @lru_cache(maxsize=1)
    def config(self) -> AppConfig:
        return load_config(self.config_path)

    @property
    def artifacts_dir(self) -> Path:
        path = self.config.app.artifacts_dir
        return path if path.is_absolute() else self.project_root / path

    def get_dataset_overview(self) -> DatasetOverview:
        config = self.config
        data_path = config.data.path if config.data.path.is_absolute() else self.project_root / config.data.path
        cache_key = (str(data_path.resolve()), data_path.stat().st_mtime_ns)
        return self._dataset_overview_cached(cache_key)

    @lru_cache(maxsize=4)
    def _dataset_overview_cached(self, _cache_key: tuple[str, int]) -> DatasetOverview:
        data_path = self.config.data.path
        if not data_path.is_absolute():
            data_path = self.project_root / data_path
        data_config = self.config.data.model_copy(update={"path": data_path})
        _, manifest, report = load_market_data(data_config)
        return DatasetOverview(
            path=manifest.path,
            instrument=manifest.instrument,
            venue=manifest.venue,
            timeframe=manifest.timeframe,
            row_count=manifest.row_count,
            date_start=manifest.date_start.isoformat(),
            date_end=manifest.date_end.isoformat(),
            sha256=manifest.sha256,
            issues=report.issues,
        )

    def get_available_strategies(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path.relative_to(self.project_root)),
            "baseline": self.config.strategy.name,
            "candidates": self.config.research.candidate_strategies,
            "family_budget": self.config.research.family_budget,
            "tuning_iterations_per_family": self.config.research.tuning_iterations_per_family,
        }

    def list_runs(self, registry: JobRegistry, kind: str | None = None) -> list[RunListItem]:
        items: list[RunListItem] = []
        job_by_run_id = {
            snapshot.atlas_run_id: snapshot
            for snapshot in registry.list_jobs()
            if snapshot.atlas_run_id is not None
        }
        pending_jobs = [snapshot for snapshot in registry.list_jobs() if snapshot.atlas_run_id is None]
        if self.artifacts_dir.exists():
            for run_dir in sorted(
                [path for path in self.artifacts_dir.iterdir() if path.is_dir()],
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                item = self._build_run_list_item(run_dir, job_by_run_id.get(run_dir.name))
                if kind is None or item.run_type == kind:
                    items.append(item)
        if kind is None:
            items.extend(self._build_pending_item(job) for job in pending_jobs)
        return sorted(items, key=lambda item: item.updated_at or "", reverse=True)

    def get_run_detail(self, run_id: str, registry: JobRegistry) -> RunDetail | None:
        job = registry.get_job(run_id)
        if job is not None and job.atlas_run_id is None:
            return RunDetail(
                run_id=job.dashboard_job_id,
                title=f"Pending {job.command_type}",
                run_type=job.command_type,
                status=job.status,
                started_at=job.started_at.isoformat(),
                updated_at=job.last_seen_at.isoformat(),
                strategy=job.strategy_or_baseline if job.command_type == "backtest" else None,
                baseline_name=job.strategy_or_baseline if job.command_type == "research" else None,
                stop_reason=None,
                error_message=None,
                overall_score=None,
                log_preview=self._tail_log(job.stdout_log_path),
                dashboard_job_id=job.dashboard_job_id,
                stdout_log_path=str(job.stdout_log_path) if job.stdout_log_path else None,
            )

        run_dir = self.artifacts_dir / run_id
        if not run_dir.exists():
            return None
        return self._build_run_detail(run_dir, registry.get_job(run_id))

    def get_heartbeat_payload(self, run_id: str, registry: JobRegistry) -> dict[str, Any] | None:
        detail = self.get_run_detail(run_id, registry)
        if detail is None:
            return None
        payload = detail.heartbeat or {}
        if detail.dashboard_job_id is not None:
            payload = {
                **payload,
                "dashboard_job_id": detail.dashboard_job_id,
                "process_status": detail.status,
                "stdout_log_path": detail.stdout_log_path,
            }
        return payload or {
            "status": detail.status,
            "run_id": detail.run_id,
            "dashboard_job_id": detail.dashboard_job_id,
            "stdout_log_path": detail.stdout_log_path,
        }

    def get_log_preview(self, run_id: str, registry: JobRegistry, *, line_limit: int = 60) -> str | None:
        detail = self.get_run_detail(run_id, registry)
        if detail is None or not detail.stdout_log_path:
            return None
        return self._tail_log(Path(detail.stdout_log_path), line_limit=line_limit)

    def _build_run_list_item(self, run_dir: Path, job: JobSnapshot | None) -> RunListItem:
        is_research = (run_dir / "session.json").exists()
        if is_research:
            session = self._read_json(run_dir / "session.json")
            heartbeat = self._read_json(run_dir / "heartbeat.json")
            summary = self._read_json(run_dir / "research_summary.json")
            dashboard_job = self._read_json(run_dir / "dashboard_job.json", default={})
            status, stop_reason = self._resolve_research_status(
                heartbeat=heartbeat,
                summary=summary,
                dashboard_job=dashboard_job,
                runtime_job=job,
            )
            return RunListItem(
                run_id=run_dir.name,
                title=f"Research {session.get('baseline_name', 'unknown')}",
                run_type="research",
                status=status,
                started_at=session.get("created_at"),
                updated_at=heartbeat.get("updated_at"),
                strategy=None,
                baseline_name=session.get("baseline_name"),
                stop_reason=stop_reason,
                overall_score=summary.get("best_overall_score"),
                dashboard_job_id=job.dashboard_job_id if job is not None else None,
                stdout_log_path=self._resolve_log_path(job, dashboard_job),
            )
        metrics = self._read_json(run_dir / "metrics.json")
        config = self._read_json(run_dir / "config.json")
        summary_text = (run_dir / "summary.md").read_text(encoding="utf-8") if (run_dir / "summary.md").exists() else ""
        strategy = self._extract_strategy_name(config, summary_text, run_dir.name)
        status = job.status if job is not None else "completed"
        return RunListItem(
            run_id=run_dir.name,
            title=f"Backtest {strategy}",
            run_type="backtest",
            status=status,
            started_at=self._run_created_at(run_dir),
            updated_at=self._run_updated_at(run_dir),
            strategy=strategy,
            baseline_name=None,
            stop_reason=None,
            overall_score=metrics.get("composite_score"),
            dashboard_job_id=job.dashboard_job_id if job is not None else None,
            stdout_log_path=str(job.stdout_log_path) if job and job.stdout_log_path else None,
        )

    def _build_run_detail(self, run_dir: Path, job: JobSnapshot | None) -> RunDetail:
        artifact_files = sorted(path.name for path in run_dir.iterdir() if path.is_file())
        is_research = (run_dir / "session.json").exists()
        if is_research:
            session = self._read_json(run_dir / "session.json")
            heartbeat = self._read_json(run_dir / "heartbeat.json")
            summary = self._read_json(run_dir / "research_summary.json")
            dashboard_job = self._read_json(run_dir / "dashboard_job.json", default={})
            status, stop_reason = self._resolve_research_status(
                heartbeat=heartbeat,
                summary=summary,
                dashboard_job=dashboard_job,
                runtime_job=job,
            )
            stdout_log_path = self._resolve_log_path(job, dashboard_job)
            return RunDetail(
                run_id=run_dir.name,
                title=f"Research {session.get('baseline_name', 'unknown')}",
                run_type="research",
                status=status,
                started_at=session.get("created_at"),
                updated_at=heartbeat.get("updated_at"),
                strategy=None,
                baseline_name=session.get("baseline_name"),
                stop_reason=stop_reason,
                error_message=summary.get("error"),
                overall_score=summary.get("best_overall_score"),
                dataset_manifest=self._read_json(run_dir / "dataset_manifest.json"),
                heartbeat=heartbeat,
                leaderboard=self._read_json(run_dir / "leaderboard.json", default=[]),
                lineage=self._read_json(run_dir / "lineage.json", default=[]),
                family_summaries=summary.get("families", []),
                strategy_memory_records=summary.get("strategy_memory_records", []),
                summary_text=json.dumps(summary, indent=2),
                artifact_files=artifact_files,
                log_preview=self._tail_log(Path(stdout_log_path) if stdout_log_path else None),
                dashboard_job_id=job.dashboard_job_id if job is not None else None,
                stdout_log_path=stdout_log_path,
            )
        metrics = self._read_json(run_dir / "metrics.json")
        summary_text = (run_dir / "summary.md").read_text(encoding="utf-8") if (run_dir / "summary.md").exists() else None
        config = self._read_json(run_dir / "config.json")
        strategy = self._extract_strategy_name(config, summary_text or "", run_dir.name)
        return RunDetail(
            run_id=run_dir.name,
            title=f"Backtest {strategy}",
            run_type="backtest",
            status=job.status if job is not None else "completed",
            started_at=self._run_created_at(run_dir),
            updated_at=self._run_updated_at(run_dir),
            strategy=strategy,
            baseline_name=None,
            stop_reason=None,
            error_message=None,
            overall_score=metrics.get("composite_score"),
            metrics=metrics,
            dataset_manifest=self._read_json(run_dir / "dataset_manifest.json"),
            summary_text=summary_text,
            artifact_files=artifact_files,
            equity_points=self._load_equity_points(run_dir / "equity.csv"),
            log_preview=self._tail_log(job.stdout_log_path if job is not None else None),
            dashboard_job_id=job.dashboard_job_id if job is not None else None,
            stdout_log_path=str(job.stdout_log_path) if job and job.stdout_log_path else None,
        )

    def _build_pending_item(self, job: JobSnapshot) -> RunListItem:
        title = "Pending backtest" if job.command_type == "backtest" else "Pending research"
        return RunListItem(
            run_id=job.dashboard_job_id,
            title=title,
            run_type=job.command_type,
            status=job.status,
            started_at=job.started_at.isoformat(),
            updated_at=job.last_seen_at.isoformat(),
            strategy=job.strategy_or_baseline if job.command_type == "backtest" else None,
            baseline_name=job.strategy_or_baseline if job.command_type == "research" else None,
            stop_reason=None,
            overall_score=None,
            dashboard_job_id=job.dashboard_job_id,
            stdout_log_path=str(job.stdout_log_path) if job.stdout_log_path else None,
        )

    def _extract_strategy_name(self, config: dict[str, Any], summary_text: str, fallback: str) -> str:
        strategy = config.get("strategy", {}).get("name")
        if strategy:
            return str(strategy)
        for line in summary_text.splitlines():
            if line.startswith("- Strategy:"):
                return line.split("`")[1]
        return fallback

    def _read_json(self, path: Path, default: Any | None = None) -> Any:
        if not path.exists():
            return {} if default is None else default
        return json.loads(path.read_text(encoding="utf-8"))

    def _run_created_at(self, run_dir: Path) -> str:
        return datetime.fromtimestamp(run_dir.stat().st_ctime, tz=timezone.utc).isoformat()

    def _run_updated_at(self, run_dir: Path) -> str:
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()

    def _load_equity_points(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        if len(frame) > 300:
            frame = frame.tail(300)
        return frame.to_dict(orient="records")

    def _tail_log(self, path: Path | None, *, line_limit: int = 60) -> str | None:
        if path is None or not path.exists():
            return None
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return None
        return "\n".join(lines[-line_limit:])

    def _resolve_log_path(self, job: JobSnapshot | None, dashboard_job: dict[str, Any]) -> str | None:
        if job is not None and job.stdout_log_path is not None:
            return str(job.stdout_log_path)
        log_path = dashboard_job.get("stdout_log_path")
        return str(log_path) if isinstance(log_path, str) and log_path else None

    def _resolve_research_status(
        self,
        *,
        heartbeat: dict[str, Any],
        summary: dict[str, Any],
        dashboard_job: dict[str, Any],
        runtime_job: JobSnapshot | None,
    ) -> tuple[str, str | None]:
        if runtime_job is not None:
            return runtime_job.status, heartbeat.get("stop_reason") or summary.get("stop_reason")

        heartbeat_status = str(heartbeat.get("status", "unknown"))
        stop_reason = heartbeat.get("stop_reason") or summary.get("stop_reason")
        if heartbeat_status != "running":
            return heartbeat_status, stop_reason

        pid = dashboard_job.get("pid")
        if isinstance(pid, int) and self._pid_is_alive(pid):
            return "running", stop_reason

        updated_at = self._parse_dt(heartbeat.get("updated_at"))
        if updated_at is not None:
            age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
            if age_seconds <= STALE_HEARTBEAT_SECONDS:
                return "running", stop_reason

        log_path_raw = dashboard_job.get("stdout_log_path")
        log_path = Path(log_path_raw) if isinstance(log_path_raw, str) and log_path_raw else None
        log_preview = self._tail_log(log_path, line_limit=80)
        if log_preview and ("Traceback" in log_preview or "RuntimeError" in log_preview):
            return "failed", stop_reason or "stale_process_failed"
        return "stopped", stop_reason or "stale_process_orphaned"

    def _parse_dt(self, value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _pid_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
