from __future__ import annotations

import json
import os
import pty
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from atlas.common.config import load_config
from atlas_admin.models import JobSnapshot


@dataclass(slots=True)
class _JobHandle:
    dashboard_job_id: str
    process: subprocess.Popen[str]
    master_fd: int
    command_type: str
    config_path: Path
    started_at: datetime
    last_seen_at: datetime
    atlas_run_id: str | None
    strategy_or_baseline: str | None
    stdout_log_path: Path
    runtime_config_path: Path | None
    stop_requested: bool = False


class JobRegistry:
    def __init__(self, project_root: Path, artifacts_dir: Path):
        self.project_root = project_root
        self.artifacts_dir = artifacts_dir
        self.runtime_dir = project_root / ".atlas_admin_runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.runtime_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.runtime_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.runtime_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobHandle] = {}

    def launch_backtest(self, config_path: Path, strategy: str | None = None) -> JobSnapshot:
        command = [sys.executable, "-m", "atlas", "backtest", "run", "--config", str(config_path)]
        if strategy:
            command.extend(["--strategy", strategy])
        return self._launch(command, "backtest", config_path, strategy, None)

    def launch_research(
        self,
        config_path: Path,
        baseline: str | None = None,
        *,
        candidate_budget: int | None = None,
        max_generations: int | None = None,
        use_llm: bool | None = None,
    ) -> JobSnapshot:
        effective_config_path = config_path
        runtime_config_path: Path | None = None
        if any(value is not None for value in (candidate_budget, max_generations, use_llm)):
            runtime_config_path = self._write_runtime_config(
                config_path,
                {
                    "research": {
                        **({"candidate_budget": candidate_budget} if candidate_budget is not None else {}),
                        **({"max_generations": max_generations} if max_generations is not None else {}),
                        **({"use_llm": use_llm} if use_llm is not None else {}),
                    }
                },
            )
            effective_config_path = runtime_config_path

        command = [sys.executable, "-m", "atlas", "research", "run", "--config", str(effective_config_path)]
        if baseline:
            command.extend(["--baseline", baseline])
        return self._launch(command, "research", effective_config_path, baseline, runtime_config_path)

    def stop(self, target_id: str) -> bool:
        handle: _JobHandle | None = None
        with self._lock:
            handle = self._jobs.get(target_id)
            if handle is None:
                handle = next(
                    (job for job in self._jobs.values() if job.atlas_run_id == target_id),
                    None,
                )
            if handle is None:
                return False
            handle.stop_requested = True
            if handle.process.poll() is not None:
                return True
        try:
            try:
                os.killpg(handle.process.pid, signal.SIGINT)
            except ProcessLookupError:
                return True
            handle.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(handle.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return True
        with self._lock:
            self._refresh_handle(handle)
        return True

    def list_jobs(self) -> list[JobSnapshot]:
        with self._lock:
            snapshots = [self._snapshot(handle) for handle in self._jobs.values()]
        return sorted(snapshots, key=lambda item: item.started_at, reverse=True)

    def get_job(self, target_id: str) -> JobSnapshot | None:
        for snapshot in self.list_jobs():
            if snapshot.dashboard_job_id == target_id or snapshot.atlas_run_id == target_id:
                return snapshot
        return None

    def _launch(
        self,
        command: list[str],
        command_type: str,
        config_path: Path,
        strategy_or_baseline: str | None,
        runtime_config_path: Path | None,
    ) -> JobSnapshot:
        dashboard_job_id = uuid.uuid4().hex[:12]
        stdout_log_path = self.logs_dir / f"{dashboard_job_id}.log"
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=True,
            start_new_session=True,
        )
        os.close(slave_fd)
        now = datetime.now(timezone.utc)
        handle = _JobHandle(
            dashboard_job_id=dashboard_job_id,
            process=process,
            master_fd=master_fd,
            command_type=command_type,
            config_path=config_path,
            started_at=now,
            last_seen_at=now,
            atlas_run_id=None,
            strategy_or_baseline=strategy_or_baseline,
            stdout_log_path=stdout_log_path,
            runtime_config_path=runtime_config_path,
        )
        with self._lock:
            self._jobs[dashboard_job_id] = handle
            self._write_job_metadata(handle)
        threading.Thread(target=self._pump_pty_output, args=(handle,), daemon=True).start()
        return self._snapshot(handle)

    def _snapshot(self, handle: _JobHandle) -> JobSnapshot:
        self._refresh_handle(handle)
        return_code = handle.process.poll()
        status = "running"
        if return_code is not None:
            status = "stopped" if handle.stop_requested else ("completed" if return_code == 0 else "failed")
        return JobSnapshot(
            dashboard_job_id=handle.dashboard_job_id,
            command_type=handle.command_type,
            status=status,
            config_path=handle.config_path,
            started_at=handle.started_at,
            last_seen_at=handle.last_seen_at,
            atlas_run_id=handle.atlas_run_id,
            strategy_or_baseline=handle.strategy_or_baseline,
            stdout_log_path=handle.stdout_log_path,
            return_code=return_code,
            stop_requested=handle.stop_requested,
        )

    def _refresh_handle(self, handle: _JobHandle) -> None:
        handle.last_seen_at = datetime.now(timezone.utc)
        if handle.atlas_run_id is None:
            handle.atlas_run_id = self._discover_run_id(handle)
            if handle.atlas_run_id is not None:
                self._write_run_metadata(handle)
        if handle.process.poll() is not None and handle.runtime_config_path is not None:
            handle.runtime_config_path.unlink(missing_ok=True)
            handle.runtime_config_path = None
        self._write_job_metadata(handle)

    def _discover_run_id(self, handle: _JobHandle) -> str | None:
        if not self.artifacts_dir.exists():
            return None
        prefix = (
            f"-research_{handle.strategy_or_baseline}-"
            if handle.command_type == "research"
            else f"-{handle.strategy_or_baseline}-"
        )
        candidates: list[Path] = []
        for run_dir in self.artifacts_dir.iterdir():
            if not run_dir.is_dir() or prefix not in run_dir.name:
                continue
            created = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
            if created < handle.started_at:
                continue
            candidates.append(run_dir)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return candidates[0].name

    def _write_runtime_config(self, config_path: Path, overlay: dict[str, Any]) -> Path:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        research_section = payload.setdefault("research", {})
        research_section.update(overlay.get("research", {}))
        runtime_path = self.configs_dir / f"{uuid.uuid4().hex}.yaml"
        runtime_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return runtime_path

    def _write_job_metadata(self, handle: _JobHandle) -> None:
        status = "running"
        return_code = handle.process.poll()
        if return_code is not None:
            status = "stopped" if handle.stop_requested else ("completed" if return_code == 0 else "failed")
        payload = {
            "dashboard_job_id": handle.dashboard_job_id,
            "pid": handle.process.pid,
            "command_type": handle.command_type,
            "config_path": str(handle.config_path),
            "started_at": handle.started_at.isoformat(),
            "last_seen_at": handle.last_seen_at.isoformat(),
            "atlas_run_id": handle.atlas_run_id,
            "strategy_or_baseline": handle.strategy_or_baseline,
            "stdout_log_path": str(handle.stdout_log_path),
            "runtime_config_path": str(handle.runtime_config_path) if handle.runtime_config_path else None,
            "stop_requested": handle.stop_requested,
            "return_code": return_code,
            "status": status,
        }
        (self.jobs_dir / f"{handle.dashboard_job_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_run_metadata(self, handle: _JobHandle) -> None:
        if handle.atlas_run_id is None:
            return
        run_dir = self.artifacts_dir / handle.atlas_run_id
        if not run_dir.exists():
            return
        payload = {
            "dashboard_job_id": handle.dashboard_job_id,
            "pid": handle.process.pid,
            "command_type": handle.command_type,
            "stdout_log_path": str(handle.stdout_log_path),
            "started_at": handle.started_at.isoformat(),
            "last_seen_at": handle.last_seen_at.isoformat(),
            "status": "running" if handle.process.poll() is None else ("stopped" if handle.stop_requested else ("completed" if handle.process.returncode == 0 else "failed")),
        }
        (run_dir / "dashboard_job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _pump_pty_output(self, handle: _JobHandle) -> None:
        with handle.stdout_log_path.open("wb") as log_handle:
            while True:
                try:
                    chunk = os.read(handle.master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                log_handle.write(chunk)
                log_handle.flush()
        try:
            os.close(handle.master_fd)
        except OSError:
            pass


def build_default_registry(project_root: Path, config_path: Path) -> JobRegistry:
    config = load_config(config_path)
    artifacts_dir = (
        config.app.artifacts_dir
        if config.app.artifacts_dir.is_absolute()
        else project_root / config.app.artifacts_dir
    )
    return JobRegistry(project_root=project_root, artifacts_dir=artifacts_dir)
