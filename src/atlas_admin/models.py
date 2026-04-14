from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetOverview:
    path: Path
    instrument: str
    venue: str
    timeframe: str
    row_count: int
    date_start: str
    date_end: str
    sha256: str
    issues: list[str]


@dataclass(slots=True)
class JobSnapshot:
    dashboard_job_id: str
    command_type: str
    status: str
    config_path: Path
    started_at: datetime
    last_seen_at: datetime
    atlas_run_id: str | None = None
    strategy_or_baseline: str | None = None
    stdout_log_path: Path | None = None
    return_code: int | None = None
    stop_requested: bool = False


@dataclass(slots=True)
class RunListItem:
    run_id: str
    title: str
    run_type: str
    status: str
    started_at: str | None
    updated_at: str | None
    strategy: str | None
    baseline_name: str | None
    stop_reason: str | None
    overall_score: float | None
    dashboard_job_id: str | None = None
    stdout_log_path: str | None = None


@dataclass(slots=True)
class RunDetail:
    run_id: str
    title: str
    run_type: str
    status: str
    started_at: str | None
    updated_at: str | None
    strategy: str | None
    baseline_name: str | None
    stop_reason: str | None
    overall_score: float | None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    dataset_manifest: dict[str, Any] = field(default_factory=dict)
    heartbeat: dict[str, Any] | None = None
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    family_summaries: list[dict[str, Any]] = field(default_factory=list)
    strategy_memory_records: list[dict[str, Any]] = field(default_factory=list)
    summary_text: str | None = None
    artifact_files: list[str] = field(default_factory=list)
    equity_points: list[dict[str, Any]] = field(default_factory=list)
    log_preview: str | None = None
    dashboard_job_id: str | None = None
    stdout_log_path: str | None = None
