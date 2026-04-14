from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DatasetManifest:
    instrument: str
    venue: str
    timeframe: str
    path: Path
    date_start: datetime
    date_end: datetime
    row_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["date_start"] = self.date_start.isoformat()
        payload["date_end"] = self.date_end.isoformat()
        return payload


@dataclass(slots=True)
class ValidationReport:
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrategyDecision:
    target_position: float
    confidence: float
    reason: str
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not -1.0 <= self.target_position <= 1.0:
            raise ValueError("target_position must be in [-1, 1].")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty.")


@dataclass(slots=True)
class StrategyContext:
    bars: pd.DataFrame
    current_position: float
    gross_exposure: float
    equity: float
    bars_since_entry: int | None
    bars_since_exit: int | None
    expected_bar_seconds: int


@dataclass(slots=True)
class TradeRecord:
    entry_timestamp: datetime
    exit_timestamp: datetime
    direction: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    confidence: float
    reason: str
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entry_timestamp"] = self.entry_timestamp.isoformat()
        payload["exit_timestamp"] = self.exit_timestamp.isoformat()
        return payload


@dataclass(slots=True)
class BacktestResult:
    run_id: str
    strategy_name: str
    manifest: DatasetManifest
    metrics: dict[str, float]
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame
    execution_assumptions: str

