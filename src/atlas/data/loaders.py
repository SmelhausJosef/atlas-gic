from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from atlas.common.config import DataConfig
from atlas.common.models import DatasetManifest, ValidationReport


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "funding_rate"]


class DatasetValidationError(ValueError):
    """Raised when a market dataset fails validation."""


def load_market_data(data_config: DataConfig) -> tuple[pd.DataFrame, DatasetManifest, ValidationReport]:
    path = data_config.path
    report = validate_market_data(path, data_config.expected_bar_seconds)
    if not path.exists():
        raise DatasetValidationError(f"Dataset not found: {path}")
    if any(issue.startswith("missing column") for issue in report.issues):
        raise DatasetValidationError("; ".join(report.issues))

    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "timestamp"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    manifest = DatasetManifest(
        instrument=data_config.instrument,
        venue=data_config.venue,
        timeframe=data_config.timeframe,
        path=path,
        date_start=_to_datetime(frame["timestamp"].iloc[0]),
        date_end=_to_datetime(frame["timestamp"].iloc[-1]),
        row_count=len(frame),
        sha256=_hash_file(path),
    )
    return frame, manifest, report


def validate_market_data(path: Path, expected_bar_seconds: int) -> ValidationReport:
    report = ValidationReport()
    if not path.exists():
        report.issues.append(f"dataset missing: {path}")
        return report

    frame = pd.read_csv(path)
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            report.issues.append(f"missing column: {column}")

    if report.issues:
        return report

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        report.issues.append("timestamp parse failure")
        return report

    if not timestamps.is_monotonic_increasing:
        report.issues.append("timestamps not sorted ascending")

    duplicate_count = int(timestamps.duplicated().sum())
    if duplicate_count:
        report.issues.append(f"duplicate timestamps: {duplicate_count}")

    diffs = timestamps.sort_values().diff().dropna()
    expected = pd.Timedelta(seconds=expected_bar_seconds)
    gap_count = int((diffs != expected).sum())
    if gap_count:
        report.issues.append(f"detected {gap_count} bar gaps relative to {expected_bar_seconds}s cadence")

    return report


def split_train_oos(frame: pd.DataFrame, oos_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(frame) * (1 - oos_fraction))
    split_index = max(split_index, 1)
    train = frame.iloc[:split_index].reset_index(drop=True)
    oos = frame.iloc[split_index:].reset_index(drop=True)
    return train, oos


def rolling_test_slices(frame: pd.DataFrame, splits: int) -> list[pd.DataFrame]:
    if splits < 2:
        return [frame]
    chunk_size = max(len(frame) // splits, 1)
    windows: list[pd.DataFrame] = []
    for index in range(splits):
        start = index * chunk_size
        end = len(frame) if index == splits - 1 else min((index + 1) * chunk_size, len(frame))
        window = frame.iloc[start:end].reset_index(drop=True)
        if len(window) > 1:
            windows.append(window)
    return windows


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_datetime(value: pd.Timestamp) -> datetime:
    return value.to_pydatetime()

