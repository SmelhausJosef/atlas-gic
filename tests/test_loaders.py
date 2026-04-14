from __future__ import annotations

import pandas as pd

from atlas.data.loaders import load_market_data, validate_market_data
from atlas.common.config import load_config


def test_load_market_data_builds_manifest(config_path):
    config = load_config(config_path)
    frame, manifest, report = load_market_data(config.data)
    assert len(frame) == manifest.row_count
    assert manifest.instrument == "BTCUSDT-PERP"
    assert report.issues == []
    assert len(manifest.sha256) == 64


def test_validate_market_data_detects_duplicates(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
            "funding_rate": [0, 0],
        }
    )
    path = tmp_path / "dupes.csv"
    frame.to_csv(path, index=False)
    report = validate_market_data(path, expected_bar_seconds=5)
    assert any("duplicate timestamps" in issue for issue in report.issues)

