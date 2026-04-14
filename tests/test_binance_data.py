from __future__ import annotations

import pandas as pd

from atlas.data.binance import _attach_funding_rates, _build_request_windows, _resample_agg_trades_to_5s


def test_resample_agg_trades_to_5s_builds_ohlcv():
    payload = [
        {"T": 1_700_000_000_000, "p": "100.0", "q": "1.0"},
        {"T": 1_700_000_000_500, "p": "101.0", "q": "2.0"},
        {"T": 1_700_000_004_900, "p": "99.0", "q": "1.5"},
        {"T": 1_700_000_005_100, "p": "102.0", "q": "1.0"},
    ]
    bars = _resample_agg_trades_to_5s(payload)
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 100.0
    assert bars.iloc[0]["high"] == 101.0
    assert bars.iloc[0]["low"] == 99.0
    assert bars.iloc[0]["close"] == 99.0
    assert bars.iloc[0]["volume"] == 4.5


def test_attach_funding_rates_distributes_events():
    bars = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:05Z",
                "2026-01-01T00:00:10Z",
                "2026-01-01T00:00:15Z",
            ],
            "open": [1, 1, 1, 1],
            "high": [1, 1, 1, 1],
            "low": [1, 1, 1, 1],
            "close": [1, 1, 1, 1],
            "volume": [1, 1, 1, 1],
            "funding_rate": [0.0, 0.0, 0.0, 0.0],
        }
    )
    funding = [{"fundingTime": 1_767_225_615_000, "fundingRate": "0.0004"}]
    enriched = _attach_funding_rates(bars, funding)
    assert round(enriched["funding_rate"].sum(), 10) == 0.0004


def test_build_request_windows_splits_range():
    start = pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime()
    end = pd.Timestamp("2026-01-01T00:30:00Z").to_pydatetime()
    windows = _build_request_windows(start, end, 15 * 60 * 1000)
    assert len(windows) == 2
    assert windows[0].start_ms < windows[0].end_ms
