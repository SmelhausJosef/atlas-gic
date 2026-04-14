from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from atlas.common.models import TradeRecord
from atlas.evaluation.composite_score import composite_score
from atlas.evaluation.metrics import calculate_metrics


def test_metrics_include_expected_fields():
    equity_curve = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="5s", tz="UTC"),
            "equity": [100.0, 102.0, 101.0, 104.0],
            "position": [0, 1, 1, 0],
            "gross_exposure": [0.0, 1.0, 1.0, 0.0],
        }
    )
    trades = [
        TradeRecord(
            entry_timestamp=datetime.now(timezone.utc),
            exit_timestamp=datetime.now(timezone.utc),
            direction="LONG",
            quantity=1.0,
            entry_price=100.0,
            exit_price=102.0,
            pnl=2.0,
            pnl_pct=0.02,
            holding_bars=2,
            confidence=0.6,
            reason="test",
            tags=["test"],
        )
    ]
    metrics = calculate_metrics(equity_curve, trades, periods_per_year=1000, expected_bar_seconds=5)
    assert metrics["net_return"] > 0
    assert metrics["trade_count"] == 1.0
    assert composite_score(metrics) != 0

