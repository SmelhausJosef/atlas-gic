from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _build_market_frame(rows: int = 500) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5s", tz="UTC")
    phase = np.linspace(0, 20, rows)
    close = 100 + np.sin(phase) * 1.5 + np.cos(phase * 0.5) * 0.4
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    volume = 10 + np.abs(np.sin(phase)) * 2
    funding_rate = np.where(np.arange(rows) % 20 == 0, 0.00001, 0.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        }
    )


@pytest.fixture()
def market_frame() -> pd.DataFrame:
    return _build_market_frame()


@pytest.fixture()
def market_csv(tmp_path: Path, market_frame: pd.DataFrame) -> Path:
    path = tmp_path / "btc_5s.csv"
    market_frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def config_path(tmp_path: Path, market_csv: Path) -> Path:
    path = tmp_path / "config.yaml"
    payload = {
        "app": {"name": "test", "seed": 1, "artifacts_dir": str(tmp_path / "artifacts"), "log_level": "INFO"},
        "data": {
            "path": str(market_csv),
            "instrument": "BTCUSDT-PERP",
            "venue": "test",
            "timeframe": "5s",
            "expected_bar_seconds": 5,
        },
        "backtest": {"initial_equity": 10000.0, "periods_per_year": 6307200, "allow_short": True},
        "costs": {"trading_fee_bps": 1.0, "slippage_bps": 1.0, "latency_penalty_bps": 1.0},
        "risk": {"max_leverage": 1.5, "max_position_abs": 1.0, "liquidation_buffer_pct": 0.08},
        "strategy": {
            "name": "btc_mean_reversion_v1",
            "lookback_bars": 80,
            "mean_window": 20,
            "std_window": 20,
            "vol_window": 10,
            "min_volatility": 0.00005,
            "entry_zscore": 1.1,
            "exit_zscore": 0.2,
            "max_holding_bars": 20,
            "cooldown_bars": 2,
        },
        "research": {
            "enabled": True,
            "use_llm": False,
            "provider": "codex_sdk",
            "model": "gpt-5.4",
            "llm_candidate_count": 1,
            "candidate_strategies": ["btc_mean_reversion_tight_v1"],
            "improvement_epsilon": 0.0,
            "minimum_trade_count": 3,
            "max_drawdown_pct": 0.5,
            "stressed_cost_multiplier": 1.5,
            "oos_fraction": 0.25,
            "walk_forward_splits": 3,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path
