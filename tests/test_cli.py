from __future__ import annotations

from atlas.cli import main


def test_cli_smoke_commands(config_path):
    assert main(["data", "validate", "--config", str(config_path)]) == 0
    assert main(["backtest", "run", "--config", str(config_path), "--strategy", "btc_mean_reversion_v1"]) == 0
    assert main(["research", "run", "--config", str(config_path), "--baseline", "btc_mean_reversion_v1"]) == 0
