from __future__ import annotations

from atlas.backtest.engine import BacktestEngine
from atlas.common.artifacts import ArtifactManager
from atlas.common.config import load_config
from atlas.data.loaders import load_market_data
from atlas.strategies import get_strategy_class


def test_backtest_is_deterministic(config_path):
    config = load_config(config_path)
    frame, manifest, _ = load_market_data(config.data)
    strategy_cls = get_strategy_class("btc_mean_reversion_v1")
    engine = BacktestEngine(config)

    result_a = engine.run(frame, manifest, strategy_cls(config.strategy), "btc_mean_reversion_v1")
    result_b = engine.run(frame, manifest, strategy_cls(config.strategy), "btc_mean_reversion_v1")

    assert result_a.metrics == result_b.metrics
    assert len(result_a.trades) == len(result_b.trades)


def test_artifact_bundle_contains_expected_files(config_path):
    config = load_config(config_path)
    frame, manifest, _ = load_market_data(config.data)
    strategy_cls = get_strategy_class("btc_mean_reversion_v1")
    engine = BacktestEngine(config)
    result = engine.run(frame, manifest, strategy_cls(config.strategy), "btc_mean_reversion_v1")
    manager = ArtifactManager(config)
    artifact_dir = manager.write_backtest_run(result, config)

    assert (artifact_dir / "config.json").exists()
    assert (artifact_dir / "dataset_manifest.json").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "trades.csv").exists()
    assert (artifact_dir / "equity.csv").exists()
    assert (artifact_dir / "summary.md").exists()

