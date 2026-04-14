from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from atlas.backtest.engine import BacktestEngine
from atlas.common.artifacts import ArtifactManager
from atlas.common.config import AppConfig, load_config
from atlas.common.logging import configure_logging
from atlas.data.binance import (
    fetch_and_save_binance_5s_dataset,
    fetch_and_save_binance_archive_5s_dataset,
)
from atlas.data.loaders import DatasetValidationError, load_market_data, validate_market_data
from atlas.research.experiment_runner import ExperimentRunner
from atlas.strategies import get_strategy_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    data_parser = subparsers.add_parser("data")
    data_subparsers = data_parser.add_subparsers(dest="action", required=True)
    validate_parser = data_subparsers.add_parser("validate")
    validate_parser.add_argument("--config", required=True)
    fetch_parser = data_subparsers.add_parser("fetch-binance")
    fetch_parser.add_argument("--symbol", default="BTCUSDT")
    fetch_parser.add_argument("--hours", type=int, default=1)
    fetch_parser.add_argument("--output", default="data/market/btc_perp_5s.csv")
    archive_parser = data_subparsers.add_parser("fetch-binance-archive")
    archive_parser.add_argument("--symbol", default="BTCUSDT")
    archive_parser.add_argument("--days", type=int, default=28)
    archive_parser.add_argument("--output", default="data/market/btc_perp_5s.csv")

    backtest_parser = subparsers.add_parser("backtest")
    backtest_subparsers = backtest_parser.add_subparsers(dest="action", required=True)
    run_backtest_parser = backtest_subparsers.add_parser("run")
    run_backtest_parser.add_argument("--config", required=True)
    run_backtest_parser.add_argument("--strategy", required=False)

    research_parser = subparsers.add_parser("research")
    research_subparsers = research_parser.add_subparsers(dest="action", required=True)
    run_research_parser = research_subparsers.add_parser("run")
    run_research_parser.add_argument("--config", required=True)
    run_research_parser.add_argument("--baseline", required=False)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config: AppConfig | None = None
    if not (args.domain == "data" and args.action in {"fetch-binance", "fetch-binance-archive"}):
        config = load_config(Path(args.config))
        configure_logging(config.app.log_level)
    else:
        configure_logging("INFO")

    if args.domain == "data" and args.action == "validate":
        assert config is not None
        return _run_validate(config)
    if args.domain == "data" and args.action == "fetch-binance":
        return _run_fetch_binance(args.symbol, args.hours, Path(args.output))
    if args.domain == "data" and args.action == "fetch-binance-archive":
        return _run_fetch_binance_archive(args.symbol, args.days, Path(args.output))
    if args.domain == "backtest" and args.action == "run":
        assert config is not None
        strategy_name = args.strategy or config.strategy.name
        return _run_backtest(config, strategy_name)
    if args.domain == "research" and args.action == "run":
        assert config is not None
        baseline = args.baseline or config.strategy.name
        return _run_research(config, baseline)

    parser.error("Unsupported command.")
    return 2


def _run_validate(config: AppConfig) -> int:
    try:
        data_frame, manifest, report = load_market_data(config.data)
    except DatasetValidationError as exc:
        print(json.dumps({"status": "error", "issues": [str(exc)]}, indent=2))
        return 1

    print(
        json.dumps(
            {
                "status": "ok" if not report.issues else "warning",
                "row_count": len(data_frame),
                "manifest": manifest.to_dict(),
                "issues": report.issues,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _run_fetch_binance(symbol: str, hours: int, output: Path) -> int:
    manifest = fetch_and_save_binance_5s_dataset(symbol=symbol, hours=hours, output_path=output)
    print(json.dumps({"status": "ok", "dataset": manifest}, indent=2))
    return 0


def _run_fetch_binance_archive(symbol: str, days: int, output: Path) -> int:
    manifest = fetch_and_save_binance_archive_5s_dataset(symbol=symbol, days=days, output_path=output)
    print(json.dumps({"status": "ok", "dataset": manifest}, indent=2))
    return 0


def _run_backtest(config: AppConfig, strategy_name: str) -> int:
    data_frame, manifest, report = load_market_data(config.data)
    if report.issues:
        print(json.dumps({"status": "warning", "issues": report.issues}, indent=2))
    strategy_cls = get_strategy_class(strategy_name)
    strategy = strategy_cls(config.strategy)
    engine = BacktestEngine(config)
    result = engine.run(data_frame, manifest, strategy, strategy_name=strategy_name)
    artifact_manager = ArtifactManager(config)
    artifact_dir = artifact_manager.write_backtest_run(result, config)
    print(json.dumps({"status": "ok", "artifact_dir": str(artifact_dir), "metrics": result.metrics}, indent=2))
    return 0


def _run_research(config: AppConfig, baseline_name: str) -> int:
    runner = ExperimentRunner(config)
    summary = runner.run(baseline_name)
    print(json.dumps(summary, indent=2))
    return 0
