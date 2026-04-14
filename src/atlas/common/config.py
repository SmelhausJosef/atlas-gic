from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AppSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "atlas-5s-research"
    seed: int = 7
    artifacts_dir: Path = Path("artifacts/runs")
    log_level: str = "INFO"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    instrument: str
    venue: str
    timeframe: str = "5s"
    expected_bar_seconds: int = 5


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_equity: float = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    allow_short: bool = True


class CostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trading_fee_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    latency_penalty_bps: float = Field(ge=0)

    @property
    def total_bps(self) -> float:
        return self.trading_fee_bps + self.slippage_bps + self.latency_penalty_bps


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_leverage: float = Field(gt=0)
    max_position_abs: float = Field(gt=0, le=1.0)
    liquidation_buffer_pct: float = Field(gt=0, lt=1.0)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    lookback_bars: int = Field(ge=10)
    mean_window: int = Field(ge=2)
    std_window: int = Field(ge=2)
    vol_window: int = Field(ge=2)
    min_volatility: float = Field(ge=0)
    entry_zscore: float = Field(gt=0)
    exit_zscore: float = Field(ge=0)
    max_holding_bars: int = Field(ge=1)
    cooldown_bars: int = Field(ge=0)


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    use_llm: bool = False
    provider: str = "disabled"
    model: str = "disabled"
    llm_candidate_count: int = Field(default=1, ge=1, le=3)
    candidate_strategies: list[str] = Field(default_factory=list)
    improvement_epsilon: float = Field(ge=0)
    minimum_trade_count: int = Field(ge=0)
    max_drawdown_pct: float = Field(gt=0, lt=1.0)
    stressed_cost_multiplier: float = Field(ge=1.0)
    oos_fraction: float = Field(gt=0, lt=0.5)
    walk_forward_splits: int = Field(ge=2)
    max_generations: int = Field(default=1, ge=1, le=20)
    candidate_budget: int = Field(default=4, ge=1, le=200)
    max_stagnation_generations: int = Field(default=1, ge=1, le=20)
    strategy_memory_dir: Path = Path("artifacts/strategy_memory")

    @property
    def family_budget(self) -> int:
        return self.candidate_budget

    @property
    def tuning_iterations_per_family(self) -> int:
        return self.max_generations

    @property
    def max_family_stagnation_iterations(self) -> int:
        return self.max_stagnation_generations


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppSection
    data: DataConfig
    backtest: BacktestConfig
    costs: CostsConfig
    risk: RiskConfig
    strategy: StrategyConfig
    research: ResearchConfig


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return AppConfig.model_validate(payload)
