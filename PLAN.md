# PLAN

## Goal

Turn this fork into a practical AI-driven trading research framework where:

- AI proposes and refines strategies in a constrained way.
- A deterministic backtest engine evaluates every change with realistic costs and risk constraints.
- Only strategies that pass offline validation move to paper trading.
- Only strategies that survive forward validation are eligible for live deployment.

## Current State Assessment

This repository is currently a blueprint, not a runnable trading system.

What already exists:

- High-level architecture documents for multi-agent trading.
- Placeholder prompt examples.
- A partial `JANUS` weighting module.
- Example result artifacts.

What is missing for a real framework:

- A runnable market data pipeline.
- A deterministic backtest engine.
- Strategy interfaces and execution rules.
- Experiment orchestration.
- Validation gates.
- Paper trading and live deployment logic.
- Operational safety controls.

## Principles

1. AI may modify strategies, prompts, and bounded configuration only.
2. AI must never modify the scoring engine, slippage model, fee model, or risk guardrails.
3. Every experiment must be reproducible from code, config, dataset version, and seed.
4. Live trading must always be gated by deterministic rules outside the LLM.
5. Human approval should be required before first live deployment of any strategy family.

## Recommended Product Direction

I would not start by rebuilding the full ATLAS 25-agent architecture.

I would start with a narrower MVP:

- One market: BTC perpetuals.
- One venue adapter first.
- One bar size first, e.g. `1h`.
- One strategy family first, e.g. trend-following or mean reversion.
- One AI-editable strategy file.
- One fixed evaluation pipeline.

After that works, I would expand to:

- Multiple strategy families.
- Multiple market regimes.
- Ensemble selection.
- Optional higher-level agent orchestration.

## Phase 1 - Make the Repo Runnable

Create a real codebase under `src/` with concrete modules instead of architecture-only placeholders.

Planned modules:

- `src/data/`
- `src/backtest/`
- `src/strategies/`
- `src/evaluation/`
- `src/research/`
- `src/paper/`
- `src/live/`
- `src/common/`

First deliverables:

- Project configuration and environment management.
- Typed config objects.
- Structured logging.
- Deterministic experiment runner.
- Local artifact storage for results.

## Phase 2 - Build the Fixed Evaluation Harness

This is the most important phase. The harness must be trustworthy before AI starts iterating.

Implement:

- Historical candle ingestion.
- Funding, fee, and slippage modelling.
- Position accounting.
- Leverage and liquidation protection logic.
- Trade logs and equity curve generation.
- Baseline benchmark strategies.

Evaluation outputs per run:

- Net return.
- Sharpe and Sortino.
- Max drawdown.
- Win rate.
- Profit factor.
- Exposure stats.
- Turnover.
- Stability across time splits.

I would use a composite score with penalties so AI cannot over-optimize only one metric.

## Phase 3 - Define the Strategy Contract

Create a narrow strategy interface that AI is allowed to modify.

Suggested boundaries:

- AI can change signal logic.
- AI can change entry and exit rules.
- AI can change sizing within configured limits.
- AI can add filters and regime conditions.
- AI cannot change broker adapters, accounting, or evaluation code.

Example contract:

- Input: market features and current portfolio state.
- Output: target position, confidence, rationale metadata.

This keeps the editable surface area small and auditable.

## Phase 4 - Add the Autoresearch Loop

Recreate the useful part of `autoresearch`, but for strategies instead of training code.

Loop:

1. Select the current baseline strategy.
2. Ask the LLM for one bounded modification.
3. Save candidate on a git branch or as an experiment artifact.
4. Run the fixed backtest suite.
5. Compare against the baseline with hard acceptance rules.
6. Keep or reject.
7. Log the attempt.

Acceptance should require:

- Better composite score.
- No regression beyond thresholds on drawdown or turnover.
- Non-trivial number of trades.
- Improvement across multiple splits, not only one period.

## Phase 5 - Add Robust Validation

Before paper trading, every promoted strategy should pass:

- In-sample backtest.
- Out-of-sample backtest.
- Walk-forward validation.
- Sensitivity tests for fees and slippage.
- Monte Carlo reshuffling / bootstrap checks if applicable.
- Regime breakdown analysis.

If a strategy only works in a single narrow period, it should be rejected.

## Phase 6 - Paper Trading Layer

After offline validation, promote strategies to paper trading.

Paper trading should include:

- Real-time market data ingestion.
- Signal generation on schedule.
- Simulated fills with realistic assumptions.
- Strategy-level and portfolio-level logs.
- Forward performance dashboard.

Promotion to live should require a minimum paper-trading track record over a fixed time window.

## Phase 7 - Live Deployment with Hard Guardrails

Live deployment should be deliberately boring and rule-based.

Required controls:

- Max position size.
- Max gross and net exposure.
- Max leverage.
- Daily loss limit.
- Per-strategy kill switch.
- Exchange connectivity checks.
- Circuit breaker on missing data or stale prices.

The LLM should never have direct authority to bypass these controls.

## Phase 8 - Portfolio and Multi-Agent Expansion

Only after the single-strategy framework works would I add higher-order orchestration.

Then I would consider:

- Multiple BTC strategy families.
- Cohort training by regime.
- A JANUS-like selector to weight regimes or cohorts.
- Separate AI roles:
  - Strategy generator.
  - Risk reviewer.
  - Regime classifier.
  - Portfolio allocator.

I would still keep final live constraints deterministic.

## Suggested Near-Term Milestones

### Milestone 1

Produce a runnable BTC backtester with one hand-written baseline strategy and full metrics.

### Milestone 2

Add one AI-editable strategy file and a single experiment loop with keep/reject logic.

### Milestone 3

Add walk-forward validation and experiment logging.

### Milestone 4

Add paper trading.

### Milestone 5

Add live deployment with guardrails and small capital allocation.

## Files I Would Create First

- `src/data/loaders.py`
- `src/backtest/engine.py`
- `src/backtest/costs.py`
- `src/backtest/portfolio.py`
- `src/strategies/base.py`
- `src/strategies/btc_trend_v1.py`
- `src/evaluation/metrics.py`
- `src/evaluation/composite_score.py`
- `src/research/llm_mutator.py`
- `src/research/experiment_runner.py`
- `src/research/promotion.py`
- `configs/base.yaml`
- `configs/btc_1h.yaml`

## Risks I Would Address Early

- Backtest overfitting.
- Leakage from future data.
- Unrealistic slippage assumptions.
- Too much strategy edit freedom for the LLM.
- Regime changes breaking fragile strategies.
- Confusing prompt quality improvement with true trading edge.

## My Recommended Immediate Next Step

I would ignore most of the current marketing-style architecture for the first implementation pass and build the fixed BTC research harness first.

If the harness is not trustworthy, every later AI layer becomes noise.
