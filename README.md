# atlas-gic

Deterministic BTC perpetual research framework for **5-second bars**.

This repository now contains a runnable MVP under [`src/atlas`](/Users/josefsmelhaus/Projects/atlas-gic/src/atlas) with:

- local OHLCV + funding CSV ingestion,
- deterministic next-bar-close backtests,
- conservative fee/slippage/latency assumptions,
- baseline mean reversion strategies,
- reproducible run artifacts,
- a bounded research loop that compares candidate strategies against a fixed evaluation harness.

Optional LLM mutation is implemented through the **Codex SDK** provider. In the MVP it generates bounded strategy parameter overrides, not arbitrary code edits.

## What Is In Scope

- One instrument at a time, default `BTCUSDT-PERP`
- One timeframe, default `5s`
- One conservative execution model: signal on bar `N`, fill on bar `N+1`
- Offline research only

## What Is Out of Scope

- Tick replay or order book simulation
- Paper trading or live execution
- The legacy `JANUS` and `MiroFish` prototypes

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Dataset Format

Place a CSV at `data/market/btc_perp_5s.csv` or point the config to another file.

Required columns:

```text
timestamp,open,high,low,close,volume,funding_rate
```

- `timestamp` must be parseable as UTC time.
- `funding_rate` is interpreted at the bar level.

## Commands

Validate the dataset:

```bash
python -m atlas data validate --config configs/base.yaml
```

Fetch a Binance USDⓈ-M futures dataset and derive `5s` bars from aggregate trades:

```bash
python -m atlas data fetch-binance --symbol BTCUSDT --hours 12 --output data/market/btc_perp_5s.csv
```

For an initial bootstrap dataset, start smaller to avoid Binance REST rate limits:

```bash
python -m atlas data fetch-binance --symbol BTCUSDT --hours 1 --output data/market/btc_perp_5s.csv
```

For multi-week history, prefer the official Binance daily archives:

```bash
python -m atlas data fetch-binance-archive --symbol BTCUSDT --days 28 --output data/market/btc_perp_5s.csv
```

Run a backtest:

```bash
python -m atlas backtest run --config configs/base.yaml --strategy btc_mean_reversion_v1
```

Run the research comparison loop:

```bash
python -m atlas research run --config configs/base.yaml --baseline btc_mean_reversion_v1
```

Artifacts are written to `artifacts/runs/<run_id>/`.

## Codex SDK Integration

The research loop can optionally ask a local Codex agent for bounded strategy mutations.

- Set `research.use_llm: true`
- Keep `research.provider: codex_sdk`
- Set `research.model` to a supported Codex model, default `gpt-5.4`
- Optionally export `ATLAS_CODEX_BIN` if your local Codex binary is not on `PATH`

The Python Codex SDK is experimental and relies on the local Codex app-server. If it is unavailable, the runner will raise an explicit setup error instead of silently falling back.

## Notes On Legacy Content

The files under [`src/janus.py`](/Users/josefsmelhaus/Projects/atlas-gic/src/janus.py) and [`src/mirofish`](/Users/josefsmelhaus/Projects/atlas-gic/src/mirofish) are preserved as legacy prototypes. They are intentionally not part of the MVP runtime path.

## Binance Note

Binance USDⓈ-M Futures REST klines support `1m` and larger intervals in the official API docs. For this MVP, `5s` bars are therefore derived from the official `aggTrades` endpoint instead of native kline data, and funding is distributed across bars inside each funding interval.

For longer lookbacks, prefer Binance archive dumps or another bulk historical source over the REST path, because `aggTrades` paging can hit rate limits on high-volume symbols.
