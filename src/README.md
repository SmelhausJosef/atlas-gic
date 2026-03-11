# Source Code

The ATLAS framework is implemented in Python. The core modules are:

## Architecture

```
src/
├── agents/
│   ├── backtest_loop.py      # Main training loop (378 iterations)
│   ├── eod_cycle.py          # Daily 25-agent debate pipeline
│   ├── market_data.py        # Data feeds with triple cross-validation
│   ├── scorecard.py          # Agent performance tracking and Sharpe calculation
│   └── autoresearch.py       # Prompt modification, keep/revert logic
│
├── prompts/
│   └── [25 agent prompt files]  # Trained prompts (proprietary)
│
├── data/
│   ├── state/                # Current portfolio state
│   ├── backtest/             # Historical backtest data
│   └── track_record/         # Performance tracking
│
└── utils/
    ├── llm.py                # Anthropic API wrapper
    ├── logging.py            # Structured logging
    └── git_ops.py            # Autoresearch git operations
```

## Key Components

### backtest_loop.py
The main training loop that runs one iteration per trading day:
- Loads historical market data for the current date
- Runs all 25 agents through the debate pipeline
- Scores recommendations against actual outcomes
- Updates Darwinian weights
- Triggers autoresearch if conditions met

### eod_cycle.py
The daily end-of-day cycle that orchestrates all agents:
- Layer 1: Macro agents run in parallel
- Layer 2: Sector desks receive macro regime
- Layer 3: Superinvestors filter sector picks
- Layer 4: CRO reviews, CIO decides

### autoresearch.py
The self-improvement engine:
- Identifies lowest-Sharpe agent
- Generates targeted prompt modification
- Creates git feature branch
- Tracks 5-day performance window
- Merges or reverts based on improvement

### scorecard.py
Agent performance tracking:
- Logs every recommendation with forward returns
- Calculates rolling Sharpe ratios
- Updates Darwinian weights daily
- Maintains historical score data

## Data Flow

```
Market Data (FMP/Finnhub/Polygon)
         ↓
    market_data.py
         ↓
    eod_cycle.py → Layer 1 → Layer 2 → Layer 3 → Layer 4
         ↓
    Portfolio Actions
         ↓
    scorecard.py → Agent Scores → Darwinian Weights
         ↓
    autoresearch.py → Prompt Modifications
```

## Not Included

The implementation details are proprietary. This directory describes the structure for reference.

**Not included in this repository:**
- Trained agent prompts
- API integration code
- Position management logic
- Risk management rules
- Deployment configuration

## Contact

For access to the full codebase or licensing inquiries:

**Chris Worsey**
chris@generalintelligencecapital.com
[generalintelligencecapital.com](https://generalintelligencecapital.com)
