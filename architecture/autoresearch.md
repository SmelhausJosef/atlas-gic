# Autoresearch: Self-Improving Agent Prompts

## Concept

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), ATLAS uses a similar pattern for financial markets:

| Karpathy's Version | ATLAS Version |
|-------------------|---------------|
| Agent modifies `train.py` | Agent modifies prompt file |
| 5-minute GPU training | 5 trading days observation |
| Check validation loss | Check agent Sharpe ratio |
| Keep or revert | git commit or git reset |

The agent prompts are the weights being optimised. The loss function is the Sharpe ratio of the agent's recommendations.

## The Loop

```
┌─────────────────────────────────────────────────────────┐
│                   AUTORESEARCH LOOP                      │
│                                                          │
│  1. Identify worst-performing agent (lowest Sharpe)     │
│                         ↓                                │
│  2. Generate ONE targeted prompt modification           │
│                         ↓                                │
│  3. Create git feature branch                           │
│                         ↓                                │
│  4. Run for 5 trading days with modified prompt         │
│                         ↓                                │
│  5. Calculate new Sharpe ratio                          │
│                         ↓                                │
│  ┌─────────────────────────────────────────────────┐    │
│  │ IF new_sharpe > old_sharpe:                      │    │
│  │     git merge (keep modification)                │    │
│  │ ELSE:                                            │    │
│  │     git reset (revert to previous)               │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                                │
│  6. Repeat from step 1                                  │
└─────────────────────────────────────────────────────────┘
```

## Scoring Methodology

### Recommendation Tracking

Every agent recommendation is logged with:
- Ticker
- Direction (LONG/SHORT)
- Conviction (1-100)
- Entry price (at time of recommendation)
- Forward returns (tracked at 1d, 5d, 20d)

### Sharpe Calculation

For each agent, we calculate a rolling Sharpe ratio:

```python
def agent_sharpe(recommendations, lookback_days=60):
    returns = []
    for rec in recommendations[-lookback_days:]:
        # Weight return by conviction
        weighted_return = rec.forward_return * (rec.conviction / 100)
        # Flip sign for SHORT recommendations
        if rec.direction == 'SHORT':
            weighted_return *= -1
        returns.append(weighted_return)

    return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
```

### Agent Selection

The agent with the lowest rolling Sharpe is selected for modification. Agents cannot be modified more than once per 5-day period.

## Modification Generation

The modification prompt asks Claude to:

1. Review the agent's recent recommendations
2. Identify patterns in what went wrong
3. Propose ONE specific, targeted change to the prompt
4. The change should address the identified failure mode

Example modifications discovered during backtest:
- "Add momentum filter to prevent high-conviction longs during sector weakness"
- "Add DXY threshold check before any EM shorts"
- "Require technical confirmation before bullish calls"

## Darwinian Weights

Separate from autoresearch, each agent has a "Darwinian weight" that affects how much influence they have on final decisions.

### Weight Update Rules

After each trading day:
```python
for agent in agents:
    if agent in top_quartile_performers:
        agent.weight = min(2.5, agent.weight * 1.05)
    elif agent in bottom_quartile_performers:
        agent.weight = max(0.3, agent.weight * 0.95)
```

### Weight Boundaries
- **Ceiling**: 2.5 (maximum influence)
- **Floor**: 0.3 (near-silenced but not removed)
- **Starting**: 1.0 (neutral)

### Effect on CIO

The CIO weights each agent's recommendation:
```python
weighted_signal = sum(
    agent.recommendation * agent.darwinian_weight
    for agent in agents
) / sum(agent.darwinian_weight for agent in agents)
```

High-weight agents (2.5) have 8x the influence of minimum-weight agents (0.3).

## Results from 378-Day Backtest

### Autoresearch Stats
- Modifications attempted: 54
- Kept (improved Sharpe): 16 (30%)
- Reverted (no improvement): 37 (70%)

### Agents Most Modified
| Agent | Modifications | Kept |
|-------|--------------|------|
| Emerging Markets | 18 | 10 |
| Financials | 14 | 3 |
| Semiconductor | 7 | 2 |

### Notable Improvements
- Financials: Sharpe improved from -4.14 to 0.45
- Emerging Markets: Sharpe improved from -0.45 to -0.06
- Semiconductor: Sharpe improved from -0.26 to -0.06

### Final Darwinian Weights

**Maximum (2.5):**
Geopolitical, Commodities, Volatility, Energy, Industrials, Ackman, Aschenbrenner

**Minimum (0.3):**
CIO, Central Bank, Semiconductor, Institutional Flow

### Key Insight

The CIO (portfolio manager) was independently downweighted to 0.3 by the Darwinian system - the lowest possible weight. This revealed that signal generation was not the bottleneck; portfolio management was.

Individual agents improved through autoresearch. But without active management rules in the CIO, good signals didn't translate to good returns.

## Version Control Integration

Every modification is tracked in git:

```bash
# Create branch for modification
git checkout -b autoresearch/financials-momentum-filter

# Modify prompt file
# ... agent makes targeted change ...

# Commit
git commit -m "autoresearch: add momentum filter to financials"

# After 5 days, check result
if sharpe_improved:
    git checkout main
    git merge autoresearch/financials-momentum-filter
else:
    git checkout main
    git branch -D autoresearch/financials-momentum-filter
```

This creates a complete audit trail of what was tried and what worked.
