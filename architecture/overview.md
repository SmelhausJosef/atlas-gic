# ATLAS Architecture Overview

## System Design

ATLAS is a 4-layer multi-agent system where each layer progressively filters and refines trading signals.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LAYER 1: MACRO                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Central  │ │Geopolit-│ │  China  │ │ Dollar  │ │ Yield   │       │
│  │Bank     │ │ical     │ │         │ │         │ │ Curve   │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐       │
│  │Commodit-│ │Volatil- │ │Emerging │ │News     │ │Institut-│       │
│  │ies      │ │ity      │ │Markets  │ │Sentiment│ │ional    │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                              ↓                                       │
│                    REGIME: RISK_ON / RISK_OFF / NEUTRAL             │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       LAYER 2: SECTOR DESKS                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Semicond-│ │ Energy  │ │ Biotech │ │Consumer │ │Industri-│       │
│  │uctor    │ │         │ │         │ │         │ │als      │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│  ┌─────────┐ ┌───────────────────────────────────────────────┐     │
│  │Financi- │ │ Relationship Mapper (supply chain, ownership) │     │
│  │als      │ └───────────────────────────────────────────────┘     │
│  └─────────┘                                                        │
│                              ↓                                       │
│                    SECTOR PICKS: Long/Short per sector              │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     LAYER 3: SUPERINVESTORS                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │Druckenmiller│ │Aschenbrenner│ │   Baker     │ │   Ackman    │   │
│  │(Macro/Mom)  │ │(AI/Compute) │ │(Deep Tech)  │ │(Quality)    │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│                              ↓                                       │
│                    FILTERED PICKS: Philosophy-aligned ideas         │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       LAYER 4: DECISION                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │   CRO   │ │ Alpha   │ │Autonomo-│ │   CIO   │                   │
│  │(Risk)   │ │Discovery│ │us Exec  │ │(Final)  │                   │
│  └─────────┘ └─────────┘ └─────────┘ └────┬────┘                   │
│                                           ↓                          │
│                              PORTFOLIO ACTIONS                       │
│                         (BUY / SELL / HOLD + sizing)                │
└─────────────────────────────────────────────────────────────────────┘
```

## Information Flow

1. **Market Data** → Layer 1 agents receive macro data and produce regime signals
2. **Regime Signal** → Layer 2 sector desks filter opportunities based on macro backdrop
3. **Sector Picks** → Layer 3 superinvestors apply their philosophy filters
4. **Filtered Ideas** → Layer 4 synthesises all views (weighted by Darwinian scores) and executes

## Key Properties

- **Progressive Filtering**: Each layer reduces signal noise
- **Specialisation**: Agents focus on their domain of expertise
- **Weighted Synthesis**: Better-performing agents have more influence
- **Adversarial Review**: CRO attacks every idea before execution

## Agent Count

| Layer | Agents | Purpose |
|-------|--------|---------|
| Macro | 10 | Regime assessment |
| Sector | 7 | Stock selection |
| Superinvestor | 4 | Philosophy filtering |
| Decision | 4 | Execution |
| **Total** | **25** | |

## Daily Execution Cycle

```
06:00  Market data refresh
06:05  Layer 1 macro agents run in parallel
06:10  Regime signal published
06:15  Layer 2 sector desks run in parallel
06:25  Sector picks published
06:30  Layer 3 superinvestors run in parallel
06:40  Filtered ideas published
06:45  Layer 4 CRO reviews (adversarial)
06:50  CIO synthesises and decides
07:00  Trade execution (if market open)
```

## Technology

- **LLM**: Claude Sonnet via Anthropic API
- **Orchestration**: Python async pipeline
- **State**: JSON files for positions, scores, prompts
- **Version Control**: Git for prompt evolution tracking
