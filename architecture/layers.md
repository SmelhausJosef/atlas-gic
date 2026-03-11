# ATLAS Layer Architecture

## Layer 1: Macro (10 Agents)

### Purpose
Assess the global macro environment and produce a regime signal that downstream layers use to calibrate risk.

### Agents

| Agent | Data Inputs | Output |
|-------|-------------|--------|
| **Central Bank** | Fed/ECB statements, rate decisions, dot plots | Policy stance, rate trajectory |
| **Geopolitical** | News feeds, conflict indicators, sanctions | Risk events, regional views |
| **China** | PMI, property data, stimulus announcements | China growth regime |
| **Dollar** | DXY, trade-weighted indices, yield differentials | Dollar direction |
| **Yield Curve** | 2Y/10Y spread, real rates, term premium | Recession probability |
| **Commodities** | Oil, gold, copper, agricultural | Inflation/growth signals |
| **Volatility** | VIX, MOVE, credit spreads, skew | Risk appetite |
| **Emerging Markets** | EM currencies, spreads, flows | EM regime |
| **News Sentiment** | Headlines, social media, earnings calls | Market mood |
| **Institutional Flow** | COT data, fund flows, 13F filings | Smart money positioning |

### Output Format
Each agent outputs:
```json
{
  "signal": "BULLISH | BEARISH | NEUTRAL",
  "conviction": 1-100,
  "rationale": "key reasoning"
}
```

### Aggregation
Signals are Darwinian-weighted and combined into a single regime:
- **RISK_ON**: Net bullish, increase equity exposure
- **RISK_OFF**: Net bearish, reduce exposure / hedge
- **NEUTRAL**: Mixed signals, maintain current positioning

---

## Layer 2: Sector Desks (7 Agents)

### Purpose
Identify the best long and short opportunities within each sector, informed by the macro regime.

### Agents

| Agent | Coverage | Key Metrics |
|-------|----------|-------------|
| **Semiconductor** | NVDA, AMD, AVGO, TSM, ASML, etc. | AI capex, inventory cycles |
| **Energy** | XOM, CVX, SLB, OXY, etc. | Oil prices, OPEC, production |
| **Biotech** | Large/mid cap biotech | Pipeline catalysts, FDA dates |
| **Consumer** | Discretionary and staples | Spending data, margins |
| **Industrials** | Defence, aerospace, machinery | Backlogs, capex cycles |
| **Financials** | Banks, insurance, asset managers | NIM, credit quality |
| **Relationship Mapper** | Cross-sector | Supply chains, ownership links |

### Data Inputs
- Macro regime from Layer 1
- Sector ETF performance
- Individual stock fundamentals
- Technical indicators
- Earnings calendar

### Output Format
```json
{
  "sector_view": "OVERWEIGHT | NEUTRAL | UNDERWEIGHT",
  "top_long": {"ticker": "XXXX", "conviction": 80, "thesis": "..."},
  "top_short": {"ticker": "YYYY", "conviction": 60, "thesis": "..."}
}
```

---

## Layer 3: Superinvestors (4 Agents)

### Purpose
Filter sector picks through different investment philosophies to ensure ideas match proven styles.

### Agents

| Agent | Philosophy | Focus |
|-------|------------|-------|
| **Druckenmiller** | Macro + momentum | Big asymmetric bets aligned with macro trends |
| **Aschenbrenner** | AI/compute thesis | Who benefits from the AI capex supercycle? |
| **Baker** | Deep tech/biotech | Real IP moats, defensible technology |
| **Ackman** | Quality compounder | Pricing power, FCF, clear catalysts |

### Data Inputs
- Current portfolio positions
- Sector desk recommendations
- Position P&L and holding periods

### Output Format
```json
{
  "portfolio_review": [
    {"ticker": "XXXX", "verdict": "HOLD | ADD | TRIM | EXIT", "rationale": "..."}
  ],
  "missing_name": {"ticker": "YYYY", "thesis": "..."}
}
```

---

## Layer 4: Decision (4 Agents)

### Purpose
Final synthesis, risk review, and execution.

### Agents

| Agent | Role |
|-------|------|
| **CRO** | Adversarial risk officer - attacks every idea |
| **Alpha Discovery** | Finds names not mentioned by other agents |
| **Autonomous Execution** | Converts signals to sized trades |
| **CIO** | Final synthesis and portfolio decision |

### CRO Function
The CRO receives all recommendations and actively tries to find reasons NOT to act:
- Concentration risk
- Correlation to existing positions
- Macro headwinds
- Valuation concerns
- Technical breakdown risk

Only ideas that survive CRO review reach the CIO.

### CIO Function
The CIO:
1. Receives all agent outputs weighted by Darwinian scores
2. Synthesises consensus and divergence
3. Makes final BUY/SELL/HOLD decisions
4. Determines position sizes
5. Manages overall portfolio exposure

### Output Format
```json
{
  "actions": [
    {"ticker": "XXXX", "action": "BUY", "shares": 100, "rationale": "..."}
  ],
  "portfolio_exposure": {"gross": 0.8, "net": 0.3},
  "risk_commentary": "..."
}
```

---

## Cross-Layer Communication

Layers communicate through structured JSON messages stored in a shared state directory:
```
data/state/
├── macro_regime.json      # Layer 1 output
├── sector_picks.json      # Layer 2 output
├── superinvestor_views.json  # Layer 3 output
├── cro_review.json        # Layer 4 risk review
└── portfolio_actions.json # Final CIO decisions
```

Each layer reads the previous layer's output and writes its own, creating an audit trail of the decision process.
