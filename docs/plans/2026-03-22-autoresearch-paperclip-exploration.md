# AutoResearch + Paperclip Exploration for HRP

**Date:** 2026-03-22
**Status:** Exploration / Pre-RFC

## Context

Two external projects offer patterns that could significantly improve HRP's agent framework:

1. **[autoresearch](https://github.com/karpathy/autoresearch)** (Karpathy) — Autonomous AI-driven experiment iteration for LLM training. Core idea: give an agent a single editable file, a fixed time-budget experiment, a clear scalar metric, and let it iterate autonomously.

2. **[Paperclip](https://github.com/paperclipai/paperclip)** (paperclipai) — Open-source Node.js/React orchestration for multi-agent organizations. Treats agents as employees with org charts, budgets, heartbeats, and audit trails.

This document explores how to apply both to HRP's research pipeline.

---

## Part 1: AutoResearch Pattern Applied to HRP

### Core Insight

AutoResearch's power comes from a tight loop:

```
while time_remaining:
    mutate(strategy)        # Agent edits train.py
    result = evaluate()     # Fixed 5-min experiment
    if result > best:
        keep(strategy)
    else:
        revert(strategy)
```

HRP's current `SignalScientist` does a **single pass** IC scan. The `MLScientist` does a bounded parameter sweep (50 trials). Neither operates as an autonomous iteration loop where an LLM agent reasons about results and decides what to try next.

### Proposal: AutoResearch Loop for Signal Discovery

#### Design

```
┌─────────────────────────────────────────────────────┐
│                 AutoResearch Runner                  │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Strategy │───>│ Backtest │───>│ Evaluate │      │
│  │ Mutator  │    │ Runner   │    │ & Decide │      │
│  │ (LLM)    │<───│ (fixed   │<───│ (LLM)   │      │
│  │          │    │  budget) │    │          │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │                               │             │
│       v                               v             │
│  strategy.json                  leaderboard.json    │
│  (mutable)                      (append-only)       │
└─────────────────────────────────────────────────────┘
```

**Three components (mirroring autoresearch's three files):**

| AutoResearch | HRP Equivalent | Role |
|---|---|---|
| `prepare.py` (immutable) | `run_backtest()` + risk checks + kill gates | Fixed evaluation infrastructure |
| `train.py` (agent-editable) | `strategy.json` — feature selection, weights, signal logic, thresholds | The thing the agent mutates |
| `program.md` (agent instructions) | `research_program.md` — constraints, what to try, what metrics matter | Agent guidance |

#### `strategy.json` — The Mutable Artifact

```json
{
  "name": "momentum_vol_adjusted_v3",
  "features": ["momentum_20d", "volatility_60d"],
  "weights": [0.6, -0.4],
  "signal_method": "linear_combination",
  "lookback_days": 20,
  "top_pct": 0.10,
  "rebalance_frequency": "weekly",
  "stop_loss": {"type": "atr_trailing", "multiplier": 2.0}
}
```

The LLM agent can modify any field. The evaluation harness is fixed.

#### Fixed-Budget Experiment

Each iteration runs a **2-minute wall-clock backtest** (vs autoresearch's 5-min GPU training):
- Load 2 years of daily data for S&P 500
- Generate signals from `strategy.json`
- Run VectorBT backtest with IBKR costs
- Calculate: Sharpe, max drawdown, IC, stability score
- Log to leaderboard

**Throughput:** ~30 experiments/hour, ~200 overnight (8 hours).

#### Evaluation & Decision (LLM)

After each experiment, the LLM sees:
```
Experiment #17 results:
  Sharpe: 1.15 (prev best: 1.32)
  MaxDD: -18% (prev best: -15%)
  IC: 0.041 (prev best: 0.045)
  Stability: 0.72 (prev best: 0.68)

Leaderboard (top 5):
  #12: Sharpe=1.32, IC=0.045, features=[momentum_20d, rsi_14d]
  #8:  Sharpe=1.28, IC=0.043, features=[momentum_60d, volatility_20d]
  ...

What would you like to try next? Modify strategy.json.
```

The LLM decides: add a feature? change weights? try a different signal method? The key insight from autoresearch is that **the LLM learns from the trajectory of experiments**, not just one-shot.

### Proposal: AutoResearch Loop for ML Model Architecture

#### Design

Same pattern, different mutable artifact:

```json
{
  "model_type": "lightgbm",
  "features": ["momentum_20d", "volatility_60d", "rsi_14d"],
  "target": "returns_5d",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 50
  },
  "walk_forward": {
    "n_folds": 5,
    "train_months": 24,
    "test_months": 3,
    "purge_days": 5,
    "embargo_days": 2
  },
  "feature_engineering": {
    "interactions": [["momentum_20d", "volatility_60d"]],
    "transformations": {"rsi_14d": "rank"}
  }
}
```

**Budget:** 3-minute walk-forward validation per iteration (vs 2-min backtest for signals).

**Throughput:** ~20 experiments/hour, ~130 overnight.

**Guard rails (immutable):**
- Max 10 features (overfitting prevention)
- Max 500 estimators (compute budget)
- Stability score must be calculated (can't skip)
- Kill gates still apply to final candidates

### Integration with Existing Pipeline

The autoresearch loop slots in **before** the current pipeline:

```
AutoResearch Loop (overnight, autonomous)
    ├── Signal Discovery Loop → top 5 strategies
    └── ML Architecture Loop → top 3 model configs
         │
         v
Current Pipeline (daytime, event-driven)
    SignalScientist (now: validate top candidates from overnight)
    → AlphaResearcher → MLScientist → ... → Human CIO
```

The overnight loop produces **candidates**. The existing pipeline **validates** them with full rigor. This preserves all existing kill gates, risk checks, and human approval.

### Implementation Plan

| Step | What | Effort |
|------|------|--------|
| 1 | Define `strategy.json` and `model_config.json` schemas | S |
| 2 | Build `AutoResearchRunner` class with fixed-budget backtest | M |
| 3 | Build LLM mutation/evaluation prompts (`research_program.md`) | M |
| 4 | Add leaderboard persistence (DuckDB table or JSON) | S |
| 5 | Integrate with `SignalScientist` (feed overnight results) | S |
| 6 | Integrate with `MLScientist` (feed overnight model configs) | S |
| 7 | Add Streamlit leaderboard page | S |
| 8 | Overnight scheduling via launchd | S |

**Key files to create:**
- `hrp/agents/autoresearch_runner.py` — Main loop
- `hrp/agents/autoresearch_prompts.py` — LLM prompts for mutation/evaluation
- `hrp/research/strategy_schema.py` — Strategy JSON schema + validation
- `docs/agents/autoresearch-program.md` — Agent instructions (the `program.md` equivalent)

---

## Part 2: Paperclip as Orchestration Layer

### Current State: LineageEventWatcher

HRP's orchestration is a custom `LineageEventWatcher` that:
- Polls DuckDB `lineage` table every 60 seconds
- Matches events to registered callbacks
- Triggers downstream agents

**Strengths:** Simple, no external dependencies, tightly integrated with DuckDB.
**Weaknesses:** No real-time visibility, no budget dashboard, no approval UI, no cross-agent conversation tracing.

### Paperclip Architecture Mapping

```
Paperclip Concept          HRP Mapping
─────────────────          ───────────
Company                    HRP Platform
CEO Agent                  CIO Agent
Department Heads           Pipeline stage leads
Individual Contributors   Signal Scientist, ML Scientist, etc.
Board of Directors         Human CIO
Org Chart                  10-stage pipeline
Ticket System              Hypothesis lifecycle
Budget                     Token budget per SDKAgent
Heartbeat                  IngestionJob.run() + lineage events
Goal Ancestry              hypothesis_id → lineage chain
```

### Integration Architecture

Paperclip would replace `LineageEventWatcher` and `IngestionScheduler`, **not** the agents themselves.

```
┌─────────────────────────────────────────────┐
│              Paperclip Server               │
│           (Node.js, port 3100)              │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Org     │  │ Budget   │  │ Audit     │ │
│  │ Chart   │  │ Manager  │  │ Log       │ │
│  └─────────┘  └──────────┘  └───────────┘ │
│        │            │             │         │
│        v            v             v         │
│  ┌─────────────────────────────────────┐   │
│  │         Heartbeat / Events          │   │
│  └─────────────────────────────────┬───┘   │
└────────────────────────────────────│────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
     ┌────────v──────┐    ┌─────────v───────┐    ┌────────v──────┐
     │ Python Agent  │    │ Python Agent    │    │ Claude Code   │
     │ (heartbeat    │    │ (heartbeat      │    │ Session       │
     │  wrapper)     │    │  wrapper)       │    │ (native)      │
     │               │    │                 │    │               │
     │ SignalSci     │    │ MLScientist     │    │ AlphaResearch │
     │ KillGate      │    │ QuantDev        │    │ CIO Agent     │
     │ RiskMgr       │    │ QualitySentinel │    │ ReportGen     │
     └───────────────┘    └─────────────────┘    └───────────────┘
     (deterministic)      (deterministic)        (LLM-powered)
```

### Heartbeat Wrapper for Python Agents

Each deterministic Python agent gets a thin wrapper that speaks Paperclip's heartbeat protocol:

```python
class PaperclipHeartbeatWrapper:
    """Wraps an HRP ResearchAgent for Paperclip orchestration."""

    def __init__(self, agent: ResearchAgent, paperclip_url: str):
        self.agent = agent
        self.paperclip_url = paperclip_url

    def on_heartbeat(self, task: dict) -> dict:
        """Called by Paperclip when work is assigned."""
        result = self.agent.run()

        # Report back to Paperclip
        return {
            "status": "completed" if result.get("status") == "success" else "failed",
            "result": result,
            "cost": 0,  # deterministic agents have no token cost
        }
```

### What Changes, What Stays

| Component | Change? | Details |
|---|---|---|
| `LineageEventWatcher` | **Replace** | Paperclip handles event routing |
| `IngestionScheduler` | **Replace** | Paperclip handles scheduling |
| Lineage table | **Keep** | Still write events for audit trail + DuckDB queries |
| Agent `run()` pattern | **Keep** | Agents still have same execute() logic |
| `SDKAgent` token tracking | **Augment** | Report costs to Paperclip budget manager |
| Hypothesis lifecycle | **Keep** | Status machine unchanged |
| PlatformAPI | **Keep** | Agents still access data the same way |

### Paperclip Org Chart for HRP

```yaml
company: HRP Research Platform
ceo: cio-agent

departments:
  signal-discovery:
    head: signal-scientist
    members:
      - autoresearch-signal-loop    # New: overnight iteration
    budget: $50/month               # Compute + token budget

  model-development:
    head: ml-scientist
    members:
      - autoresearch-ml-loop        # New: overnight iteration
      - ml-quality-sentinel
    budget: $100/month

  validation:
    head: kill-gate-enforcer
    members:
      - quant-developer
      - validation-analyst
    budget: $75/month

  risk:
    head: risk-manager
    budget: $25/month

  executive:
    head: cio-agent
    members:
      - alpha-researcher
      - report-generator
    budget: $200/month

board:
  - human-cio                       # Approval authority
```

### Dual Event System

During migration, both systems coexist:

1. **Paperclip events** — Agent scheduling, heartbeats, budget, task assignment
2. **DuckDB lineage events** — Hypothesis lifecycle, audit trail, experiment tracking

Bridge: Paperclip task completion triggers a lineage event write via the heartbeat wrapper.

### Implementation Plan

| Step | What | Effort |
|------|------|--------|
| 1 | Install Paperclip locally, create HRP company | S |
| 2 | Build `PaperclipHeartbeatWrapper` for Python agents | M |
| 3 | Register all 10 agents in Paperclip org chart | S |
| 4 | Migrate scheduling from APScheduler to Paperclip triggers | M |
| 5 | Wire SDKAgent token costs to Paperclip budget reporting | S |
| 6 | Replace LineageEventWatcher event routing with Paperclip | M |
| 7 | Build bridge: Paperclip task completion → lineage event write | S |
| 8 | Add Paperclip dashboard link to Streamlit sidebar | S |

**Key files to create:**
- `hrp/agents/paperclip_wrapper.py` — Heartbeat wrapper for Python agents
- `hrp/agents/paperclip_bridge.py` — Paperclip ↔ DuckDB lineage bridge
- `paperclip/` — Paperclip config, org chart, agent definitions

**Key files to modify:**
- `hrp/agents/scheduler.py` — Remove `LineageEventWatcher`, add Paperclip client
- `hrp/agents/sdk_agent.py` — Report token costs to Paperclip

---

## Part 3: Combined Roadmap

### Phase 1: AutoResearch Loop (High Value, Low Risk)

Build the autonomous experiment iteration loop. This is independent of orchestration changes and delivers immediate value (10-100x more experiments per research cycle).

### Phase 2: Paperclip Orchestration (Medium Value, Medium Risk)

Replace `LineageEventWatcher` + `IngestionScheduler` with Paperclip. Gains: real-time dashboard, budget enforcement, conversation tracing. Risk: Node.js dependency, v0.3 maturity, dual event system complexity.

### Phase 3: Unified Control Plane

Once both are stable:
- Paperclip manages all agent lifecycle and scheduling
- AutoResearch loops run as Paperclip-managed agents with overnight budgets
- Human CIO uses Paperclip UI for approvals (replaces manual DB updates)
- Streamlit dashboard focuses on research results, not agent operations

---

## Open Questions

1. **AutoResearch LLM choice** — Use Claude (via existing SDKAgent) or a cheaper model (Haiku) for the mutation loop? At 200 iterations/night, token costs matter.

2. **Leaderboard persistence** — DuckDB table (consistent with everything else) or flat JSON files (simpler, autoresearch-style)?

3. **Paperclip PostgreSQL** — Run embedded (simple) or use an external Postgres instance? HRP is otherwise Postgres-free.

4. **Migration strategy** — Big bang switch from LineageEventWatcher to Paperclip, or gradual migration agent-by-agent?

5. **Approval UX** — Should Human CIO approvals move to Paperclip UI, or stay in Streamlit dashboard?
