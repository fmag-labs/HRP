# Research Agents Design Brainstorm

**Date:** January 25, 2025
**Status:** Draft - Pending Implementation
**Related:** Tier 2 Intelligence (85% complete) - Research Agents feature

---

## Goal

Build a multi-agent quant research team that runs autonomously and coordinates through a shared workspace in the Cursor AI environment. Agents should simulate roles found at top quantitative hedge funds (Two Sigma, DE Shaw, Renaissance, Citadel).

---

## Design Decisions

### Coordination Model
**Decision:** Autonomous with shared workspace
- Each agent works independently
- Findings shared to common registry (hypotheses, MLflow, lineage)
- Other agents pick up relevant work
- User (CIO) reviews validated findings for final approval

### Execution Model
**Decision:** Both scheduled and on-demand
- Scheduled runs for regular operations (e.g., nightly discovery)
- Manual triggers via MCP tools for ad-hoc research

---

## Research Findings: Real Hedge Fund Structures

### Organizational Models

1. **Centralized/Collaborative** (DE Shaw, Two Sigma, RenTech)
   - Single cohesive team, shared knowledge base
   - Collaboration across disciplines
   - Best fit for our shared workspace model

2. **Pod Structure** (Citadel, Millennium)
   - Independent trading teams with separate P&Ls
   - Not suitable for personal research platform

### Key Insight: Risk Independence

> "The first sign of a powerless CRO is direct reporting to a single portfolio manager."
> — The Hedge Fund Journal

Risk management must be independent from alpha generation to prevent conflicts of interest.

### Sources
- [D.E. Shaw – the quant king](https://rupakghose.substack.com/p/de-shaw-the-quant-king)
- [The Hedge Fund Journal - Risk Practices](https://thehedgefundjournal.com/risk-practices-in-hedge-funds/)
- [QuantStart - Getting a Job at a Top Tier Quant Fund](https://www.quantstart.com/articles/Getting-a-Job-in-a-Top-Tier-Quant-Hedge-Fund/)
- [Street of Walls - Role of the Quantitative Analyst](https://www.streetofwalls.com/finance-training-courses/quantitative-hedge-fund-training/role-of-quant-analyst/)

---

## Original Proposal: 15 Agents (5 Pods)

```
CIO / Research Director (final authority)
│
├── Quantitative Research Pod
│   ├── Alpha Researcher (Quant)
│   ├── Feature & Signal Scientist
│   └── Regime & Market Structure Analyst
│
├── ML & Data Science Pod
│   ├── ML Scientist (Modeling)
│   ├── Training Reliability Auditor
│   └── Data Integrity & Leakage Sentinel
│
├── Engineering Pod
│   ├── Quant Developer (Infra & Performance)
│   ├── Pipeline & Experiment Orchestrator
│   └── Dashboard & UX Architect
│
├── Risk & Evaluation Pod
│   ├── Risk Manager Agent
│   ├── Execution & Cost Realism Analyst
│   └── Model Validation & Stress Tester
│
└── Deployment & Ops Pod
    ├── Paper Trading Operator
    ├── Live Trading Readiness Agent
    └── Monitoring & Kill-Switch Agent
```

### Strengths
- Good separation of concerns into pods
- Risk & Evaluation Pod is independent (matches best practice)
- ML Pod has specialized audit roles (Training Reliability, Leakage Sentinel)
- Regime Analyst addresses market condition awareness
- Forward-thinking Deployment Pod for Tier 4

### Concerns
1. **15 agents is high** - Coordination overhead, some may be idle
2. **Potential overlap** between agents (see consolidation analysis)
3. **Deployment Pod premature** - HRP at Tier 2, paper trading is Tier 4
4. **Dashboard & UX Architect** - Project work, not ongoing agent

---

## Consolidation Analysis

### Quantitative Research Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Alpha Researcher | Keep | Core hypothesis development |
| Feature & Signal Scientist | Keep | Distinct: signal discovery vs strategy design |
| Regime & Market Structure Analyst | Merge into Alpha Researcher | Can be a capability, not separate agent |

**Result:** 3 → 2 agents

### ML & Data Science Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| ML Scientist | Keep | Core modeling work |
| Training Reliability Auditor | Merge | Both are ML quality control |
| Data Integrity & Leakage Sentinel | Merge | Same category |

**Merged into:** ML Quality Sentinel (training reliability + leakage detection + data integrity)

**Result:** 3 → 2 agents

### Engineering Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Quant Developer | Keep | Core implementation |
| Pipeline & Experiment Orchestrator | Merge into Quant Dev | Infrastructure is Quant Dev territory |
| Dashboard & UX Architect | Remove | Project work, not ongoing agent |

**Result:** 3 → 1 agent

### Risk & Evaluation Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Risk Manager | Keep | Core independent oversight |
| Execution & Cost Realism Analyst | Merge | Both validate real-world viability |
| Model Validation & Stress Tester | Merge | Same validation focus |

**Merged into:** Validation Analyst (stress testing + execution realism + model validation)

**Result:** 3 → 2 agents

### Deployment & Ops Pod

| Original | Recommendation | Reasoning |
|----------|---------------|-----------|
| Paper Trading Operator | Defer | Tier 4 functionality |
| Live Trading Readiness Agent | Defer | Tier 4 functionality |
| Monitoring & Kill-Switch Agent | Defer | Tier 4 functionality |

**Result:** 3 → 0 agents (deferred to Tier 4)

---

## Proposed Options

### Option A: 8 Agents (Lean) - RECOMMENDED

```
You (CIO/Research Director)
│
├── Alpha Researcher (with regime awareness)
├── Signal Scientist
│
├── ML Scientist
├── ML Quality Sentinel
│
├── Quant Developer
│
├── Risk Manager
├── Validation Analyst
│
└── Report Generator
```

**Why 8 agents:**
- Each agent has distinct, full-time job
- No idle agents waiting for work
- Covers full research lifecycle: Discovery → Modeling → Implementation → Risk → Reporting
- Matches current HRP platform capabilities
- Easy to expand later

### Option B: 10 Agents (Moderate)

```
You (CIO/Research Director)
│
├── Alpha Researcher
├── Signal Scientist
├── Regime Analyst (kept separate)
│
├── ML Scientist
├── ML Quality Sentinel
│
├── Quant Developer
│
├── Risk Manager
├── Validation Analyst
│
├── Report Generator
└── Deployment Monitor (placeholder)
```

### Option C: 12 Agents (Full but streamlined)

```
You (CIO/Research Director)
│
├── Alpha Researcher
├── Signal Scientist
├── Regime Analyst
│
├── ML Scientist
├── ML Quality Sentinel
│
├── Quant Developer
├── Pipeline Orchestrator (kept separate)
│
├── Risk Manager
├── Validation Analyst
│
├── Report Generator
└── Deployment Monitor
```

---

## Agent Role Definitions (Option A)

### 1. Alpha Researcher
**Focus:** Hypothesis development, strategy design, regime awareness
**Inputs:** Market data, features, existing hypotheses
**Outputs:** New hypotheses with thesis, prediction, falsification criteria
**Schedule:** Weekly discovery runs + on-demand
**Uses:** `create_hypothesis`, `get_features`, `get_prices`, `run_backtest`

### 2. Signal Scientist
**Focus:** Feature engineering, signal discovery, predictive pattern identification
**Inputs:** Price data, existing features, alternative data
**Outputs:** New feature definitions, signal strength reports, IC analysis
**Schedule:** Weekly scans + on-demand
**Uses:** `get_available_features`, `get_features`, feature computation APIs

### 3. ML Scientist
**Focus:** Model development, training, hyperparameter optimization
**Inputs:** Features, targets, hypothesis specifications
**Outputs:** Trained models, walk-forward validation results
**Schedule:** On-demand (triggered by new hypotheses)
**Uses:** `train_ml_model`, `run_walk_forward_validation`, `get_supported_models`

### 4. ML Quality Sentinel
**Focus:** Training reliability, data leakage detection, overfitting prevention
**Inputs:** Training data, model outputs, feature sets
**Outputs:** Audit reports, leakage warnings, reliability scores
**Schedule:** Runs after every ML training job
**Uses:** Overfitting guards, leakage validators, Sharpe decay monitoring

### 5. Quant Developer
**Focus:** Backtest implementation, code quality, performance optimization, pipeline orchestration
**Inputs:** Strategy specifications, model outputs
**Outputs:** Optimized backtests, performance reports, infrastructure improvements
**Schedule:** On-demand + monitors experiment queue
**Uses:** `run_backtest`, MLflow APIs, feature computation

### 6. Risk Manager
**Focus:** Position limits, drawdown monitoring, portfolio risk, independent oversight
**Inputs:** Backtest results, portfolio state, market conditions
**Outputs:** Risk reports, limit violations, strategy vetoes
**Schedule:** Continuous monitoring + reviews before deployment approval
**Uses:** Risk validation APIs, strategy metrics

### 7. Validation Analyst
**Focus:** Stress testing, execution realism, model validation, out-of-sample testing
**Inputs:** Strategies, backtest results, market scenarios
**Outputs:** Stress test reports, execution cost estimates, validation verdicts
**Schedule:** Runs on all strategies before promotion to "validated"
**Uses:** Parameter sensitivity, robustness testing, statistical validation

### 8. Report Generator
**Focus:** Synthesize findings, create human-readable summaries
**Inputs:** All agent outputs, hypothesis status, experiment results
**Outputs:** Weekly research reports, hypothesis summaries, action recommendations
**Schedule:** Weekly + on-demand
**Uses:** `get_lineage`, `list_hypotheses`, `get_experiment`, `analyze_results`

---

## Shared Workspace: How Agents Coordinate

### Registry Points (existing HRP infrastructure)
1. **Hypothesis Registry** - Agents create/update hypotheses with status
2. **MLflow Experiments** - All training/backtest results logged
3. **Lineage System** - Full audit trail with actor tracking
4. **Feature Store** - Shared feature definitions and values

### Workflow Example
```
1. Signal Scientist discovers promising momentum signal
   → Creates draft hypothesis HYP-2025-042

2. Alpha Researcher picks up HYP-2025-042
   → Refines thesis, adds falsification criteria
   → Updates status to "testing"

3. ML Scientist sees "testing" hypothesis
   → Runs walk-forward validation
   → Logs results to MLflow

4. ML Quality Sentinel audits the training
   → Checks for leakage, overfitting
   → Adds audit report to lineage

5. Validation Analyst runs stress tests
   → Parameter sensitivity, regime analysis
   → Updates hypothesis with validation results

6. Risk Manager reviews validated hypothesis
   → Checks risk limits, portfolio fit
   → Approves or flags concerns

7. Report Generator summarizes for CIO review
   → Weekly report includes HYP-2025-042 findings
   → CIO decides whether to deploy
```

---

## Implementation Considerations

### Agent Infrastructure
- Extend existing `IngestionJob` pattern or create new `ResearchAgent` base class
- Each agent needs: scheduled execution, MCP tool access, logging, error handling
- Consider using Claude Agent SDK for agent implementation

### MCP Integration
- Agents use existing 22 MCP tools
- May need additional tools for agent-to-agent communication
- Actor tracking already supports agent identification

### Permissions
- Agents cannot deploy strategies (existing rule)
- Risk Manager can veto but not approve deployment
- Only CIO (user) has final approval authority

---

## Next Steps

1. [ ] Decide on Option A, B, or C
2. [ ] Define detailed specifications for each agent
3. [ ] Design agent base class / infrastructure
4. [ ] Implement agents incrementally (start with 2-3 core agents)
5. [ ] Test coordination through shared workspace
6. [ ] Add scheduling and MCP triggers
7. [ ] Expand to full team

---

## Open Questions

1. **Agent implementation:** Claude Agent SDK vs custom implementation?
2. **Scheduling:** APScheduler (existing) vs separate agent orchestrator?
3. **Communication:** Direct agent-to-agent or only via shared registries?
4. **Priority:** Which 2-3 agents to build first?

---

## Document History

- **2025-01-25:** Initial brainstorm captured from conversation
