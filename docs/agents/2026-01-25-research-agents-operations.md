# Research Agents: Day-to-Day Operations Projection

**Date:** January 25, 2026
**Status:** Draft - Brainstorm Documentation
**Related:** [Research Agents Design](2026-01-25-research-agents-design.md)

---

## Overview

This document projects how the **10-agent research team** would operate day-to-day, leveraging HRP platform capabilities. It covers the ML experimentation pipeline, timeline from cold start to presented strategies, and the CIO Agent's autonomous decision-making role.

**Agents:**
1. Signal Scientist - Feature discovery and hypothesis creation
2. Alpha Researcher - Strategy generation and hypothesis refinement
3. ML Scientist - Model training and walk-forward validation
4. ML Quality Sentinel - Experiment auditing (overfitting, leakage)
5. Quant Developer - Strategy backtesting
6. **Pipeline Orchestrator** - Baseline execution, parallel experiments, kill gates
7. Validation Analyst - Stress testing and robustness checks
8. Risk Manager - Portfolio-level risk review
9. Report Generator - Research synthesis and reporting
10. **CIO Agent** - Autonomous hypothesis scoring and deployment decisions

---

## Daily Operations: The Research Machine

### 6:00 PM ET - Market Close Cascade

**Automated Pipeline (existing infrastructure):**
```
IngestionScheduler triggers:
├── 6:00 PM  PriceIngestionJob    → Fresh OHLCV for S&P 500
├── 6:05 PM  UniverseUpdateJob    → Check for index changes
├── 6:10 PM  FeatureComputationJob → Recalculate all 44 features
└── 6:15 PM  DataQualityJob       → Validate data, send alerts if issues
```

**Then the agents wake up:**

### 6:30 PM - ML Quality Sentinel
- Scans today's feature computations for anomalies
- Checks for data drift vs historical distributions
- Flags anything suspicious → writes to lineage with `actor='agent:ml-quality-sentinel'`

### 7:00 PM - Signal Scientist
- Runs nightly signal scans using fresh features
- Tests correlation of features vs forward returns
- If IC > threshold on any signal:
  ```python
  api.create_hypothesis(
      title="RSI divergence predicts 5-day returns",
      thesis="...",
      prediction="IC > 0.03 sustained",
      falsification="IC < 0.01 or unstable across regimes",
      actor='agent:signal-scientist'
  )
  ```
- Logs findings to MLflow experiment `signal-scans/YYYY-MM-DD`

---

## Weekly Cycle

| Day | Agent Activity |
|-----|----------------|
| **Monday 7 PM** | Alpha Researcher generates strategies, reviews hypotheses, refines falsification criteria |
| **Tuesday 6 AM** | Pipeline Orchestrator runs baseline experiments, parallel testing, kill gates |
| **Wednesday** | ML Scientist runs walk-forward validation on `testing` hypotheses |
| **Thursday** | Validation Analyst stress tests + Risk Manager reviews |
| **Friday** | **CIO Agent scores validated hypotheses** + Report Generator compiles weekly summary |
| **Saturday** | Fundamentals ingestion (weekly job) |
| **Sunday** | System idle |

### Monday - Alpha Researcher
**Strategy Generation (NEW):**
- Generates 3-5 complete strategy specifications per week from multiple sources:
  - **Signal Analysis:** Transforms high-IC features into strategy concepts
  - **Academic Literature:** Adaptations of published factor research
  - **Market Regimes:** Strategies targeting specific volatility/liquidity conditions
- Writes comprehensive strategy spec documents to `docs/strategies/STRATEGY-NAME.md`
- Fires `ALPHA_RESEARCHER_COMPLETE` event to trigger Pipeline Orchestrator

**Hypothesis Review:**
- Reviews all draft hypotheses from Signal Scientist
- Adds regime awareness: "Does this signal work in high-vol vs low-vol?"
- Refines falsification criteria to be testable
- Promotes promising ones to `status='testing'`

### Tuesday - Pipeline Orchestrator
**Baseline Execution:**
- Automatically generates baseline experiments for new strategies:
  - Multiple model types: ridge, lasso, random_forest, lightgbm
  - Standard feature combinations: single factor, multi-factor
  - Default hyperparameters: no tuning (speed over optimization)
- Executes 8-12 baseline experiments per strategy (4 models × 2-3 feature sets)
- Parallel processing: all models run simultaneously using `n_jobs=-1`

**Parallel Experimentation:**
- Runs all baselines in parallel across available CPU cores
- Typical runtime: 15-30 minutes for full baseline suite
- Logs all results to MLflow with `stage='baseline'` tag
- Tracks experiment lineage: `strategy_name → model_type → features → metrics`

**Kill Gates (Early Termination):**
After each baseline completes:
1. **Statistical Kill Gate:** Sharpe < 0.5 or IC < 0.02 → KILL
2. **Stability Kill Gate:** stability_score > 1.5 → KILL
3. **Overfitting Kill Gate:** Sharpe decay > 60% → KILL
4. **Leakage Kill Gate:** Target correlation > 0.95 → KILL

**Decisions:**
- **PASS:** All gates cleared → Promote to `status='testing'` for ML Scientist
- **FAIL:** Any gate triggered → Mark as `status='rejected'` with reason
- **CONDITIONAL:** Marginal results (e.g., Sharpe 0.45-0.5) → Flag for review

**Output:**
- Summary report: `docs/reports/YYYY-MM-DD/baseline-STRATEGY-NAME.md`
- Database updates: hypothesis status, experiment results, kill gate decisions
- Lineage events: `PIPELINE_ORCHESTRATOR_PASS` or `PIPELINE_ORCHESTRATOR_KILL`

### Wednesday - ML Scientist
- Picks up hypotheses in `testing` status
- Runs walk-forward validation:
  ```python
  result = walk_forward_validate(
      config=WalkForwardConfig(
          model_type='ridge',
          features=['momentum_20d', 'rsi_14d'],
          n_folds=5,
          n_jobs=-1,  # Parallel processing
      ),
      symbols=universe,
      log_to_mlflow=True,
  )
  ```
- Updates hypothesis with results, stability scores

### Thursday - Validation Analyst + Risk Manager

**Validation Analyst:**
- Runs parameter sensitivity on validated models
- Stress tests across market regimes (2008, 2020, 2022)
- Checks execution realism (can we actually trade this?)

**Risk Manager (independent):**
- Reviews all strategies approaching `validated` status
- Checks portfolio-level impact if strategy deployed
- Can flag concerns or veto, but **cannot approve**

### Friday - Report Generator + CIO Agent

**CIO Agent runs autonomous scoring:**
```python
# Score all validated hypotheses
for hypothesis in validated_hypotheses:
    score = cio_agent.score_hypothesis(
        hypothesis_id=hypothesis.id,
        experiment_data=extract_ml_metrics(hypothesis),
        risk_data=extract_risk_metrics(hypothesis),
        economic_data=extract_economic_context(hypothesis),
        cost_data=extract_execution_costs(hypothesis),
    )

    # Log decision to lineage
    if score.decision == "CONTINUE":
        mark_for_deployment(hypothesis.id)
    elif score.decision == "KILL":
        update_hypothesis_status(hypothesis.id, "rejected")
```

**Report Generator produces:**
```
Weekly Research Report - Week of Jan 20, 2026

HYPOTHESES SUMMARY
- 3 new hypotheses created (Signal Scientist)
- 2 promoted to testing (Alpha Researcher)
- 1 passed walk-forward (ML Scientist)
- 1 passed validation (Validation Analyst)
- 0 vetoed by Risk Manager

CIO AGENT DECISIONS
┌─────────────────────────────────────────────────────┐
│ HYP-2026-042: Momentum + Low Vol Factor            │
│ Decision: CONTINUE ✓ (0.78)                        │
│ Statistical: 0.82 | Risk: 0.75 | Economic: 0.70   │
│ Status: Awaiting human CIO approval for paper      │
├─────────────────────────────────────────────────────┤
│ HYP-2026-043: RSI Mean Reversion                   │
│ Decision: CONDITIONAL (0.65)                       │
│ Statistical: 0.70 | Risk: 0.50 | Economic: 0.80   │
│ Action: Refine risk controls, retest              │
├─────────────────────────────────────────────────────┤
│ HYP-2026-044: Multi-factor Composite               │
│ Decision: KILL (0.42)                              │
│ Reason: Poor statistical significance, Sharpe decay │
└─────────────────────────────────────────────────────┘

EXPERIMENTS THIS WEEK
- 12 walk-forward runs
- 847 hyperparameter trials tracked
- 2 leakage warnings (addressed)

NEXT WEEK PRIORITIES
- Signal Scientist: Test alternative data signals
- Alpha Researcher: Investigate sector momentum
```

---

## ML Experimentation: Who Does What

| Stage | Agent | What They Do | HRP Tools Used |
|-------|-------|--------------|----------------|
| **1. Signal Discovery** | Signal Scientist | Tests feature predictiveness (IC analysis) | `get_features`, correlation scans |
| **2. Strategy Generation** | Alpha Researcher | Generates strategy specs, formalizes hypotheses | `create_hypothesis`, strategy spec documents |
| **3. Baseline Experiments** | Pipeline Orchestrator | Runs parallel baseline tests, applies kill gates | `run_backtest`, parallel execution, kill gates |
| **4. Model Training** | ML Scientist | Trains models, tunes hyperparameters | `walk_forward_validate`, MLflow |
| **5. Training Audit** | ML Quality Sentinel | Checks leakage, overfitting, Sharpe decay | `HyperparameterTrialCounter`, `TargetLeakageValidator` |
| **6. Strategy Backtest** | Quant Developer | Runs full backtest with realistic costs | `run_backtest`, `generate_ml_predicted_signals` |
| **7. Stress Testing** | Validation Analyst | Parameter sensitivity, regime stress | `check_parameter_sensitivity` |
| **8. Risk Review** | Risk Manager | Portfolio fit, drawdown limits | `validate_strategy` |
| **9. CIO Decision** | **CIO Agent** | Scores across 4 dimensions, makes deployment decision | `CIOAgent.score_hypothesis()` |

---

## ML Experimentation Loop (Detail)

```
Signal Scientist                    Alpha Researcher
     │                                   │
     │ "momentum_20d has IC=0.04"        │
     │                                   │
     ▼                                   ▼
Alpha Researcher                 Strategy Generation
     │                                   │
     │ Creates HYP-2026-001              │ Generates strategy spec
     │ status='testing'                  │ docs/strategies/STRATEGY.md
     │                                   │
     └───────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │      PIPELINE ORCHESTRATOR            │
        │                                       │
        │  1. Generate 8-12 baseline experiments│
        │     - 4 model types × 2-3 feature sets│
        │     - Default hyperparameters         │
        │                                       │
        │  2. Run all baselines in parallel     │
        │     - n_jobs=-1 (all cores)           │
        │     - Runtime: 15-30 min              │
        │                                       │
        │  3. Apply Kill Gates:                 │
        │     ✓ Sharpe < 0.5 → KILL             │
        │     ✓ IC < 0.02 → KILL                │
        │     ✓ Stability > 1.5 → KILL          │
        │     ✓ Sharpe decay > 60% → KILL       │
        │     ✓ Target leakage > 0.95 → KILL    │
        │                                       │
        │  4. Decision:                         │
        │     - PASS → Promote to ML Scientist  │
        │     - FAIL → Reject with reason       │
        │     - CONDITIONAL → Flag for review   │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼ (if PASS)
        ┌───────────────────────────────────────┐
        │         ML SCIENTIST LOOP             │
        │                                       │
        │  for model_type in [ridge, lasso,    │
        │                      rf, lightgbm]:   │
        │                                       │
        │    for features in feature_combos:   │
        │                                       │
        │      walk_forward_validate(          │
        │        model_type=model_type,        │
        │        features=features,            │
        │        n_folds=5,                    │
        │      )                               │
        │                                       │
        │      log_to_mlflow()                 │
        │      counter.log_trial()  ← max 50   │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │       ML QUALITY SENTINEL             │
        │                                       │
        │  - Check Sharpe decay (train vs test)│
        │  - Verify no target leakage          │
        │  - Validate feature count < 30       │
        │  - Flag if stability_score > 1.0     │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼
              Best model selected
              (highest IC, stable, no leakage)
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │        QUANT DEVELOPER                │
        │                                       │
        │  signals = generate_ml_predicted_     │
        │    signals(                           │
        │      prices,                          │
        │      model_type="ridge",              │
        │      features=[...],                  │
        │      signal_method="rank",            │
        │      top_pct=0.1,                     │
        │    )                                  │
        │                                       │
        │  result = run_backtest(signals, ...)  │
        │                                       │
        └───────────────────────────────────────┘
                    │
                    ▼
            Full backtest metrics:
            Sharpe, drawdown, turnover, costs
```

---

## Fresh Start Timeline: Cold Start to Presented Strategies

### Day 0 (Setup)
```
6:00 PM - Data already loaded (prices, features for S&P 500)
        - Hypothesis registry empty
        - MLflow experiments empty
```

### Day 1 - Signal Discovery
```
Signal Scientist runs feature scans:

for feature in ALL_44_FEATURES:
    for forward_return in [5d, 10d, 20d]:
        ic = calculate_information_coefficient(feature, forward_return)
        if ic > 0.02:
            log_promising_signal(feature, ic)

Output: 8-12 promising signals identified
        (e.g., momentum_20d, rsi_14d, volatility_60d inversely)
```

### Day 2 - Hypothesis Formation
```
Alpha Researcher reviews signals, creates hypotheses:

HYP-2026-001: "Momentum + Low Volatility"
HYP-2026-002: "RSI Mean Reversion"
HYP-2026-003: "Multi-factor Composite"

Status: 'testing' → triggers ML Scientist
```

### Days 3-5 - ML Experimentation (The Heavy Lifting)

**This is where the bulk of computation happens:**

```python
# ML Scientist runs for each hypothesis
for hypothesis in hypotheses_in_testing:

    # Test multiple model types
    for model_type in ['ridge', 'lasso', 'random_forest', 'lightgbm']:

        # Test feature combinations
        for feature_set in generate_feature_combinations(hypothesis.signals):

            # Walk-forward validation (5 folds × 8 years)
            result = walk_forward_validate(
                config=WalkForwardConfig(
                    model_type=model_type,
                    features=feature_set,
                    start_date=date(2015, 1, 1),
                    end_date=date(2023, 12, 31),
                    n_folds=5,
                    window_type='expanding',
                    n_jobs=-1,  # Parallel
                ),
                symbols=sp500_universe,  # ~450 stocks
                log_to_mlflow=True,
            )

            # Track trial count (max 50 per hypothesis)
            counter.log_trial(model_type, params, result.mean_ic)
```

**Per hypothesis, ML Scientist tests:**
- 4 model types × 5-10 feature combos = 20-40 experiments
- Each experiment: 5-fold walk-forward = ~2-5 min with parallel processing
- Total per hypothesis: ~1-3 hours of compute

**ML Quality Sentinel runs after each batch:**
```python
# Check every completed experiment
for experiment in todays_experiments:

    # Sharpe decay check
    decay = SharpeDecayMonitor(max_decay_ratio=0.5)
    decay.check(train_sharpe, test_sharpe)

    # Leakage check
    leakage = TargetLeakageValidator(correlation_threshold=0.95)
    leakage.check(features_df, target)

    # Feature count check
    validator = FeatureCountValidator(max_threshold=50)
    validator.check(feature_count, sample_count)
```

### Days 6-7 - Backtesting & Validation

**Quant Developer** runs full backtests on best models:

```python
# For each hypothesis with passing ML validation
signals = generate_ml_predicted_signals(
    prices,
    model_type="ridge",  # Winner from ML experiments
    features=["momentum_20d", "volatility_60d"],
    signal_method="rank",
    top_pct=0.1,
    train_lookback=252,
    retrain_frequency=21,
)

result = run_backtest(signals, config, prices)
# Returns: Sharpe, max_drawdown, win_rate, turnover, etc.
```

**Validation Analyst** stress tests:

```python
# Parameter sensitivity
experiments = {
    "baseline": backtest_with_lookback(20),
    "short": backtest_with_lookback(10),
    "long": backtest_with_lookback(40),
}
robustness = check_parameter_sensitivity(experiments)

# Regime stress testing
for regime in ['2008_crisis', '2020_covid', '2022_rates']:
    run_backtest(signals, regime_period)
```

### Day 8 - Risk Review, CIO Agent Scoring & Report

**Risk Manager** validates:
```python
result = validate_strategy({
    "sharpe": 1.1,
    "num_trades": 500,
    "max_drawdown": 0.15,
    "win_rate": 0.54,
})
# Checks: Sharpe > 0.5, trades > 100, DD < 25%, etc.
```

**CIO Agent** scores across 4 dimensions:
```python
score = cio_agent.score_hypothesis(
    hypothesis_id="HYP-2026-001",
    experiment_data={"sharpe": 1.1, "stability_score": 0.7, ...},
    risk_data={"max_drawdown": 0.15, "volatility": 0.12, ...},
    economic_data={"thesis": "...", "uniqueness": "novel", ...},
    cost_data={"turnover": 0.25, "capacity": "high", ...},
)
# Decision: CONTINUE (0.78) → Approved for human review
```

**Report Generator** compiles findings for human CIO review.

---

## Timeline Summary

| Day | Activity | Output |
|-----|----------|--------|
| **1** | Signal Scientist scans features | 8-12 promising signals |
| **2** | Alpha Researcher creates hypotheses | 3-5 formal hypotheses |
| **3-5** | ML Scientist runs experiments | 100-200 ML experiments logged |
| **3-5** | ML Quality Sentinel audits | Leakage/overfitting flags |
| **6** | Quant Developer backtests winners | Full backtest metrics |
| **7** | Validation Analyst stress tests | Robustness reports |
| **8** | Risk Manager reviews | Approved/flagged strategies |
| **8** | **CIO Agent scores** | Deployment decisions (CONTINUE/CONDITIONAL/KILL) |
| **8** | Report Generator compiles | **Strategies presented to human CIO** |

**Result: ~8 calendar days from cold start to first ML strategies ready for human review.**

---

## Accelerating the Timeline

| Optimization | Impact |
|--------------|--------|
| Run Signal Scientist + Alpha Researcher same day | -1 day |
| Parallel ML experiments across hypotheses | -1 day |
| Pre-define feature combinations (skip search) | -1 day |
| Use only `ridge` + `lightgbm` (skip slower models) | Faster experiments |

**Aggressive timeline: 5 days** to first presented strategies.

---

## Ongoing Cadence (After Initial Run)

Once the system is warm:

```
Week N:
├── Mon: Signal Scientist finds 2 new signals
├── Tue: Alpha Researcher creates 1 hypothesis
├── Wed-Thu: ML Scientist tests (existing pipeline warm)
├── Fri: CIO Agent scores + CIO reviews 1-2 new strategies + ongoing refinements

Throughput: 1-3 new validated strategies per week
- ~40% automatically approved by CIO Agent (CONTINUE)
- ~30% sent back for refinement (CONDITIONAL)
- ~30% rejected (KILL/PIVOT)
```

---

## CIO Role: Where You Fit

### What Agents CANNOT Do (by design)
- Deploy any strategy to paper or live trading
- Override Risk Manager vetoes
- Modify deployed strategies
- Approve capital allocation

### CIO Agent: Autonomous Scoring & Decision Making

The **CIO Agent** is an autonomous agent that evaluates validated hypotheses across 4 dimensions and makes deployment decisions:

```python
from hrp.agents import CIOAgent

agent = CIOAgent(job_id="cio-weekly-001", actor="agent:cio")

# Score a hypothesis across all 4 dimensions
score = agent.score_hypothesis(
    hypothesis_id="HYP-2026-001",
    experiment_data={"sharpe": 1.5, "stability_score": 0.6, ...},
    risk_data={"max_drawdown": 0.12, "volatility": 0.11, ...},
    economic_data={"thesis": "...", "uniqueness": "novel", ...},
    cost_data={"turnover": 0.25, "capacity": "high", ...},
)

print(f"Decision: {score.decision}")  # CONTINUE, CONDITIONAL, KILL, PIVOT
print(f"Total Score: {score.total:.2f}")  # 0.75+ for CONTINUE
```

**Scoring Dimensions:**
| Dimension | Weight | Key Metrics |
|-----------|--------|-------------|
| **Statistical** | 35% | Sharpe ratio, stability, IC, fold consistency |
| **Risk** | 30% | Max drawdown, volatility, regime stability, Sharpe decay |
| **Economic** | 25% | Thesis quality, uniqueness, black box count, agent reports |
| **Cost** | 10% | Slippage survival, turnover, capacity, execution complexity |

**Decisions:**
- **CONTINUE** (Total ≥ 0.75): Approved for paper trading
- **CONDITIONAL** (0.60-0.74): Needs refinement, retest after changes
- **KILL** (< 0.60): Rejected, insufficient promise
- **PIVOT**: Different direction shows promise (e.g., signal works inversely)

### What the Human CIO Does

**1. Weekly Review (~15 min Friday)**
- Read Report Generator's summary
- Review CIO Agent decisions (override if needed)
- Final approve/reject for paper trading deployment
- Review any KILL/PIVOT decisions for strategic insights

**2. Strategic Direction**
- "Focus on low-turnover strategies this month"
- "Investigate why momentum stopped working in Q4"
- Agents pick up these directives

**3. Ad-hoc Queries via MCP**
```
CIO: "Why did the CIO Agent reject HYP-2026-042?"

→ CIO Agent provides detailed score breakdown
→ Risk Manager adds context
→ Response: "Statistical score was weak (0.4) due to Sharpe decay >50%"
```

**4. Final Authority (unchanged)**
```
Pipeline: Discovery → Modeling → Validation → Risk Review
                                                ↓
                                       CIO Agent Scoring
                                                ↓
                                         Human CIO APPROVAL
                                                ↓
                                          Paper Trading
```

### Typical Week

| Day | Human Time | What's Happening |
|-----|------------|------------------|
| Mon | 0 min | Agents processing weekend findings |
| Tue | 0 min | ML training running autonomously |
| Wed | 5 min | Glance at alerts if any leakage/risk flags |
| Thu | 0 min | Validation + Risk review in progress |
| Fri | 15 min | CIO Agent scores hypotheses + review report, final approvals |
| Sat | 0 min | Fundamentals ingestion (weekly) |
| Sun | 0 min | System idle |

**Total weekly commitment: ~20 minutes** for a full quant research operation (CIO Agent reduces from 35 min).

---

## Information Flow Diagram

```
                    ┌─────────────────┐
                    │   CIO (User)    │
                    │  Weekly review  │
                    │  Final approval │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
    ┌─────────────────┐            ┌─────────────────┐
    │ Report Generator│◄───────────│  Risk Manager   │
    │ (synthesizes)   │            │  (independent)  │
    └────────┬────────┘            └────────┬────────┘
             │                              │
    ┌────────┴────────────────────┬─────────┴────┐
    ▼                             ▼              ▼
┌──────────┐  ┌──────────────┐  ┌────────────────┐
│   ML     │  │  Validation  │  │    Quant       │
│Scientist │  │   Analyst    │  │   Developer    │
└────┬─────┘  └──────────────┘  └────────────────┘
     │                 ▲
     ▼                 │
┌──────────┐           │
│ML Quality│           │
│ Sentinel │           │
└──────────┘           │
                       │
    ┌──────────────────┴────────────────┐
    ▼                                   ▼
┌────────────────┐            ┌─────────────────┐
│Alpha Researcher│◄───────────│ Signal Scientist│
│(strategy)      │            │ (features)      │
└────────────────┘            └─────────────────┘
        ▲                             ▲
        └─────────────────────────────┘
              Shared: Hypothesis Registry
                      MLflow Experiments
                      Lineage System
                      Feature Store

                    ┌─────────────────┐
                    │   CIO Agent     │
                    │ (Auto Scoring)  │
                    │ - Statistical   │
                    │ - Risk          │
                    │ - Economic      │
                    │ - Cost          │
                    └────────┬────────┘
                             │
                             ▼
                    Deployment Decisions
                    (CONTINUE/CONDITIONAL/
                     KILL/PIVOT)
```

---

## Next Steps

### Implementation Status

| Agent | Status | Notes |
|-------|--------|-------|
| Signal Scientist | ✅ Implemented | Nightly signal scans, hypothesis creation |
| Alpha Researcher | ✅ Implemented (Enhanced) | Strategy generation, hypothesis refinement, regime analysis |
| ML Scientist | ✅ Implemented | Walk-forward validation, ML experiments |
| ML Quality Sentinel | ✅ Implemented | Overfitting detection, leakage checks |
| Quant Developer | ✅ Implemented | Strategy backtesting with realistic costs |
| **Pipeline Orchestrator** | ✅ Implemented (NEW) | Baseline execution, parallel experiments, kill gates |
| Validation Analyst | ✅ Implemented | Parameter sensitivity, stress testing |
| Risk Manager | ✅ Implemented | Independent portfolio-level risk oversight |
| Report Generator | ✅ Implemented | Daily/weekly research reports |
| **CIO Agent** | ✅ Implemented | Autonomous scoring across 4 dimensions |

### Remaining Work

1. [ ] Complete Quant Developer agent (automate backtest generation)
2. [ ] Implement event-driven agent coordination (lineage triggers)
3. [ ] Add deployment pipeline (paper trading integration)
4. [ ] Human-in-the-loop approval workflow for CIO decisions

---

## Document History

- **2026-01-25:** Initial brainstorm documentation from conversation
- **2026-01-28:** Updated to include CIO Agent (9th agent), autonomous scoring, and updated implementation status
- **2026-01-28:** Updated to reflect Risk Manager implementation (✅ Implemented - independent portfolio-level risk oversight)
- **2026-01-29:** Major update - Added Pipeline Orchestrator (10th agent), enhanced Alpha Researcher with strategy generation, updated decision pipeline

My recommendation for first 2-3 agents:                                                                                                                                           
  ┌──────────┬─────────────────────┬───────────────────────────────────────────────────────────────────┐                                                                            
  │ Priority │        Agent        │                             Rationale                             │                                                                            
  ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤                                                                            
  │ 1st      │ Signal Scientist    │ Starts the pipeline - discovers signals that feed everything else │                                                                            
  ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤                                                                            
  │ 2nd      │ ML Scientist        │ Core value - runs the experiments that produce strategies         │                                                                            
  ├──────────┼─────────────────────┼───────────────────────────────────────────────────────────────────┤                                                                            
  │ 3rd      │ ML Quality Sentinel │ Safety net - prevents garbage from propagating                    │                                                                            
  └──────────┴─────────────────────┴───────────────────────────────────────────────────────────────────┘                                                                            
  Why this order:                                                                                                                                                                   
  1. Signal Scientist can work standalone (scans features, logs findings)                                                                                                           
  2. ML Scientist consumes Signal Scientist output and produces the actual strategies                                                                                               
  3. ML Quality Sentinel catches problems early before wasting compute downstream                                                                                                   
                                                                                                                                                                                    
  What we defer:                                                                                                                                                                    
  - Alpha Researcher: You can manually refine hypotheses initially                                                                                                                  
  - Quant Developer: Backtesting can be triggered manually                                                                                                                          
  - Validation/Risk/Report: Later stages, less urgent                                                                                                                               
                                                                                                                                                                                    
  Which agents would you like to start with?                                                                                                                                        
                                                                                                                                                                                    
✻ Sautéed for 1m 41s                                                                                                                                                                
                                                                                                                                                                                    
❯ let's start with signal scientist. give me a full description of the agent for review before we build it        