# Event-Driven Pipeline Setup Guide

## Overview

The HRP event-driven pipeline is now **active** and running. Agents will automatically trigger downstream workflows based on lineage events in the database.

## What Was Just Set Up

### Scheduled Jobs (7 Jobs)

| Job | Schedule | Purpose |
|-----|----------|---------|
| `price_ingestion` | Daily 6:00 PM ET | Fetch daily price data |
| `universe_update` | Daily 6:05 PM ET | Update S&P 500 universe |
| `feature_computation` | Daily 6:10 PM ET | Compute 44 technical features |
| `signal_scientist_weekly` | Monday 7:00 PM ET | Discover predictive signals |
| `model_monitoring` | Daily 6:00 AM ET | Monitor deployed models + audit safety net |
| `daily_report` | Daily 7:00 AM ET | Generate daily research summary |
| `weekly_report` | Sunday 8:00 PM ET | Generate weekly research summary |
| `lineage_event_watcher` | Every 60 seconds | Poll for lineage events and trigger agents |

### Event-Driven Pipeline Triggers (6 Triggers)

| Trigger Name | Event | Triggers | Callback |
|---------------|-------|----------|----------|
| `signal_scientist_to_alpha_researcher` | `hypothesis_created` | Signal Scientist creates hypothesis | Alpha Researcher reviews draft hypotheses |
| `alpha_researcher_to_ml_scientist` | `alpha_researcher_complete` | Alpha Researcher promotes to testing | ML Scientist runs walk-forward validation |
| `ml_scientist_to_quality_sentinel` | `experiment_completed` | ML Scientist finishes validation | ML Quality Sentinel audits for overfitting |
| `ml_quality_sentinel_to_quant_developer` | `ml_quality_sentinel_audit` | Quality Sentinel passes audit | Quant Developer implements strategy |
| `quant_developer_to_pipeline_orchestrator` | `quant_developer_backtest_complete` | Backtest completed | Pipeline Orchestrator runs parameter sweeps |
| `pipeline_orchestrator_to_validation_analyst` | `pipeline_orchestrator_complete` | Orchestrator finishes | Validation Analyst stress-tests before deployment |

## How the Pipeline Works

### 1. Signal Discovery Phase (Weekly - Monday 7 PM)

**Trigger:** Scheduled job

**Agent:** Signal Scientist

**Action:**
- Scans 44 features for predictive signals
- Creates hypotheses for promising signals (IC > 0.03)
- Writes to `draft` status

**Output:** Lineage event `hypothesis_created` with actor `agent:signal-scientist`

---

### 2. Hypothesis Review Phase (Automatic - within ~1 minute)

**Trigger:** `hypothesis_created` event from Signal Scientist

**Agent:** Alpha Researcher

**Action:**
- Reviews draft hypotheses using Claude API
- Provides economic rationale
- Refines thesis/falsification criteria
- Promotes promising hypotheses to `testing` status
- Generates strategy specifications to `docs/strategies/`

**Output:** Lineage event `alpha_researcher_complete` with actor `agent:alpha-researcher`

---

### 3. Model Validation Phase (Automatic - within ~1 minute)

**Trigger:** `alpha_researcher_complete` event

**Agent:** ML Scientist

**Action:**
- Runs walk-forward validation on `testing` hypotheses
- Tests with multiple models (ridge, lasso, random forest, etc.)
- Calculates Sharpe, IC, stability metrics
- Promotes to `validated` if passes criteria

**Output:** Lineage event `experiment_completed` with actor `agent:ml-scientist`

---

### 4. Quality Audit Phase (Automatic - within ~1 minute)

**Trigger:** `experiment_completed` event

**Agent:** ML Quality Sentinel

**Action:**
- Audits experiments for overfitting signals
- Checks: Sharpe decay, target leakage, feature count, stability
- Passes or flags issues

**Output:** Lineage event `ml_quality_sentinel_audit` with actor `agent:ml-quality-sentinel`

---

### 5. Strategy Implementation Phase (Automatic - within ~1 minute)

**Trigger:** `ml_quality_sentinel_audit` event (passed)

**Agent:** Quant Developer

**Action:**
- Implements validated strategies
- Runs backtests with realistic costs
- Tests parameter variations
- Generates backtest results

**Output:** Lineage event `quant_developer_backtest_complete` with actor `agent:quant-developer`

---

### 6. Parallel Experiments Phase (Automatic - within ~5-10 minutes)

**Trigger:** `quant_developer_backtest_complete` event

**Agent:** Pipeline Orchestrator

**Action:**
- Runs parallel parameter sweeps
- Tests regime scenarios
- Applies kill gates (early termination for poor performers)

**Output:** Lineage event `pipeline_orchestrator_complete` with actor `agent:pipeline-orchestrator`

---

### 7. Pre-Deployment Testing (Automatic - within ~2 minutes)

**Trigger:** `pipeline_orchestrator_complete` event

**Agent:** Validation Analyst

**Action:**
- Stress-tests validated strategies
- Parameter sensitivity analysis
- Time/regime stability checks
- Execution cost analysis

**Output:** Lineage event `validation_analyst_complete` with actor `agent:validation-analyst`

---

### 8. Deployment Decision (Manual - Weekly review)

**Agent:** CIO Agent (via scheduled weekly review)

**Action:**
- Reviews all validated hypotheses
- Scores across 4 dimensions (Statistical, Risk, Economic, Cost)
- Makes final decision: CONTINUE / CONDITIONAL / KILL / PIVOT
- Records decision to database

## Verifying the Pipeline is Working

### Check Scheduler Status

```bash
# Check if scheduler is running
launchctl list | grep hrp

# View scheduler logs
tail -f ~/hrp-data/logs/scheduler.error.log
```

### Monitor Agent Activity

```python
from hrp.data.db import get_db
from datetime import date, timedelta

db = get_db()

# Get recent agent activity
events = db.fetchall("""
    SELECT lineage_id, event_type, actor, timestamp, hypothesis_id
    FROM lineage
    WHERE actor LIKE 'agent:%'
    ORDER BY timestamp DESC
    LIMIT 20
""")

for event in events:
    print(f"{event[3]}: {event[2]} - {event[1]}")
    if event[4]:
        print(f"  Hypothesis: {event[4]}")
```

### View Trigger Status

```python
from hrp.agents.scheduler import IngestionScheduler

scheduler = IngestionScheduler()

# Check registered triggers
print(f"Active triggers: {len(scheduler._triggers)}")

for trigger in scheduler._triggers:
    print(f"{trigger.name}: {trigger.event_type}")
```

## What Happens Next

### Immediate Actions

1. **ML Scientist will run** on `HYP-2026-002` (the hypothesis in "testing" status)
2. **Agents will trigger** automatically when events occur
3. **Reports will be generated** daily at 7 AM and weekly on Sunday

### Hypothesis Flow to Watch

```
Currently in Testing (needs ML Scientist):
└── HYP-2026-002: Test (created 2026-01-22)

Validated (but Risk Manager vetoed - need re-evaluation):
├── HYP-2026-005: volatility_60d predicts monthly returns
├── HYP-2026-004: Test: Momentum predicts returns
└── HYP-2026-001: Short-Term Volatility Risk Premium

New Drafts (from last night):
├── HYP-2026-016: volume_ratio predicts monthly returns
├── HYP-2026-015: volume_ratio predicts bi-weekly returns
├── HYP-2026-014: Sentiment Reversal Oscillator
├── HYP-2026-013: Trend Aversion Quality
└── HYP-2026-012: Order Flow Imbalance Momentum
```

## Troubleshooting

### If Agents Don't Trigger

**Check 1: Is the Event Watcher running?**
```bash
# Check for lineage_event_watcher job
launchctl list | grep hrp
```

**Check 2: Are lineage events being created?**
```python
from hrp.data.db import get_db
db = get_db()
events = db.fetchall("SELECT COUNT(*) FROM lineage WHERE timestamp >= NOW() - INTERVAL '1 hour'")
print(f"Events in last hour: {events[0]}")
```

**Check 3: Are triggers registered?**
```python
from hrp.agents.scheduler import IngestionScheduler
scheduler = IngestionScheduler()
print(f"Triggers: {len(scheduler._triggers)}")
```

### Restart the Pipeline

```bash
# Stop scheduler
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Restart
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Or start manually with full pipeline
python -m hrp.agents.run_scheduler --with-research-triggers
```

## Running the Pipeline Manually

If you need to trigger agents manually:

```python
# 1. Signal Scientist (if you want new discoveries)
from hrp.agents import SignalScientist
scientist = SignalScientist()
result = scientist.run()

# 2. Alpha Researcher (review draft hypotheses)
from hrp.agents import AlphaResearcher
researcher = AlphaResearcher()
result = researcher.run()

# 3. ML Scientist (validate testing hypotheses)
from hrp.agents import MLScientist
scientist = MLScientist()
result = scientist.run()

# 4. ML Quality Sentinel (audit experiments)
from hrp.agents import MLQualitySentinel
sentinel = MLQualitySentinel()
result = sentinel.run()

# Note: ML Quality Sentinel has dual purposes:
# - Event-driven: Audits new experiments immediately (within 60s)
# - Scheduled (6 AM): Monitors deployed models for degradation

# 5. CIO Agent (make deployment decisions)
from hrp.agents import CIOAgent
agent = CIOAgent()
agent.run_weekly_review()
```

## Next Steps

The pipeline is now active. Watch for:
1. **ML Scientist** to validate `HYP-2026-002`
2. **New reports** in `docs/reports/YYYY-MM-DD/`
3. **Lineage events** in the database
4. **Strategy specs** in `docs/strategies/`

### Daily Report Tomorrow

At 7:00 AM ET, the daily report will show:
- Agent activity summary
- New hypotheses created
- Hypotheses promoted
- Experiments completed
- Actionable insights

### Weekly Report on Sunday

At 8:00 PM ET, the weekly report will summarize:
- Full week of research activity
- Top performing signals
- Deployment recommendations
- Risk assessment

---

**Status:** ✅ Event-driven pipeline is **ACTIVE** and running
