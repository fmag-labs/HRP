# HRP Scheduler Status and Purpose

## What is the Scheduler?

The `IngestionScheduler` is a **long-running background service** that coordinates:
1. **Daily data ingestion** (keeps the database current)
2. **Weekly research jobs** (automated hypothesis discovery)
3. **Event-driven agent coordination** (pipeline triggers)
4. **Report generation** (daily/summaries, weekly/reviews)

It runs **24/7** as a background daemon via launchd.

---

## Current Status: ✅ RUNNING

**Process ID:** 61098
**Started:** 2026-01-30 1:02 PM ET
**Command:** `python -m hrp.agents.run_scheduler --with-research-triggers --with-signal-scan --with-quality-sentinel --with-daily-report --with-weekly-report`

---

## Scheduled Jobs (10 jobs)

| Job | Schedule | Purpose |
|-----|----------|---------|
| **price_ingestion** | Daily 6:00 PM ET | Fetch market data from yfinance |
| **universe_update** | Daily 6:05 PM ET | Update S&P 500 universe with exclusions |
| **feature_computation** | Daily 6:10 PM ET | Compute 44 technical features |
| **fundamentals_ingestion** | Saturday 10:00 AM ET | Ingest fundamental data (P/E, P/B, etc.) |
| **signal_scientist_weekly** | Monday 7:00 PM ET | Scan 44 features for predictive signals |
| **model_monitoring** | Daily 6:00 AM ET | Monitor deployed models + audit safety net |
| **daily_backup** | Daily 2:00 AM ET | Backup database |
| **daily_report** | Daily 7:00 AM ET | Generate daily research summary |
| **weekly_report** | Sunday 8:00 PM ET | Generate weekly research review |
| **lineage_event_watcher** | Every 60 seconds | Poll lineage events & trigger agents |

---

## Event-Driven Pipeline Triggers (6 triggers)

The `lineage_event_watcher` polls the database every 60 seconds for new events and triggers downstream agents:

| Trigger | Event | Triggers Agent |
|--------|-------|---------------|
| `signal_scientist_to_alpha_researcher` | `hypothesis_created` | Alpha Researcher |
| `alpha_researcher_to_ml_scientist` | `alpha_researcher_complete` | ML Scientist |
| `ml_scientist_to_quality_sentinel` | `experiment_completed` | ML Quality Sentinel |
| `ml_quality_sentinel_to_validation_analyst` | `ml_quality_sentinel_audit` | Validation Analyst* |

*Note: There are also triggers for Quant Developer and Pipeline Orchestrator when enabled.

---

## Why Jobs Don't Show in My Query

When you query `IngestionScheduler()` from a new Python process, you're creating a **new instance** separate from the running scheduler. The actual jobs are in the **background process** (PID 52024).

To see the actual running jobs, check the logs or use the platform API to query recent activity.

---

## Daily Scheduler Workflow

### 6:00 PM - Price Ingestion
```
PriceIngestionJob runs
→ Fetches OHLCV data for universe symbols
→ Inserts into prices table
→ Event: price_ingestion_complete
```

### 6:05 PM - Universe Update
```
UniverseUpdateJob runs
→ Fetches S&P 500 from Wikipedia
→ Applies exclusions (financials, REITs, penny stocks)
→ Updates universe table
→ Event: universe_update_complete
```

### 6:10 PM - Feature Computation
```
FeatureComputationJob runs
→ Computes 44 technical features for all symbols
→ Momentum, volatility, volume, oscillators, etc.
→ Updates features table
→ Event: feature_computation_complete
```

---

## Weekly Research Workflow

### Monday 7:00 PM - Signal Scan
```
SignalScientist runs
→ Scans 44 features for predictive IC
→ Creates hypotheses for signals with IC > 0.03
→ Event: hypothesis_created
→ Triggers: Alpha Researcher
```

### Automatic: Alpha Researcher (within ~1 min)
```
AlphaResearcher runs
→ Reviews draft hypotheses via Claude API
→ Promotes promising to "testing" status
→ Generates strategy specs to docs/strategies/
→ Event: alpha_researcher_complete
→ Triggers: ML Scientist
```

### Automatic: ML Scientist (within ~5-10 min)
```
MLScientist runs
→ Walk-forward validation on testing hypotheses
→ Tests multiple models (ridge, lasso, RF, etc.)
→ Calculates Sharpe, IC, stability metrics
→ Promotes to "validated" if passes
→ Event: experiment_completed
→ Triggers: ML Quality Sentinel
```

### Automatic: ML Quality Sentinel (within ~1 minute)
```
MLQualitySentinel runs (event-driven)
→ Audits specific experiment for overfitting
→ Checks: Sharpe decay, target leakage, feature count
→ Passes or flags issues
→ Event: ml_quality_sentinel_audit
→ Triggers: Validation Analyst (if passed)
```

### Daily: Model Monitoring (6 AM ET)
```
MLQualitySentinel runs (scheduled)
→ Monitors deployed models for IC degradation
→ Acts as safety net for missed experiments (7-day window)
→ Sends alerts for critical issues
→ Note: New experiments already audited via event-driven triggers
```

---

## How to Verify It's Working

### 1. Check Process
```bash
ps aux | grep "hrp.agents.run_scheduler" | grep -v grep
```

### 2. Check Logs
```bash
# Real-time log watching
tail -f ~/hrp-data/logs/scheduler.error.log

# Recent job activity
tail -50 ~/hrp-data/logs/scheduler.error.log | grep "Added job\|Scheduled\|trigger"
```

### 3. Check Database Activity
```python
from hrp.data.db import get_db
from datetime import date, timedelta

db = get_db()

# Recent ingestion activity
ingestion = db.fetchall("""
    SELECT event_type, actor, timestamp
    FROM lineage
    WHERE actor LIKE 'system:%'
    ORDER BY timestamp DESC
    LIMIT 10
""")

print("Recent Data Ingestion:")
for event in ingestion:
    print(f"  {event[2]}: {event[0]} - {event[1]}")

# Recent agent activity
agents = db.fetchall("""
    SELECT event_type, actor, timestamp, hypothesis_id
    FROM lineage
    WHERE actor LIKE 'agent:%'
    ORDER BY timestamp DESC
    LIMIT 10
""")

print("\nRecent Agent Activity:")
for event in agents:
    print(f"  {event[2]}: {event[1]}")
    if event[3]:
        print(f"    Hypothesis: {event[3]}")
```

### 4. Check Agent Jobs via Platform API
```python
from hrp.api.platform import PlatformAPI

api = PlatformAPI()

# Check hypothesis status pipeline
draft = api.list_hypotheses(status="draft")
testing = api.list_hypotheses(status="testing")
validated = api.list_hypotheses(status="validated")

print(f"Draft: {len(draft)}")
print(f"Testing: {len(testing)}")
print(f"Validated: {len(validated)}")
```

---

## Stopping the Scheduler

### Temporary Stop
```bash
# Stop the background process
kill 61098

# Restart later with:
python -m hrp.agents.run_scheduler --with-research-triggers --with-signal-scan --with-quality-sentinel --with-daily-report --with-weekly-report &
```

### Permanent Stop (via launchd)
```bash
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist

# To re-enable:
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist
```

---

## Managing the Scheduler

### View Scheduled Jobs
```bash
# Via Platform API - see what's scheduled
python -c "
from hrp.agents.scheduler import IngestionScheduler
s = IngestionScheduler()
for job in s.scheduler.get_jobs():
    print(f'{job.id}: {job.name} (next: {job.next_run_time})'
"
```

### Run Jobs Manually
```bash
# Run specific job now
python -m hrp.agents.cli run-job price_ingestion

# Run all ingestion jobs
python -m hrp.agents.cli run-ingestion
```

### Run Agents Manually
```python
# Signal scan
from hrp.agents import SignalScientist
scientist = SignalScientist()
result = scientist.run()

# Alpha Researcher
from hrp.agents import AlphaResearcher
researcher = AlphaResearcher()
result = researcher.run()

# ML Scientist
from hrp.agents import MLScientist
scientist = MLScientist()
result = scientist.run()
```

---

## Troubleshooting

### Scheduler Not Running

**Problem:** Process not found in `ps aux`

**Solution:**
```bash
# Restart via launchd
launchctl unload ~/Library/LaunchAgents/com.hrp.scheduler.plist
launchctl load ~/Library/LaunchAgents/com.hrp.scheduler.plist

# Or start manually
python -m hrp.agents.run_scheduler --with-research-triggers &
```

### Jobs Not Executing

**Problem:** Jobs scheduled but not running

**Check:**
```bash
# Check scheduler status
launchctl list | grep hrp

# Check for errors
tail -100 ~/hrp-data/logs/scheduler.error.log

# Verify database connectivity
python -c "from hrp.data.db import get_db; print(get_db())"
```

### Agents Not Triggering

**Problem:** Events created but downstream agents don't run

**Check:**
```bash
# Check lineage event watcher is polling
tail -f ~/hrp-data/logs/scheduler.error.log | grep -E "poll|trigger"

# Check for lineage events
python -c "
from hrp.data.db import get_db
from datetime import timedelta
db = get_db()
events = db.fetchall(\"SELECT COUNT(*) FROM lineage WHERE timestamp >= ?\",
                 ((date.today() - timedelta(hours=1)).isoformat(),))
print(f'Events in last hour: {events[0]}')
"
```

---

## Summary

| Aspect | Status |
|--------|--------|
| **Scheduler Process** | ✅ Running (PID 61098) |
| **Daily Data Jobs** | ✅ Scheduled (6:00 PM, 6:05 PM, 6:10 PM ET) |
| **Research Jobs** | ✅ Scheduled (signal scan Mon 7 PM, model monitoring daily 6 AM) |
| **Event Pipeline** | ✅ Active (6 triggers, polling every 60s) |
| **Reports** | ✅ Scheduled (daily 7 AM, weekly Sun 8 PM) |

The scheduler is the **central coordination system** for the entire HRP platform - it keeps data fresh, drives research automation, and coordinates the agent pipeline.

**Key Design Note**: ML Quality Sentinel serves dual purposes:
1. **Event-driven** (immediate): Audits new experiments within 60 seconds of completion
2. **Scheduled** (daily 6 AM): Monitors deployed models for degradation and acts as a safety net
