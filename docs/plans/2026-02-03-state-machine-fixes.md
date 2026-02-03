# HRP State Machine & Pipeline Consistency Fixes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix schema/code mismatches where agents write invalid statuses and deployment status values don't match schema constraints.

**Architecture:** Agents use the 6-status model (draft, testing, validated, rejected, deployed, deleted) but some write non-existent statuses. Fix by mapping to valid statuses + metadata/lineage events for tracking.

**Tech Stack:** Python 3.11, DuckDB, pytest

---

## What's Already Done (Previous Session)

| Task | Status | Files Changed |
|------|--------|---------------|
| Updated decision-pipeline.md to match 6-status model | ✅ Done | `docs/agents/decision-pipeline.md` |
| Added `signal_scan_complete`, `ml_scientist_validation` to EventType | ✅ Done | `hrp/research/lineage.py` |
| Updated schema CHECK constraint for lineage events | ✅ Done | `hrp/data/schema.py` |
| Added state machine diagram to CLAUDE.md | ✅ Done | `CLAUDE.md` |
| Added `pipeline_stage` column to hypotheses | ✅ Done | `hrp/data/schema.py`, `hrp/research/hypothesis.py` |

---

## Task 1: Fix Quant Developer Invalid Status

**Files:**
- Modify: `hrp/agents/quant_developer.py:1021-1026`
- Test: `tests/test_agents/test_quant_developer.py`

**Step 1: Find and examine the test file**

Run: `ls tests/test_agents/test_quant_developer.py`

**Step 2: Write/update test for valid status**

```python
def test_update_hypothesis_uses_valid_status(self, mock_api):
    """Quant Developer should use 'validated' status, not 'backtested'."""
    # ... setup ...
    # Assert api.update_hypothesis was called with status="validated"
    mock_api.update_hypothesis.assert_called_with(
        hypothesis_id=ANY,
        status="validated",  # NOT "backtested"
        metadata=ANY,
        actor=ANY,
    )
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_agents/test_quant_developer.py -k "valid_status" -v`
Expected: FAIL (currently uses "backtested")

**Step 4: Fix the code**

Change line 1023 in `hrp/agents/quant_developer.py`:
```python
# Before:
status="backtested",

# After:
status="validated",
```

Also add `pipeline_stage` update after the status update:
```python
self.api.update_hypothesis(
    hypothesis_id=hypothesis_id,
    status="validated",
    metadata=metadata,
    actor=self.ACTOR,
)
# Track pipeline progress
from hrp.research.hypothesis import update_pipeline_stage
update_pipeline_stage(hypothesis_id, "quant_backtest", actor=self.ACTOR)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_agents/test_quant_developer.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add hrp/agents/quant_developer.py tests/test_agents/test_quant_developer.py
git commit -m "fix(quant_developer): use valid 'validated' status instead of 'backtested'"
```

---

## Task 2: Fix Validation Analyst Invalid Status

**Files:**
- Modify: `hrp/agents/validation_analyst.py:181-185`
- Test: `tests/test_agents/test_validation_analyst.py`

**Step 1: Run existing tests**

Run: `pytest tests/test_agents/test_validation_analyst.py -v`

**Step 2: Fix the code**

Change line 183 in `hrp/agents/validation_analyst.py`:
```python
# Before:
"validation_failed",

# After:
"testing",  # Demote back to testing for rework
```

**Step 3: Run tests**

Run: `pytest tests/test_agents/test_validation_analyst.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add hrp/agents/validation_analyst.py
git commit -m "fix(validation_analyst): use valid 'testing' status instead of 'validation_failed'"
```

---

## Task 3: Fix Risk Manager Invalid Status

**Files:**
- Modify: `hrp/agents/risk_manager.py:600-602`
- Test: `tests/test_agents/test_risk_manager.py`

**Step 1: Run existing tests**

Run: `pytest tests/test_agents/test_risk_manager.py -v`

**Step 2: Fix the code**

Change line 602 in `hrp/agents/risk_manager.py`:
```python
# Before:
status="validated" if assessment.passed else "risk_vetoed",

# After:
status="validated",  # Always keep validated; veto tracked via lineage event
```

The `risk_veto` lineage event already tracks vetoes (line ~620). The status stays `validated` but the hypothesis is blocked from deployment by the veto event.

**Step 3: Run tests**

Run: `pytest tests/test_agents/test_risk_manager.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add hrp/agents/risk_manager.py
git commit -m "fix(risk_manager): keep 'validated' status, track veto via lineage event"
```

---

## Task 4: Fix Deployment Status Mismatch

**Files:**
- Modify: `hrp/ml/deployment.py` (lines 232, 326, 416 for "success"; 196, 263, 364, 442 for "failed")
- Test: `tests/test_ml/test_deployment.py`

**Step 1: Run existing tests**

Run: `pytest tests/test_ml/test_deployment.py -v`

**Step 2: Fix the code - replace all occurrences**

```python
# Before:
status="success"

# After:
status="active"
```

```python
# Before:
status="failed"

# After:
status="rolled_back"
```

**Step 3: Run tests**

Run: `pytest tests/test_ml/test_deployment.py -v`
Expected: PASS (or update test assertions if they check for "success"/"failed")

**Step 4: Commit**

```bash
git add hrp/ml/deployment.py
git commit -m "fix(deployment): use schema-valid statuses 'active'/'rolled_back' instead of 'success'/'failed'"
```

---

## Task 5: Fix Dashboard Ingestion Status Query

**Files:**
- Modify: `hrp/dashboard/pages/ingestion_status.py` (lines 83, 95, 104, 151)

**Step 1: Fix the queries**

Replace all occurrences:
```python
# Before:
status = 'success'

# After:
status = 'completed'
```

**Step 2: Verify manually**

Run: `streamlit run hrp/dashboard/app.py` and check ingestion status page shows data.

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/ingestion_status.py
git commit -m "fix(dashboard): query 'completed' status to match ingestion_log schema"
```

---

## Task 6: Fix Kill Gate Enforcer Idempotency

**Files:**
- Modify: `hrp/agents/kill_gate_enforcer.py`
- Test: `tests/test_agents/test_kill_gate_enforcer.py`

**Step 1: Find where metadata should be set**

Look for where hypothesis is updated after kill gate processing.

**Step 2: Add metadata stamp after processing**

```python
# After processing each hypothesis, add:
self.api.update_hypothesis(
    hypothesis_id=hypothesis_id,
    metadata={
        "kill_gate_enforcer": {
            "run_id": self.run_id,
            "run_date": datetime.now().isoformat(),
            "result": "passed" if passed else "killed",
        }
    },
    actor=self.ACTOR,
)
```

**Step 3: Run tests**

Run: `pytest tests/test_agents/test_kill_gate_enforcer.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add hrp/agents/kill_gate_enforcer.py
git commit -m "fix(kill_gate): add metadata stamp to prevent duplicate runs"
```

---

## Task 7: Clean Up Doc References

**Files:**
- Modify: `README.md` - remove broken link to `docs/agents/agent-interaction-diagram.md`
- Modify: `docs/agents/decision-pipeline.md` - remove Code Materializer references
- Modify: `docs/agents/09-cio-agent.md` - remove missing table references

**Step 1: Fix README.md**

Remove or comment out the link to the non-existent diagram.

**Step 2: Fix decision-pipeline.md**

Search for "Code Materializer" and remove or note as "not yet implemented".

**Step 3: Fix 09-cio-agent.md**

Remove references to `paper_portfolio_history` and `cio_threshold_history` tables.

**Step 4: Commit**

```bash
git add README.md docs/agents/decision-pipeline.md docs/agents/09-cio-agent.md
git commit -m "docs: remove references to non-existent artifacts"
```

---

## Verification

After all tasks complete:

```bash
# 1. Verify no invalid statuses remain
grep -rn "backtested\|validation_failed\|risk_vetoed" hrp/agents/
# Expected: No matches (except comments)

# 2. Verify deployment statuses fixed
grep -rn 'status="success"\|status="failed"' hrp/ml/deployment.py
# Expected: No matches

# 3. Run full test suite
pytest tests/test_agents/ tests/test_research/ tests/test_ml/ -v
# Expected: All pass

# 4. Verify enum/schema sync
python -c "
from hrp.research.lineage import EventType
from hrp.data.schema import TABLES
import re
schema_events = set(re.findall(r\"'([a-z_]+)'\", TABLES['lineage']))
enum_events = {e.value for e in EventType}
print('In schema but not enum:', schema_events - enum_events)
print('In enum but not schema:', enum_events - schema_events)
"
```
