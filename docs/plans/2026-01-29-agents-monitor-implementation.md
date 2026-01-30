# Agents Monitor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit dashboard for real-time monitoring and historical audit of all 11 HRP research agents

**Architecture:** Single-page Streamlit dashboard with two sections (Real-Time top, Historical bottom), status inferred from lineage table events, hybrid auto-refresh (2s active / 10s idle)

**Tech Stack:** Python 3.11+, Streamlit, DuckDB (lineage queries), pandas, loguru

---

## Phase 1: Core MVP (Basic Display)

### Task 1: Add AGENT_RUN_START Event Type

**Files:**
- Modify: `hrp/research/lineage.py:21-46`
- Test: `tests/research/test_lineage.py`

**Step 1: Write the failing test**

Add to `tests/research/test_lineage.py`:

```python
def test_agent_run_start_event_type_exists():
    """AGENT_RUN_START event type should exist in EventType enum."""
    from hrp.research.lineage import EventType

    assert hasattr(EventType, "AGENT_RUN_START")
    assert EventType.AGENT_RUN_START == "agent_run_start"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/research/test_lineage.py::test_agent_run_start_event_type_exists -v
```

Expected: FAIL with "AttributeError: 'EventType' object has no attribute 'AGENT_RUN_START'"

**Step 3: Add event type to enum**

Modify `hrp/research/lineage.py` at line 46 (after existing events):

```python
class EventType(str, Enum):
    """Supported lineage event types."""

    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    HYPOTHESIS_DELETED = "hypothesis_deleted"
    HYPOTHESIS_FLAGGED = "hypothesis_flagged"
    EXPERIMENT_RUN = "experiment_run"
    EXPERIMENT_LINKED = "experiment_linked"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"
    DEPLOYMENT_APPROVED = "deployment_approved"
    DEPLOYMENT_REJECTED = "deployment_rejected"
    AGENT_RUN_COMPLETE = "agent_run_complete"
    AGENT_RUN_START = "agent_run_start"  # NEW
    ML_QUALITY_SENTINEL_AUDIT = "ml_quality_sentinel_audit"
    ALPHA_RESEARCHER_REVIEW = "alpha_researcher_review"
    VALIDATION_ANALYST_REVIEW = "validation_analyst_review"
    RISK_REVIEW_COMPLETE = "risk_review_complete"
    RISK_VETO = "risk_veto"
    QUANT_DEVELOPER_BACKTEST_COMPLETE = "quant_developer_backtest_complete"
    ALPHA_RESEARCHER_COMPLETE = "alpha_researcher_complete"
    PIPELINE_ORCHESTRATOR_COMPLETE = "pipeline_orchestrator_complete"
    KILL_GATE_TRIGGERED = "kill_gate_triggered"
    DATA_INGESTION = "data_ingestion"
    SYSTEM_ERROR = "system_error"
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/research/test_lineage.py::test_agent_run_start_event_type_exists -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/research/lineage.py tests/research/test_lineage.py
git commit -m "feat(lineage): add AGENT_RUN_START event type

Supports real-time agent status inference by marking when agents begin execution.
"
```

---

### Task 2: Create Backend Module Skeleton

**Files:**
- Create: `hrp/dashboard/agents_monitor.py`
- Test: `tests/dashboard/test_agents_monitor.py`

**Step 1: Write the failing test**

Create `tests/dashboard/test_agents_monitor.py`:

```python
import pytest
from hrp.dashboard.agents_monitor import AgentStatus, get_all_agent_status

def test_agent_status_dataclass():
    """AgentStatus dataclass should exist with required fields."""
    status = AgentStatus(
        agent_id="signal-scientist",
        name="Signal Scientist",
        status="idle",
        last_event=None,
        elapsed_seconds=None,
        current_hypothesis=None,
        progress_percent=None,
        stats=None,
    )

    assert status.agent_id == "signal-scientist"
    assert status.status == "idle"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/dashboard/test_agents_monitor.py::test_agent_status_dataclass -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'hrp.dashboard.agents_monitor'"

**Step 3: Create module skeleton**

Create `hrp/dashboard/agents_monitor.py`:

```python
"""Agents Monitor - Backend functions for agent status and timeline."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentStatus:
    """Status of a single agent."""
    agent_id: str
    name: str
    status: str  # running, completed, failed, idle
    last_event: dict | None
    elapsed_seconds: int | None
    current_hypothesis: str | None
    progress_percent: float | None
    stats: dict | None


def get_all_agent_status(api: Any = None) -> list[AgentStatus]:
    """
    Get current status of all agents from lineage events.

    Returns list of AgentStatus objects.
    """
    # Placeholder - will implement in next task
    return []


def get_timeline(
    api: Any,
    agents: list[str],
    statuses: list[str],
    date_range: tuple,
    limit: int,
) -> list[dict]:
    """
    Get historical timeline of agent events.

    Returns list of event dicts sorted by timestamp descending.
    """
    # Placeholder - will implement later
    return []
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/dashboard/test_agents_monitor.py::test_agent_status_dataclass -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/dashboard/agents_monitor.py tests/dashboard/test_agents_monitor.py
git commit -m "feat(dashboard): create agents monitor backend skeleton

- Add AgentStatus dataclass
- Add get_all_agent_status() placeholder
- Add get_timeline() placeholder
"
```

---

### Task 3: Implement get_all_agent_status

**Files:**
- Modify: `hrp/dashboard/agents_monitor.py`
- Test: `tests/dashboard/test_agents_monitor.py`

**Step 1: Write the failing test**

Add to `tests/dashboard/test_agents_monitor.py`:

```python
def test_get_all_agent_status_returns_all_agents():
    """Should return status for all 11 agents."""
    from unittest.mock import Mock
    from hrp.dashboard.agents_monitor import get_all_agent_status

    api = Mock()
    statuses = get_all_agent_status(api)

    assert len(statuses) == 11
    agent_ids = [s.agent_id for s in statuses]
    assert "signal-scientist" in agent_ids
    assert "alpha-researcher" in agent_ids
    assert "code-materializer" in agent_ids

def test_get_all_agent_status_infers_idle_from_no_events():
    """Agent with no lineage events should be idle."""
    from unittest.mock import Mock, patch
    from hrp.dashboard.agents_monitor import get_all_agent_status

    api = Mock()
    with patch('hrp.dashboard.agents_monitor.get_lineage', return_value=[]):
        statuses = get_all_agent_status(api)

        signal_scientist = next(s for s in statuses if s.agent_id == "signal-scientist")
        assert signal_scientist.status == "idle"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/dashboard/test_agents_monitor.py::test_get_all_agent_status_returns_all_agents -v
```

Expected: FAIL (returns empty list from placeholder)

**Step 3: Implement get_all_agent_status**

Replace placeholder in `hrp/dashboard/agents_monitor.py`:

```python
"""Agents Monitor - Backend functions for agent status and timeline."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from hrp.research.lineage import get_lineage


@dataclass
class AgentStatus:
    """Status of a single agent."""
    agent_id: str
    name: str
    status: str  # running, completed, failed, idle
    last_event: dict | None
    elapsed_seconds: int | None
    current_hypothesis: str | None
    progress_percent: float | None
    stats: dict | None


# All agent IDs in the system
AGENT_IDS = [
    "signal-scientist",
    "alpha-researcher",
    "code-materializer",
    "ml-scientist",
    "ml-quality-sentinel",
    "quant-developer",
    "pipeline-orchestrator",
    "validation-analyst",
    "risk-manager",
    "cio",
    "report-generator",
]


def get_all_agent_status(api: Any = None) -> list[AgentStatus]:
    """
    Get current status of all agents from lineage events.

    Returns list of AgentStatus objects.
    """
    all_statuses = []

    for agent_id in AGENT_IDS:
        # Get last 10 events for this agent
        events = get_lineage(
            actor=f"agent:{agent_id}",
            limit=10,
        )

        status = _infer_agent_status(agent_id, events)
        all_statuses.append(status)

    return all_statuses


def _infer_agent_status(agent_id: str, events: list) -> AgentStatus:
    """Infer agent status from recent lineage events."""
    # Display name
    name = agent_id.replace("-", " ").title()

    # No events = idle
    if not events:
        return AgentStatus(
            agent_id=agent_id,
            name=name,
            status="idle",
            last_event=None,
            elapsed_seconds=None,
            current_hypothesis=None,
            progress_percent=None,
            stats=None,
        )

    latest = events[0]
    now = datetime.now(timezone.utc)
    time_since = (now - latest["timestamp"]).total_seconds()

    # Check for active run (start without complete)
    has_start = any(e["event_type"] == "agent_run_start" for e in events)
    has_complete = any("complete" in e["event_type"].lower() for e in events)

    # Determine status
    if has_start and not has_complete and time_since < 300:  # 5 min timeout
        status = "running"
        elapsed = int(time_since)
        current_hyp = latest.get("hypothesis_id")
        progress = _extract_progress(latest.get("details", {}))
        stats = _extract_stats(latest.get("details", {}))
    elif _is_failed_event(latest):
        status = "failed"
        elapsed = None
        current_hyp = None
        progress = None
        stats = None
    elif "complete" in latest["event_type"].lower():
        status = "completed"
        elapsed = None
        current_hyp = None
        progress = None
        stats = _extract_stats(latest.get("details", {}))
    else:
        status = "idle"
        elapsed = None
        current_hyp = None
        progress = None
        stats = None

    return AgentStatus(
        agent_id=agent_id,
        name=name,
        status=status,
        last_event=latest,
        elapsed_seconds=elapsed,
        current_hypothesis=current_hyp,
        progress_percent=progress,
        stats=stats,
    )


def _is_failed_event(event: dict) -> bool:
    """Check if event represents a failure."""
    if "failed" in event["event_type"].lower():
        return True
    details = event.get("details", {})
    if details.get("error"):
        return True
    return False


def _extract_progress(details: dict) -> float | None:
    """Extract progress percentage from event details."""
    if "progress" in details:
        return details["progress"] * 100
    if "hypotheses_processed" in details and "hypotheses_total" in details:
        total = details["hypotheses_total"]
        if total > 0:
            return (details["hypotheses_processed"] / total) * 100
    return None


def _extract_stats(details: dict) -> dict | None:
    """Extract stats from event details."""
    stats = {}
    if "hypotheses_processed" in details:
        stats["processed"] = details["hypotheses_processed"]
    if "hypotheses_promoted" in details:
        stats["promoted"] = details["hypotheses_promoted"]
    if "experiments_audited" in details:
        stats["audited"] = details["experiments_audited"]
    if "experiments_passed" in details:
        stats["passed"] = details["experiments_passed"]
    return stats if stats else None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/dashboard/test_agents_monitor.py::test_get_all_agent_status_returns_all_agents -v
pytest tests/dashboard/test_agents_monitor.py::test_get_all_agent_status_infers_idle_from_no_events -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/dashboard/agents_monitor.py tests/dashboard/test_agents_monitor.py
git commit -m "feat(dashboard): implement get_all_agent_status

- Returns status for all 11 agents
- Infers status from lineage events (idle/running/completed/failed)
- Extracts progress, stats from event details
- Tests: all agents returned, idle inference from no events
"
```

---

### Task 4: Create Streamlit Page Skeleton

**Files:**
- Create: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Step 1: Create basic Streamlit page**

Create `hrp/dashboard/pages/2_Agents_Monitor.py`:

```python
"""Agents Monitor - Real-time and historical view of all research agents."""

import streamlit as st

st.set_page_config(
    page_title="Agents Monitor",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agents Monitor")
st.markdown("Real-time monitoring and historical audit of all research agents.")

# Placeholder sections
st.header("Real-Time Monitor")
st.info("Coming soon...")

st.markdown("---")
st.header("Historical Timeline")
st.info("Coming soon...")
```

**Step 2: Verify page loads**

Start Streamlit:
```bash
streamlit run hrp/dashboard/app.py
```

Navigate to http://localhost:8501/Agents_Monitor

Expected: Page loads without errors, shows "Coming soon..." placeholders

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/2_Agents_Monitor.py
git commit -m "feat(dashboard): create Agents Monitor page skeleton

- Basic Streamlit page with placeholders
- Ready for real-time and timeline sections
"
```

---

### Task 5: Implement Real-Time Monitor Section

**Files:**
- Modify: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Step 1: Update page with real-time section**

Replace content in `hrp/dashboard/pages/2_Agents_Monitor.py`:

```python
"""Agents Monitor - Real-time and historical view of all research agents."""

import streamlit as st
from datetime import datetime

from hrp.dashboard.agents_monitor import get_all_agent_status

st.set_page_config(
    page_title="Agents Monitor",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agents Monitor")

# =============================================================================
# REAL-TIME MONITOR SECTION
# =============================================================================

st.header("Real-Time Monitor")

# Control bar
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Live status of all research agents")
with col2:
    if st.button("Refresh"):
        st.rerun()

# Get agent statuses
agents = get_all_agent_status()

# Display in 4-column grid
cols = st.columns(4)
for idx, agent in enumerate(agents):
    with cols[idx % 4]:
        _render_agent_card(agent)


def _render_agent_card(agent):
    """Render a single agent status card."""
    status_colors = {
        "running": "🟦",
        "completed": "🟢",
        "failed": "🔴",
        "idle": "⚪",
    }
    status_icon = status_colors.get(agent.status, "⚪")

    st.markdown(f"### {status_icon} {agent.name}")
    st.markdown(f"**Status:** `{agent.status.upper()}`")

    if agent.status == "running":
        if agent.elapsed_seconds is not None:
            st.caption(f"⏱ Elapsed: {agent.elapsed_seconds}s")
        if agent.current_hypothesis:
            st.caption(f"📋 `{agent.current_hypothesis}`")
        if agent.progress_percent is not None:
            st.progress(agent.progress_percent / 100)

    if agent.last_event:
        st.caption(f"🕐 Last: {_format_timestamp(agent.last_event['timestamp'])}")

    if agent.stats:
        for key, value in agent.stats.items():
            st.caption(f"{key}: {value}")

    st.markdown("---")


def _format_timestamp(ts):
    """Format timestamp for display."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.strftime("%H:%M:%S")


# =============================================================================
# HISTORICAL TIMELINE SECTION (Placeholder)
# =============================================================================

st.markdown("---")
st.header("Historical Timeline")
st.info("Coming soon...")
```

**Step 2: Verify page displays agents**

Refresh Streamlit page at http://localhost:8501/Agents_Monitor

Expected:
- 11 agent cards displayed in 4-column grid
- Each card shows name, status icon, status badge
- Cards show last event timestamp

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/2_Agents_Monitor.py
git commit -m "feat(dashboard): implement real-time monitor section

- Display all 11 agents in 4-column grid
- Color-coded status icons (🟦 🟢 🔴 ⚪)
- Show elapsed time for running agents
- Show progress bar when available
- Show last event timestamp
"
```

---

### Task 6: Add Auto-Refresh

**Files:**
- Modify: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Step 1: Add auto-refresh logic**

Update the page to include auto-refresh:

```python
"""Agents Monitor - Real-time and historical view of all research agents."""

import streamlit as st
import time
from datetime import datetime

from hrp.dashboard.agents_monitor import get_all_agent_status

st.set_page_config(
    page_title="Agents Monitor",
    page_icon="🤖",
    layout="wide",
)

# Session state for auto-refresh
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
    st.session_state.refresh_interval = 5
    st.session_state.last_activity = None

st.title("🤖 Agents Monitor")

# =============================================================================
# REAL-TIME MONITOR SECTION
# =============================================================================

st.header("Real-Time Monitor")

# Control bar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("Live status of all research agents")
with col2:
    auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
with col3:
    if st.button("Refresh Now"):
        st.rerun()

# Get agent statuses
agents = get_all_agent_status()

# Display in 4-column grid
cols = st.columns(4)
for idx, agent in enumerate(agents):
    with cols[idx % 4]:
        _render_agent_card(agent)


def _render_agent_card(agent):
    """Render a single agent status card."""
    status_colors = {
        "running": "🟦",
        "completed": "🟢",
        "failed": "🔴",
        "idle": "⚪",
    }
    status_icon = status_colors.get(agent.status, "⚪")

    st.markdown(f"### {status_icon} {agent.name}")
    st.markdown(f"**Status:** `{agent.status.upper()}`")

    if agent.status == "running":
        if agent.elapsed_seconds is not None:
            st.caption(f"⏱ Elapsed: {agent.elapsed_seconds}s")
        if agent.current_hypothesis:
            st.caption(f"📋 `{agent.current_hypothesis}`")
        if agent.progress_percent is not None:
            st.progress(agent.progress_percent / 100)

    if agent.last_event:
        st.caption(f"🕐 Last: {_format_timestamp(agent.last_event['timestamp'])}")

    if agent.stats:
        for key, value in agent.stats.items():
            st.caption(f"{key}: {value}")

    st.markdown("---")


def _format_timestamp(ts):
    """Format timestamp for display."""
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return ts.strftime("%H:%M:%S")


# =============================================================================
# HISTORICAL TIMELINE SECTION (Placeholder)
# =============================================================================

st.markdown("---")
st.header("Historical Timeline")
st.info("Coming soon...")

# =============================================================================
# AUTO-REFRESH LOGIC
# =============================================================================

if st.session_state.auto_refresh:
    # Check for active agents
    active_count = sum(1 for a in agents if a.status == "running")

    if active_count > 0:
        st.session_state.last_activity = datetime.now()
        st.session_state.refresh_interval = 2  # Fast refresh
    elif st.session_state.last_activity:
        idle_time = (datetime.now() - st.session_state.last_activity).total_seconds()
        if idle_time > 30:
            st.session_state.refresh_interval = 10  # Slow refresh

    time.sleep(st.session_state.refresh_interval)
    st.rerun()
```

**Step 2: Verify auto-refresh works**

1. Enable "Auto-refresh" checkbox
2. Watch page refresh automatically
3. Uncheck checkbox → refresh stops

Expected: Page refreshes every 5 seconds (default), 2s when agents running

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/2_Agents_Monitor.py
git commit -m "feat(dashboard): add hybrid auto-refresh

- Auto-refresh checkbox to enable/disable
- Fast refresh (2s) when agents running
- Slow refresh (10s) when idle for 30+ seconds
- Manual refresh button
"
```

---

## Phase 2: Enhanced Features

### Task 7: Implement get_timeline Function

**Files:**
- Modify: `hrp/dashboard/agents_monitor.py`
- Test: `tests/dashboard/test_agents_monitor.py`

**Step 1: Write the failing test**

Add to `tests/dashboard/test_agents_monitor.py`:

```python
def test_get_timeline_returns_events():
    """Should return historical events sorted by timestamp desc."""
    from unittest.mock import Mock, patch
    from hrp.dashboard.agents_monitor import get_timeline

    api = Mock()
    with patch('hrp.dashboard.agents_monitor.get_lineage') as mock_get:
        # Mock lineage response
        mock_get.return_value = [
            {
                "lineage_id": 1,
                "timestamp": datetime(2026, 1, 29, 14, 30),
                "actor": "agent:signal-scientist",
                "event_type": "agent_run_complete",
                "hypothesis_id": "HYP-001",
                "details": {"hypotheses_processed": 5},
            },
            {
                "lineage_id": 2,
                "timestamp": datetime(2026, 1, 29, 14, 25),
                "actor": "agent:alpha-researcher",
                "event_type": "agent_run_complete",
                "hypothesis_id": "HYP-002",
                "details": {},
            },
        ]

        timeline = get_timeline(
            api=api,
            agents=["Signal Scientist", "Alpha Researcher"],
            statuses=["Completed"],
            date_range=(datetime(2026, 1, 1), datetime(2026, 1, 31)),
            limit=50,
        )

        assert len(timeline) == 2
        assert timeline[0]["agent_name"] == "Signal Scientist"
        assert timeline[1]["agent_name"] == "Alpha Researcher"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/dashboard/test_agents_monitor.py::test_get_timeline_returns_events -v
```

Expected: FAIL (returns empty list from placeholder)

**Step 3: Implement get_timeline**

Replace placeholder in `hrp/dashboard/agents_monitor.py`:

```python
def get_timeline(
    api: Any,
    agents: list[str],
    statuses: list[str],
    date_range: tuple,
    limit: int,
) -> list[dict]:
    """
    Get historical timeline of agent events.

    Returns list of event dicts sorted by timestamp descending.
    """
    start_date, end_date = date_range

    # Build actor list from agent names
    actors = [f"agent:{_name_to_id(a)}" for a in agents]

    # Query lineage
    events = get_lineage(
        actors=actors,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    # Enrich with display info
    timeline = []
    for event in events:
        timeline_event = {
            "event_id": event["lineage_id"],
            "timestamp": event["timestamp"],
            "agent_name": _format_agent_name(event["actor"]),
            "status": _infer_event_status(event),
            "hypothesis_id": event.get("hypothesis_id"),
            "duration_seconds": event["details"].get("duration_seconds"),
            "items_processed": event["details"].get("hypotheses_processed"),
            "metrics": _get_event_metrics(event["details"]),
            "error": event["details"].get("error"),
            "mlflow_run_id": event["details"].get("mlflow_run_id"),
        }
        timeline.append(timeline_event)

    return timeline


def _name_to_id(name: str) -> str:
    """Convert display name to agent ID."""
    return name.lower().replace(" ", "-")


def _format_agent_name(actor: str) -> str:
    """Format actor ID for display."""
    if actor.startswith("agent:"):
        name = actor.replace("agent:", "").replace("-", " ")
        return name.title()
    return actor


def _infer_event_status(event: dict) -> str:
    """Infer status from event type and details."""
    event_type = event["event_type"]
    details = event.get("details", {})

    if "complete" in event_type.lower():
        if details.get("error") or "failed" in event_type.lower():
            return "failed"
        return "completed"
    if "failed" in event_type.lower():
        return "failed"

    return "completed"


def _get_event_metrics(details: dict) -> dict:
    """Extract metrics from event details."""
    metrics = {}
    metric_keys = ["sharpe", "ic", "stability_score", "processed", "passed", "flagged"]
    for key in metric_keys:
        if key in details:
            metrics[key] = details[key]
    return metrics
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/dashboard/test_agents_monitor.py::test_get_timeline_returns_events -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add hrp/dashboard/agents_monitor.py tests/dashboard/test_agents_monitor.py
git commit -m "feat(dashboard): implement get_timeline function

- Query lineage for agent events
- Enrich with display info (agent name, status, metrics)
- Sort by timestamp descending
- Tests: returns events, sorts correctly
"
```

---

### Task 8: Implement Historical Timeline Section

**Files:**
- Modify: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Step 1: Add timeline UI to page**

Update the Historical Timeline section:

```python
# =============================================================================
# HISTORICAL TIMELINE SECTION
# =============================================================================

st.markdown("---")
st.header("Historical Timeline")

# Filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    agent_filter = st.multiselect(
        "Filter by Agent",
        options=[a.name for a in agents],
        default=[a.name for a in agents],
    )

with col2:
    status_filter = st.multiselect(
        "Filter by Status",
        options=["Running", "Completed", "Failed", "Idle"],
        default=["Completed", "Failed"],
    )

with col3:
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    )

with col4:
    limit = st.selectbox("Show", options=[50, 100, 200, 500], index=0)

# Fetch and display timeline
if st.button("Load Timeline", key="load_timeline"):
    timeline = get_timeline(
        api=None,
        agents=agent_filter,
        statuses=status_filter,
        date_range=(date_range[0], date_range[1] if len(date_range) > 1 else date_range[0]),
        limit=limit,
    )

    for event in timeline:
        _render_timeline_event(event)


def _render_timeline_event(event):
    """Render expandable timeline event."""
    status_icons = {
        "completed": "✅",
        "failed": "❌",
        "running": "🔄",
    }
    icon = status_icons.get(event["status"], "📌")

    with st.expander(
        f"{icon} **{event['agent_name']}** — {event['status'].upper()} "
        f"• {_format_timestamp(event['timestamp'])} "
        f"• `{event.get('hypothesis_id', 'N/A')}`"
    ):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Details**")
            if event.get("duration_seconds"):
                st.caption(f"⏱ {event['duration_seconds']}s")
            if event.get("items_processed"):
                st.caption(f"📊 {event['items_processed']} items")

        with col2:
            st.markdown("**Results**")
            for key, value in event.get("metrics", {}).items():
                st.caption(f"{key}: {value}")
            if event.get("error"):
                st.error(event["error"])

        with col3:
            st.markdown("**Actions**")
            if event.get("mlflow_run_id"):
                st.link_button(
                    "View in MLflow",
                    f"http://localhost:5000/experiments/{event['mlflow_run_id']}",
                )
```

**Step 2: Verify timeline displays**

1. Click "Load Timeline" button
2. Should see expandable timeline events

Expected: Timeline events shown, expandable for details

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/2_Agents_Monitor.py
git commit -m "feat(dashboard): implement historical timeline section

- Filters: agent, status, date range, limit
- Load Timeline button to fetch events
- Expandable events showing details, metrics, errors
- Action links to MLflow
"
```

---

### Task 9: Add Caching for Performance

**Files:**
- Modify: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Step 1: Add Streamlit caching**

Update the page to use `@st.cache_data`:

```python
# At top of file, add:
@st.cache_data(ttl=5)
def get_all_agent_status_cached() -> list:
    """Cached version of get_all_agent_status."""
    return get_all_agent_status()

@st.cache_data(ttl=10)
def get_timeline_cached(...) -> list:
    """Cached version of get_timeline."""
    return get_timeline(...)

# In page body, replace calls with cached versions:
# agents = get_all_agent_status_cached()
# timeline = get_timeline_cached(...)
```

**Step 2: Verify caching works**

1. Load page, note response time
2. Refresh immediately, should be faster (cached)

Expected: Second load faster due to cache

**Step 3: Commit**

```bash
git add hrp/dashboard/pages/2_Agents_Monitor.py
git commit -m "perf(dashboard): add caching for better performance

- Cache agent status for 5 seconds
- Cache timeline for 10 seconds
- Reduces database queries on frequent refreshes
"
```

---

## Phase 3: Polish

### Task 10: Add Error Handling

**Files:**
- Modify: `hrp/dashboard/pages/2_Agents_Monitor.py`

**Step 1: Wrap queries in try-except**

Update the page with error handling:

```python
# Get agent statuses with error handling
try:
    agents = get_all_agent_status()
except Exception as e:
    st.error(f"Failed to load agent status: {e}")
    st.button("Retry", on_click=lambda: st.rerun())
    agents = []

# Display in 4-column grid
if agents:
    cols = st.columns(4)
    for idx, agent in enumerate(agents):
        with cols[idx % 4]:
            _render_agent_card(agent)
```

**Step 2: Commit**

```bash
git add hrp/dashboard/pages/2_Agents_Monitor.py
git commit -m "feat(dashboard): add error handling

- Try-except around agent status query
- User-friendly error message
- Retry button on failure
"
```

---

### Task 11: Add Comprehensive Tests

**Files:**
- Modify: `tests/dashboard/test_agents_monitor.py`

**Step 1: Add test for status inference**

Add tests for all status types:

```python
def test_infers_running_status():
    """Agent with start event but no complete should be running."""
    # Test implementation

def test_infers_completed_status():
    """Agent with complete event should be completed."""
    # Test implementation

def test_infers_failed_status():
    """Agent with error in details should be failed."""
    # Test implementation
```

**Step 2: Run all tests**

```bash
pytest tests/dashboard/test_agents_monitor.py -v
```

Expected: All pass

**Step 3: Commit**

```bash
git add tests/dashboard/test_agents_monitor.py
git commit -m "test(dashboard): add comprehensive status inference tests

- Tests for running, completed, failed status
- Tests for progress extraction
- Tests for stats extraction
"
```

---

### Task 12: Update Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add usage example to CLAUDE.md**

```python
### Monitor Agents

Open the Agents Monitor dashboard to see real-time status and historical timeline:

```bash
streamlit run hrp/dashboard/app.py
# Navigate to: http://localhost:8501/Agents_Monitor
```

Features:
- Real-time status for all 11 agents
- Auto-refresh when agents are active
- Historical timeline with filters
- Expandable event details
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Agents Monitor usage to CLAUDE.md

- Instructions for launching dashboard
- Feature overview
"
```

---

### Task 13: Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 2: Manual smoke test**

1. Start dashboard: `streamlit run hrp/dashboard/app.py`
2. Navigate to Agents Monitor page
3. Verify:
   - [ ] All 11 agents display
   - [ ] Status icons correct
   - [ ] Auto-refresh works
   - [ ] Timeline loads
   - [ ] Filters work
   - [ ] Expandable events show details

**Step 3: Create summary document**

Create `docs/plans/2026-01-29-agents-monitor-summary.md`:

```markdown
# Agents Monitor Implementation Summary

**Date:** January 29, 2026
**Status:** Complete

## What Was Built

Streamlit dashboard for real-time monitoring and historical audit of all 11 HRP research agents.

## Features

### Real-Time Monitor
- 4-column grid of agent cards
- Color-coded status (🟦 Running, 🟢 Completed, 🔴 Failed, ⚪ Idle)
- Progress bars for running agents
- Elapsed time display
- Current hypothesis being processed
- Hybrid auto-refresh (2s active / 10s idle)

### Historical Timeline
- Chronological event list
- Filters: agent, status, date range, limit
- Expandable events with details
- Metrics display
- Error messages for failures
- Links to MLflow

## Files Created/Modified

- `hrp/dashboard/pages/2_Agents_Monitor.py` - Main dashboard page
- `hrp/dashboard/agents_monitor.py` - Backend functions
- `tests/dashboard/test_agents_monitor.py` - Tests
- `hrp/research/lineage.py` - Added AGENT_RUN_START event
- `CLAUDE.md` - Documentation

## Next Steps

- Add resource usage monitoring (CPU, memory)
- Add agent-specific details pages
- Add export to CSV
- Add alerts configuration
```

**Step 4: Commit summary**

```bash
git add docs/plans/2026-01-29-agents-monitor-summary.md
git commit -m "docs: add Agents Monitor implementation summary"
```

---

## Verification Checklist

- [ ] All 11 agents display with correct status
- [ ] Running agents show progress and elapsed time
- [ ] Auto-refresh toggles on/off
- [ ] Timeline loads with filters working
- [ ] Expandable events show full details
- [ ] Error handling shows user-friendly messages
- [ ] All tests pass
- [ ] Page loads in < 2 seconds
- [ ] Documentation updated

---

## Notes

- Follows TDD: write failing test, implement, verify pass, commit
- Each task is independent and can be done in isolation
- Use `superpowers:executing-plans` to execute this plan
- Reference design: `docs/plans/2026-01-29-agents-monitor-design.md`
