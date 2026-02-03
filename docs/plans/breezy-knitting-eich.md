# Plan: Organize Research Output by Date + Pipeline Sequence Numbers

## Goal

Align research output (`~/hrp-data/output/research/`) with the same date-folder pattern used by reports (`~/hrp-data/output/reports/YYYY-MM-DD/`), add pipeline sequence numbers to filenames, and use `YYYY-MM-DDTHHMMSS` timestamps.

### Current state
```
output/research/2026-02-02-alpha-researcher.md      # flat, date-only, no sequence
output/reports/2026-02-02/2026-02-02-09-29-daily.md  # date subfolder, timestamp in name
```

### Target state
```
output/research/2026-02-02/2026-02-02T093015-02-alpha-researcher.md
output/research/2026-02-02/2026-02-02T093422-05-ml-quality-sentinel.md
output/reports/2026-02-02/2026-02-02T092900-daily.md   # (reports also get YYYYMMDDTHHMMSS)
```

## Pipeline Sequence Numbers

Based on `docs/agents/decision-pipeline.md` and scheduler trigger chain:

| ## | Agent | Current filename slug |
|----|-------|----------------------|
| 01 | Signal Scientist | (no research note currently) |
| 02 | Alpha Researcher | `alpha-researcher` |
| 03 | ML Scientist | (no research note currently) |
| 04 | ML Quality Sentinel | `ml-quality-sentinel` |
| 05 | Quant Developer | `quant-developer` |
| 06 | Pipeline Orchestrator | `kill-gates` |
| 07 | Validation Analyst | `validation-analyst` |
| 08 | Risk Manager | `risk-manager` |
| 09 | CIO Agent | `cio-review` (in reports/) |
| 10 | Report Generator | `daily`/`weekly` (in reports/) |

## Files to Modify

### 1. `hrp/agents/research_agents.py` (4 agents)

**ML Quality Sentinel** (~line 2453-2455):
```python
# Before
report_date = date.today().isoformat()
filename = f"{report_date}-ml-quality-sentinel.md"
filepath = get_config().data.research_dir / filename

# After
from hrp.agents.output_paths import research_note_path
filepath = research_note_path("04-ml-quality-sentinel")
```

**Validation Analyst** (~line 2971-2973): Same pattern, slug = `"07-validation-analyst"`

**Risk Manager** (~line 3847-3849): Same pattern, slug = `"08-risk-manager"`

**Quant Developer** (~line 4810-4812): Same pattern, slug = `"05-quant-developer"`

### 2. `hrp/agents/alpha_researcher.py` (~line 664-665)
```python
# Before
os.makedirs(config.research_note_dir, exist_ok=True)
filepath = f"{config.research_note_dir}/{today}-alpha-researcher.md"

# After
from hrp.agents.output_paths import research_note_path
filepath = research_note_path("02-alpha-researcher")
```

### 3. `hrp/agents/pipeline_orchestrator.py` (~line 638-743)
```python
# Before
os.makedirs(self.config.kill_gate_report_dir, exist_ok=True)
...
filepath = os.path.join(self.config.kill_gate_report_dir, f"{today}-kill-gates.md")

# After
from hrp.agents.output_paths import research_note_path
filepath = research_note_path("06-kill-gates")
```

### 4. `hrp/agents/report_generator.py` (~line 757-780)
Update `_get_report_filename()` and `_write_report()` to use `YYYY-MM-DDTHHMMSS`:
```python
# Before
f"{now.strftime('%Y-%m-%d-%H-%M')}-{self.report_type}.md"

# After
f"{now.strftime('%Y-%m-%dT%H%M%S')}-{self.report_type}.md"
```

### 5. `hrp/agents/cio.py` (~line 447)
Update CIO report filename:
```python
# Before
report_path = report_dir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}-cio-review.md"

# After
report_path = report_dir / f"{datetime.now().strftime('%Y-%m-%dT%H%M%S')}-09-cio-review.md"
```

### 6. NEW: `hrp/agents/output_paths.py` (helper module)

Centralized helper to avoid duplicating date-folder + timestamp logic across 6 agents:

```python
"""Centralized output path helpers for agent research notes."""

from datetime import datetime
from pathlib import Path

from hrp.utils.config import get_config


def research_note_path(slug: str) -> Path:
    """Build research note path with date subfolder and timestamp.

    Args:
        slug: e.g. "02-alpha-researcher", "05-quant-developer"

    Returns:
        Path like ~/hrp-data/output/research/2026-02-02/2026-02-02T093015-02-alpha-researcher.md
    """
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H%M%S")

    dir_path = get_config().data.research_dir / date_dir
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path / f"{timestamp}-{slug}.md"
```

### 7. Tests to update

- `tests/test_agents/test_alpha_researcher.py` - filepath assertions, config tests for `research_note_dir`
- `tests/test_agents/test_ml_quality_sentinel.py` - patched `_write_research_note`
- `tests/test_agents/test_validation_analyst.py` - patched `_write_research_note`
- `tests/test_agents/test_risk_manager.py` - `research_dir` mock usage
- `tests/test_agents/test_report_generator.py` - filename format assertions

## Verification

1. `pytest tests/ -v` - all tests pass
2. Run MCP tool `run_alpha_researcher` and verify output lands in `~/hrp-data/output/research/YYYY-MM-DD/` with correct naming
3. Run MCP tool `run_report_generator` and verify reports use new timestamp format
4. Run MCP tool `run_ml_quality_sentinel` and verify output path
5. Check that filenames sort correctly: `ls ~/hrp-data/output/research/2026-02-02/` should show files in pipeline order when multiple agents run
