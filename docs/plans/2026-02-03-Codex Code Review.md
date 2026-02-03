# 2026-02-03 Codex Code Review

## Scope
Review of HRP codebase + documentation with emphasis on agent pipeline, data tables, and state machines. Sources include:
- Code: `hrp/data/schema.py`, `hrp/research/*`, `hrp/agents/*`, `hrp/api/platform.py`, `hrp/ml/*`, dashboard pages
- Docs: `README.md`, `docs/agents/decision-pipeline.md`, `docs/architecture/data-pipeline-diagram.md`, `docs/agents/09-cio-agent.md`

## Integrated Overview (Implemented Behavior)
- Core state is stored in DuckDB tables: `hypotheses`, `lineage`, `hypothesis_experiments`, agent infra tables, plus data/ML monitoring tables.
- Agents communicate via lineage events and the scheduler triggers in `hrp/agents/scheduler.py` and `hrp/agents/run_job.py`.
- MLflow experiments are linked to hypotheses via `hypothesis_experiments` and referenced in lineage events.

## Key State Machines (Observed)

### Hypotheses (registry + status transitions)
- Canonical statuses in schema: `draft`, `testing`, `validated`, `rejected`, `deployed`, `deleted`.
- Registry transitions enforced only in `hrp/research/hypothesis.py`.
- Agents use `PlatformAPI.update_hypothesis()` which does not enforce registry transitions.

### Lineage (event-driven pipeline)
- Upstream events drive downstream agents:
  - `hypothesis_created` -> Alpha Researcher
  - `alpha_researcher_complete` -> ML Scientist
  - `experiment_completed` -> ML Quality Sentinel
  - `ml_quality_sentinel_audit` -> Quant Developer
  - `quant_developer_backtest_complete` -> Kill Gate Enforcer
  - `kill_gate_enforcer_complete` -> Validation Analyst
  - `validation_analyst_complete` -> Risk Manager
  - `risk_manager_assessment` -> CIO Agent

### Ingestion Log
- `ingestion_log.status` should be `running` -> `completed` or `failed`.

### Model Deployments
- Schema expects `pending` -> `active` -> `rolled_back`.
- Code logs `pending` -> `success`/`failed`.

### Agent Infra
- `agent_checkpoints`: `completed` flag toggled from false to true.
- `agent_token_usage`: append-only token usage records.

## Inconsistencies / Gaps

1) **Hypothesis status mismatches (schema vs agents)**
- Agents write statuses not allowed by schema:
  - Quant Developer: `backtested`
  - Validation Analyst: `validation_failed`
  - Risk Manager: `risk_vetoed`
- These violate the schema CHECK constraint and are likely to fail inserts/updates (often swallowed by exception handling).

2) **Docs vs schema lifecycle divergence**
- Docs refer to `AUDITED`, `PASSED`, `VETOED`, etc., which are not valid schema statuses.
- `docs/architecture/data-pipeline-diagram.md` describes `validated` and `deployed` only; decision pipeline doc defines expanded lifecycle.

3) **Deployment status mismatch**
- Schema: `pending/active/rolled_back`.
- Code: `pending/success/failed`.
- This can silently fail logging or misreport deployment history.

4) **Ingestion status mismatch in dashboard**
- Dashboard queries for `status='success'`, but ingestion log writes `completed`.

5) **Documentation references missing artifacts**
- `README.md` references `docs/agents/agent-interaction-diagram.md`, but it does not exist.
- Decision pipeline describes a Code Materializer agent, but no implementation is present in `hrp/agents/`.
- `docs/agents/09-cio-agent.md` lists tables not present in schema (`paper_portfolio_history`, `cio_threshold_history`).

6) **Lineage enum vs schema mismatch**
- DB allows event types not present in `EventType` enum (e.g., `backtest_run`, `feature_computed`, `universe_update`).
- `hrp/research/lineage.log_event()` enforces enum values and cannot log these schema‑allowed events.

7) **Kill Gate Enforcer idempotency gap**
- Agent filters hypotheses by metadata `'%kill_gate_enforcer%'` but does not set that metadata after running.

8) **CIO approvals incomplete**
- `cio_decisions.approved`, `approved_by`, `approved_at` exist but are not written anywhere.

## Recommendations (Ordered)

1) **Unify hypothesis lifecycle**
- Decide between:
  - Schema-only statuses (keep `draft/testing/validated/rejected/deployed/deleted`, move extra states into metadata)
  - Expanded schema + registry transitions (add `backtested`, `validation_failed`, `risk_vetoed`, etc.)
- Enforce in one place (prefer PlatformAPI for all writers).

2) **Align deployment statuses**
- Either update schema to accept `success/failed` or map them to `active/rolled_back` in code.

3) **Fix ingestion status labels in UI**
- Use `completed` in dashboard queries or change ingestion logging to use `success`.

4) **Normalize lineage events**
- Add schema-only event types to `EventType` or relax enum enforcement to accept schema values.

5) **Repair/trim docs to match code**
- Remove missing references or add the missing artifacts/agents.
- Ensure CIO table list matches schema or implement missing tables.

6) **Close pipeline gaps**
- Add a `kill_gate_enforcer` metadata stamp to prevent reruns.
- Implement or remove Code Materializer from pipeline references.

## Appendix: Key Files Reviewed
- Schema: `hrp/data/schema.py`
- Registry: `hrp/research/hypothesis.py`, `hrp/research/lineage.py`
- Agents: `hrp/agents/*`
- API: `hrp/api/platform.py`
- Deployment: `hrp/ml/deployment.py`
- Docs: `README.md`, `docs/agents/decision-pipeline.md`, `docs/architecture/data-pipeline-diagram.md`, `docs/agents/09-cio-agent.md`
