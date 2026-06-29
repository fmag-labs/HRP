# Running the Platform

A start-to-finish runbook: install → run → verify → configure. For the daily-use
framing and safety defaults see [Consumer Mode](consumer-mode.md); for examples
and recipes see the [Cookbook](cookbook.md).

## What you get

HRP runs as two cooperating pieces plus background services:

- **Consumer web app** (Next.js) at `http://localhost:3000` — Conviction List,
  Recommendation dossier, My Portfolio, Track Record, Research screens, Vault
  Assistant, Settings.
- **HTTP/JSON API** (FastAPI) at `http://localhost:8090` — the app's backend,
  served from `python -m hrp.api.http`.
- **Background services** — MLflow UI (`:5010`) and the research scheduler.

| Service | URL / command | Port |
|---|---|---|
| Web app | `./scripts/open_hrp.sh` or `cd web && npm run dev` | 3000 |
| API | `python -m hrp.api.http` | 8090 |
| MLflow UI | started by `hrp start` | 5010 |
| Ops/health | `python -m hrp.ops` | 8080 |

## Prerequisites

- macOS (tested on Apple Silicon), Python 3.11+, Homebrew.
- Node.js 18+ and npm (for the web app).

## 1. First-time setup

```bash
git clone https://github.com/fmag-labs/HRP.git
cd HRP
./scripts/setup.sh          # venv, deps, .env, DuckDB schema, bootstrap data
```

`setup.sh` is interactive and safe to re-run. It bootstraps ~2 years of daily
prices + 45 features for the top 20 S&P 500 names (no API key required — uses
Yahoo Finance). Use `./scripts/setup.sh --check` for verification only.

> **macOS shortcut:** double-click `Install HRP.command` instead of `setup.sh`.

## 2. Run it

The simplest path starts the API and the web app and opens your browser:

```bash
./scripts/open_hrp.sh        # → http://localhost:3000   (or: Open HRP.command)
```

Or use the unified CLI / run pieces individually:

```bash
hrp start                    # API + MLflow + scheduler
hrp start --full             # ...also the research agents
hrp start --api-only         # just the API

python -m hrp.api.http --port 8090      # API only
cd web && npm run dev                    # web app only (dev server)
```

The `hrp` CLI is the front door:

```bash
hrp status                   # what's running
hrp stop                     # stop all services
hrp doctor                   # setup checks + a data-freshness summary
hrp consult "..." --model gpt   # ask any configured LLM (see §5)
```

## 3. Verify it's working

```bash
hrp status                                   # services RUNNING
curl -s http://localhost:8090/api/health     # {"status":"ok"}
hrp doctor                                   # data freshness + counts
```

Then open `http://localhost:3000`. If data is stale or missing, the app shows a
banner explaining why (it never silently shows an empty chart).

## 4. Load more data

The bootstrap is a small slice. To load the full universe and keep it current:

```bash
python -m hrp.agents.run_job --job universe   # S&P 500 membership
python -m hrp.agents.run_job --job prices      # full universe (~20-30 min)
python -m hrp.agents.run_job --job features    # recompute features
```

To backfill a specific date range, see [Data Backfill](data-backfill.md).
Schedule these automatically via `hrp start --full` or the launchd jobs
(`./scripts/manage_launchd.sh install`).

## 5. Configure API keys (`.env`)

Most of the platform runs without keys, but some features need them. Edit `.env`
(see `.env.example` for the full list):

| Feature | Keys |
|---|---|
| Vault Assistant / research agents (Claude) | `ANTHROPIC_API_KEY` |
| Consult GPT / GLM | `OPENAI_API_KEY` / `ZAI_API_KEY` (+ optional `HRP_LLM_*` overrides) |
| Value / Dividends screens (fundamentals) | `SIMFIN_API_KEY`, then `run_job --job fundamentals` |
| Real-time / premium price data | `POLYGON_API_KEY` |
| Email reports | `RESEND_API_KEY`, `NOTIFICATION_EMAIL` |

After adding keys, `hrp consult --list-models` shows which providers are
`available`. The app's Assistant model selector reflects the same.

## 6. Multi-LLM consult

Ask any configured model — grounded (Vault Assistant in the app) or ad-hoc:

```bash
hrp consult --list-models
hrp consult "Summarize today's momentum screen" --model claude
curl -s -X POST http://localhost:8090/api/consult \
  -H 'Content-Type: application/json' \
  -d '{"question":"2+2?","model":"gpt"}'
```

## 7. Stopping & logs

```bash
hrp stop                     # stop all services
```

- Logs: `~/hrp-data/logs/` (e.g. `api.error.log`, `web.out.log`).
- Database: `~/hrp-data/hrp.duckdb`.

## Troubleshooting

| Symptom | Check |
|---|---|
| App shows "no market data" / stale banner | `hrp doctor`; load data (§4) |
| Assistant returns "not configured" (503) | the selected provider's key in `.env` (§5) |
| Value/Dividends screens empty | needs `SIMFIN_API_KEY` + `run_job --job fundamentals` |
| Port already in use | `HRP_API_PORT=8091 hrp start --api-only`, `HRP_WEB_PORT=3001 ./scripts/open_hrp.sh` |
| Web app can't reach API | API running on `:8090`? CORS origin matches (`HRP_API_CORS_ORIGINS`) |
