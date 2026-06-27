#!/usr/bin/env bash
#
# go_live.sh — One command to make HRP run autonomously on this machine (macOS).
#
# What it does (idempotent, safe to re-run):
#   1. Bootstraps the environment (venv, deps, .env, DB) via consumer_install.sh
#      if not already present.
#   2. Checks for the API keys the agents need and warns if missing.
#   3. Seeds an initial data slice (universe -> prices -> features) when a data
#      source key is present, so the pipeline has something to work on.
#   4. Installs + loads ALL scheduled agents via launchd (path-corrected).
#   5. Prints status and how to watch / run a full cycle now.
#
# Usage:
#   ./scripts/go_live.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${HRP_DATA_DIR:-$HOME/hrp-data}"
cd "$PROJECT_ROOT"

log()  { printf '\033[0;34m[go-live]\033[0m %s\n' "$1"; }
ok()   { printf '\033[0;32m[go-live]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[go-live]\033[0m %s\n' "$1"; }

if [[ "$(uname -s)" != "Darwin" ]]; then
    warn "launchd is macOS-only; on this OS the agents won't be scheduled."
    warn "Bootstrap + an on-demand cycle still work (see run_full_cycle.sh)."
fi

# 1. Bootstrap environment if needed --------------------------------------------
if [[ ! -x ".venv/bin/python" ]]; then
    log "No virtualenv found — running consumer_install.sh to bootstrap..."
    ./scripts/consumer_install.sh
else
    ok "Virtualenv present."
fi

# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -f ".env" ]]; then
    set -a; # shellcheck disable=SC1091
    source .env; set +a
fi

# 2. Check the keys the agents need ---------------------------------------------
missing=()
[[ -z "${ANTHROPIC_API_KEY:-}" ]] && missing+=("ANTHROPIC_API_KEY (Alpha Researcher, CIO, reports)")
[[ -z "${POLYGON_API_KEY:-}" && -z "${ALPACA_API_KEY:-}" && -z "${TIINGO_API_KEY:-}" ]] && \
    missing+=("a market-data key e.g. POLYGON_API_KEY (price/feature ingestion)")
if (( ${#missing[@]} > 0 )); then
    warn "These keys are not set in .env — add them so the agents can do real work:"
    for m in "${missing[@]}"; do warn "    - $m"; done
    warn "Edit: $PROJECT_ROOT/.env   (then re-run this script)."
else
    ok "Required API keys present."
fi

# 3. Seed an initial data slice (best-effort) ----------------------------------
if [[ -n "${POLYGON_API_KEY:-}${ALPACA_API_KEY:-}${TIINGO_API_KEY:-}" ]]; then
    log "Seeding initial data (universe -> prices -> features)..."
    python -m hrp.data.schema --init || true
    for job in universe prices features; do
        log "  ingest: $job"
        python -m hrp.agents.run_job --job "$job" || warn "  $job ingestion had issues (continuing)."
    done
else
    warn "Skipping initial data seed (no market-data key). The scheduled prices/"
    warn "features jobs will populate data once a key is configured."
fi

# 4. Install + load all scheduled agents ---------------------------------------
if [[ "$(uname -s)" == "Darwin" ]]; then
    log "Installing all launchd agents..."
    ./scripts/manage_launchd.sh install
    echo ""
    ./scripts/manage_launchd.sh status
fi

# 5. Summary -------------------------------------------------------------------
echo ""
ok "HRP is set up for autonomous operation."
echo ""
echo "  Watch logs:        tail -f $DATA_DIR/logs/*.log"
echo "  Dashboard:         streamlit run hrp/dashboard/app.py   (Pipeline Progress / Agents Monitor)"
echo "  Run a full cycle now (don't wait for the schedule):"
echo "                     ./scripts/run_full_cycle.sh"
echo ""
echo "  Note: the research pipeline runs autonomously, but DEPLOYING a validated"
echo "  strategy is a human gate by design (you are the CIO). Approve deployments"
echo "  in the dashboard; recommendations then flow from deployed models."
