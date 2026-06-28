#!/usr/bin/env bash
#
# run_full_cycle.sh — Drive one full agent cycle on demand so you can watch it,
# without waiting for the launchd schedule.
#
# Order mirrors the production flow:
#   data (universe -> prices -> features)
#   -> signal-scan        (Signal Scientist: discover signals, create hypotheses)
#   -> agent-pipeline x2  (Alpha Researcher -> ML Scientist -> ML Quality Sentinel
#                          -> Quant Developer -> Kill Gate -> Validation Analyst
#                          -> Risk Manager -> CIO, as lineage events accumulate)
#   -> cio-review         (CIO scoring of validated hypotheses)
#   -> recommendations    (advisory picks from deployed models)
#   -> daily-report       (summary)
#
# Pass --dry-run to see the wiring fire with no side effects.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

if [[ ! -x ".venv/bin/python" ]]; then
    echo "[cycle] No virtualenv — run ./scripts/go_live.sh first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -f ".env" ]]; then
    set -a; # shellcheck disable=SC1091
    source .env; set +a
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run() {
    printf '\033[0;34m[cycle]\033[0m === %s ===\n' "$1"
    python -m hrp.agents.run_job --job "$1" $DRY || \
        printf '\033[1;33m[cycle]\033[0m %s reported issues (continuing)\n' "$1"
}

python -m hrp.data.schema --init || true
run universe
run prices
run features
run signal-scan
run agent-pipeline
run agent-pipeline
run cio-review
run recommendations
run daily-report

printf '\033[0;32m[cycle]\033[0m Full cycle complete. Review recommendations in the app (./scripts/open_hrp.sh).\n'
