#!/usr/bin/env bash
#
# Daily local refresh for consumer mode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
    ./scripts/consumer_install.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

LOG_DIR="${HRP_DATA_DIR:-$HOME/hrp-data}/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_job() {
    local job="$1"
    shift || true
    printf '[HRP] Running %s...\n' "$job"
    python -m hrp.agents.run_job --job "$job" "$@"
}

python -m hrp.data.schema --init
run_job universe
run_job prices
run_job features
run_job quality-monitoring
run_job recommendations
run_job daily-report

printf '[HRP] Daily consumer refresh complete. Logs: %s\n' "$LOG_DIR"
