#!/usr/bin/env bash
#
# Start the HRP dashboard in local consumer mode and open it in the browser.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
    ./scripts/consumer_install.sh
fi

if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

./scripts/startup.sh start --dashboard-only

PORT="${HRP_DASHBOARD_PORT:-8501}"
URL="http://localhost:${PORT}"

if command -v open >/dev/null 2>&1; then
    open "$URL"
else
    printf 'HRP is running at %s\n' "$URL"
fi
