#!/usr/bin/env bash
#
# One-command local install for HRP consumer mode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${HRP_DATA_DIR:-$HOME/hrp-data}"

cd "$PROJECT_ROOT"

log() {
    printf '[HRP] %s\n' "$1"
}

find_python() {
    for candidate in "${PYTHON:-}" python3.12 python3.11 python3; do
        if [[ -z "$candidate" ]]; then
            continue
        fi
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
            then
                command -v "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3.11 or newer is required. Install it with Homebrew: brew install python@3.11" >&2
    exit 1
fi

log "Using Python: $("$PYTHON_BIN" --version)"

if [[ ! -d ".venv" ]]; then
    log "Creating local Python environment..."
    "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing HRP dependencies..."
python -m pip install --upgrade pip
if command -v uv >/dev/null 2>&1; then
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    uv pip install -e .
else
    python -m pip install -e .
fi

log "Creating local data folders..."
mkdir -p "$DATA_DIR"/{logs,auth,optuna,cache,output,backups,config,mlflow}

if [[ ! -f ".env" ]]; then
    log "Creating local .env with safe defaults..."
    python - <<'PY'
from pathlib import Path
import secrets

env = Path(".env")
template = Path(".env.example").read_text()
template = template.replace(
    "HRP_AUTH_COOKIE_KEY=your-secret-key-at-least-32-characters-long",
    f"HRP_AUTH_COOKIE_KEY={secrets.token_hex(32)}",
)
template = template.replace("HRP_BROKER_TYPE=ibkr", "HRP_BROKER_TYPE=paper")
template = template.replace("IBKR_ACCOUNT=DU123456", "IBKR_ACCOUNT=DU")
env.write_text(template)
env.chmod(0o600)
PY
fi

log "Initializing local database schema..."
set -a
# shellcheck disable=SC1091
source .env
set +a
python -m hrp.data.schema --init

log "Consumer mode install complete."
log "Open HRP with: ./scripts/open_hrp.sh"
