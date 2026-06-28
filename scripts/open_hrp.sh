#!/usr/bin/env bash
#
# Start HRP in local consumer mode (API + web app) and open it in the browser.

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

# 1. Start the HTTP/JSON API (serves the web app's data).
./scripts/startup.sh start --api-only

# 2. Build and serve the Next.js consumer app.
if ! command -v npm >/dev/null 2>&1; then
    echo "Error: Node.js / npm is required for the HRP web app." >&2
    echo "Install it from https://nodejs.org and re-run." >&2
    exit 1
fi

WEB_PORT="${HRP_WEB_PORT:-3000}"

cd "$PROJECT_ROOT/web"
if [[ ! -d "node_modules" ]]; then
    echo "Installing web dependencies (first run)..."
    npm install
fi
if [[ ! -d ".next" ]]; then
    echo "Building the web app (first run)..."
    npm run build
fi

# Serve in the background; the API the app talks to defaults to localhost:8090.
PORT="$WEB_PORT" nohup npm run start > "$HOME/hrp-data/logs/web.out.log" 2>&1 &

URL="http://localhost:${WEB_PORT}"

# Wait briefly for the server to come up.
for _ in $(seq 1 30); do
    if curl -s -o /dev/null "$URL" 2>/dev/null; then break; fi
    sleep 1
done

if command -v open >/dev/null 2>&1; then
    open "$URL"
else
    printf 'HRP is running at %s\n' "$URL"
fi
