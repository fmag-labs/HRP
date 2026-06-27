#!/usr/bin/env bash
#
# Install, uninstall, or inspect the single consumer-mode daily LaunchAgent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LABEL="com.hrp.consumer-daily"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="${HRP_DATA_DIR:-$HOME/hrp-data}/logs"
HOUR="${HRP_CONSUMER_DAILY_HOUR:-18}"
MINUTE="${HRP_CONSUMER_DAILY_MINUTE:-30}"

install() {
    mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PROJECT_ROOT}/scripts/run_consumer_daily.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_ROOT}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>${HOUR}</integer>
        <key>Minute</key>
        <integer>${MINUTE}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/consumer-daily.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/consumer-daily.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>${PROJECT_ROOT}</string>
    </dict>
</dict>
</plist>
PLIST

    launchctl bootout "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load "$PLIST"
    echo "Installed ${LABEL}. It runs daily at ${HOUR}:${MINUTE} local time."
}

uninstall() {
    launchctl bootout "gui/$(id -u)/${LABEL}" >/dev/null 2>&1 || true
    launchctl unload "$PLIST" >/dev/null 2>&1 || true
    rm -f "$PLIST"
    echo "Removed ${LABEL}."
}

status() {
    if launchctl list 2>/dev/null | grep -q "$LABEL"; then
        launchctl list | grep "$LABEL"
    else
        echo "${LABEL} is not loaded."
    fi
}

case "${1:-}" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status}"
        exit 1
        ;;
esac
