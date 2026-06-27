#!/bin/bash
cd "$(dirname "$0")" || exit 1
./scripts/install_consumer_launchd.sh install
echo
echo "Daily HRP is enabled."
read -r -p "Press Return to close this window."
