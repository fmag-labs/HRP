#!/bin/bash
cd "$(dirname "$0")" || exit 1
./scripts/consumer_install.sh
echo
echo "Install complete. You can now open HRP with Open HRP.command."
read -r -p "Press Return to close this window."
