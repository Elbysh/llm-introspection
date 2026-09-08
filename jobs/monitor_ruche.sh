#!/bin/bash
# Run on your own machine; SSH prompts for your password in the terminal.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    echo "Usage: bash jobs/monitor_ruche.sh JOB_ID" >&2
    exit 2
fi

# Polling happens remotely inside one SSH connection, with no stored password.
remote_command='cd "$WORKDIR/llm-introspection" && .venv/bin/python code/utils/monitor_position_detection.py results/position_detection'
exec ssh -t aurouxw@ruche.mesocentre.universite-paris-saclay.fr \
    "$remote_command --job-id $1"
