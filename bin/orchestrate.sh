#!/usr/bin/env bash
# Run a whole scoring job across models and GPUs.
#
#   bin/orchestrate.sh --job spoofceleb
#   bin/orchestrate.sh --job baselines --jobs 1
#   bin/orchestrate.sh --job mlaad --list
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PY" -m spoof_superb.orchestration.driver "$@"
