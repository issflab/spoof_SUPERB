#!/usr/bin/env bash
# Recompute the paper's Tables 5 and 6 from the score files and compare against
# the committed baseline. This is the numerical gate for any refactor.
#
#   bin/reproduce_table5.sh                       # recompute + diff vs baseline
#   bin/reproduce_table5.sh --out_dir /tmp/t5     # recompute only
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
if [ $# -gt 0 ]; then
    exec "$PY" -m spoof_superb.analysis.recompute_table5_mlaad_v10 "$@"
fi
echo "Recomputing Table 5 and diffing against tests/baseline_table5.json ..."
exec env RUN_TABLE5=1 "$PY" -m pytest "$REPO/tests/test_table5_regression.py" -v
