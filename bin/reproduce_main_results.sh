#!/usr/bin/env bash
# ============================================================================
# Reproduce the paper's two results tables from the published score files.
#
# This is the fastest way to confirm a working setup: it reads score files
# only -- no GPU, no model checkpoints, no audio.
#
# Edit the settings block below, then run:   bin/reproduce_main_results.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# "check"   recompute, then diff against tests/baseline_main_results_table.json (~2m40s)
# "compute" recompute only, and write the tables to OUT_DIR
MODE="check"

OUT_DIR="$REPO/outputs/main_results"

# ------------------------------------------------------------ END SETTINGS --

if [ "$MODE" = "compute" ]; then
    echo "+ recompute the results tables -> $OUT_DIR"
    exec "$PY" -m spoof_superb.analysis.recompute_main_results --out_dir "$OUT_DIR" "$@"
fi

echo "+ recompute the results tables and diff against tests/baseline_main_results_table.json"
exec env RUN_MAIN_RESULTS=1 "$PY" -m pytest "$REPO/tests/test_main_results_regression.py" -v "$@"
