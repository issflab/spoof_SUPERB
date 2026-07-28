#!/usr/bin/env bash
# ============================================================================
# Reproduce the paper's Tables 5 and 6 from the published score files.
#
# This is the fastest way to confirm a working setup: it reads score files
# only -- no GPU, no model checkpoints, no audio.
#
# Edit the settings block below, then run:   bin/reproduce_table5.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# "check"   recompute, then diff against tests/baseline_table5.json (~2m40s)
# "compute" recompute only, and write the tables to OUT_DIR
MODE="check"

OUT_DIR="$REPO/outputs/table5"

# ------------------------------------------------------------ END SETTINGS --

if [ "$MODE" = "compute" ]; then
    echo "+ recompute Tables 5/6 -> $OUT_DIR"
    exec "$PY" -m spoof_superb.analysis.recompute_table5_mlaad_v10 --out_dir "$OUT_DIR" "$@"
fi

echo "+ recompute Tables 5/6 and diff against tests/baseline_table5.json"
exec env RUN_TABLE5=1 "$PY" -m pytest "$REPO/tests/test_table5_regression.py" -v "$@"
