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

# "check"   code-regression gate: recompute and diff against
#           tests/baseline_main_results_table.json at zero tolerance (~2m40s).
#           This pins the CODE to the legacy tree the baseline was captured on.
#           To check a rebuilt tree against the published reference instead,
#           use bin/verify.sh -- that is the reproducibility check.
# "compute" recompute only, and write the tables to OUT_DIR
MODE="check"

# Where the tables go in "compute" mode. Empty = whatever `outputs_root` in
# configs/paths.yaml says, which is the setting every other analysis honours.
# Hardcoding a path here made that setting silently inapplicable to this script.
OUT_DIR=""

# ------------------------------------------------------------ END SETTINGS --

if [ "$MODE" = "compute" ]; then
    ARGS=()
    [ -n "$OUT_DIR" ] && ARGS+=(--out_dir "$OUT_DIR")
    echo "+ recompute the results tables${OUT_DIR:+ -> $OUT_DIR}"
    exec "$PY" -m spoof_superb.analysis.recompute_main_results "${ARGS[@]}" "$@"
fi

echo "+ recompute the results tables and diff against tests/baseline_main_results_table.json"
exec env RUN_MAIN_RESULTS=1 "$PY" -m pytest "$REPO/tests/test_main_results_regression.py" -v "$@"
