#!/usr/bin/env bash
# ============================================================================
# Reproduce the paper's two results tables from the published score files.
#
# This is the fastest way to confirm a working setup: it reads score files
# only -- no GPU, no model checkpoints, no audio.
#
# It COMPUTES. It compares against nothing. To check what it wrote against the
# published reference, run bin/verify.sh afterwards -- that is a separate step,
# and MODE="check" below does both in order.
#
# Edit the settings block below, then run:   bin/reproduce_main_results.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# "compute" recompute the tables and write them to OUT_DIR
# "check"   recompute, then verify the result against reference/analysis/ --
#           structure, cell deltas, and whether the paper's claims survive
MODE="check"

# Where the tables go. Empty = whatever `outputs_root` in configs/paths.yaml
# says, which is the setting every other analysis honours. Hardcoding a path
# here made that setting silently inapplicable to this script.
OUT_DIR=""

# ------------------------------------------------------------ END SETTINGS --

ARGS=()
[ -n "$OUT_DIR" ] && ARGS+=(--out_dir "$OUT_DIR")

echo "+ recompute the results tables${OUT_DIR:+ -> $OUT_DIR}"
"$PY" -m spoof_superb.analysis.recompute_main_results "${ARGS[@]}" "$@" || exit $?

[ "$MODE" = "check" ] || exit 0

echo
echo "+ verify the tables against reference/analysis/"
VARGS=()
[ -n "$OUT_DIR" ] && VARGS+=(--candidate "$(dirname "$OUT_DIR")")
exec "$PY" -m spoof_superb.verification analysis "${VARGS[@]}"
