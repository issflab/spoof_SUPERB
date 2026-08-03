#!/usr/bin/env bash
# ============================================================================
# Verify a finished run against the published reference. Re-scores nothing.
#
# Two independent levels:
#
#   LEVEL 1  score files   -- did the pipeline produce the same scores?
#   LEVEL 2  analysis      -- do the same conclusions come out?
#
# They fail independently, and both are worth knowing. Identical scores with a
# changed table means the ANALYSIS code moved. Drifting scores with an intact
# table means the FINDING is robust to the drift.
#
# Edit the settings block below, then run:   bin/verify.sh
# Exit 0 = pass, 1 = something failed. The reports say what.
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Which levels to run:  scores | analysis | all
LEVEL="all"

# Level 1 reference.
#
# Empty  = MANIFEST MODE. Reads reference/manifest.json, which is in this repo.
#          Costs no download: the manifest carries a digest of each cell's
#          sorted trial list, so it can prove you scored the same utterances,
#          and the reference EER, so it can tell you whether the metric moved.
#          What it cannot do is separate a real score difference from a
#          flat-DET SENSITIVE cell -- that needs the scores themselves.
#
# A path = TREE MODE. Every utterance is compared. This is the full ladder and
#          the mode a provenance claim should cite. Fetch the reference files
#          first with  bin/fetch_scores.sh.
REF_ROOT=""
REF_LAYOUT="v3"

# Restrict the sweep. Empty means "all of them".
MODELS=""
DATASETS=""

# Famous Figures names its bonafide directory differently between trees.
# Asserting that the two conventions denote the same utterances is a claim the
# caller makes, so it is never inferred. Uncomment if you hit it.
# ID_REWRITE="--ref-id-rewrite -=Bonafide"
ID_REWRITE=""

# Level 2 candidate: the outputs root holding main_results/, degradation/, tts/.
# Empty = the configured outputs_root.
OUTPUTS=""

# ------------------------------------------------------------ END SETTINGS --

ARGS=()
[ -n "$REF_ROOT" ] && ARGS+=(--ref-root "$REF_ROOT" --ref-layout "$REF_LAYOUT")
[ -n "$MODELS" ]   && ARGS+=(--models $MODELS)
[ -n "$DATASETS" ] && ARGS+=(--datasets $DATASETS)
[ -n "$ID_REWRITE" ] && ARGS+=($ID_REWRITE)

case "$LEVEL" in
  scores)
    ;;
  analysis)
    ARGS=()
    [ -n "$OUTPUTS" ] && ARGS+=(--candidate "$OUTPUTS")
    ;;
  all)
    [ -n "$OUTPUTS" ] && ARGS+=(--analysis-candidate "$OUTPUTS")
    ;;
  *)
    echo "LEVEL must be scores, analysis or all (got '$LEVEL')" >&2; exit 2 ;;
esac

echo "+ python -m spoof_superb.verification $LEVEL ${ARGS[*]}"
exec "$PY" -m spoof_superb.verification "$LEVEL" "${ARGS[@]}" "$@"
