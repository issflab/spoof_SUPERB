#!/usr/bin/env bash
# ============================================================================
# Run the paper's three analyses over a score tree, then check the result.
#
#   1. main results        the benchmark table   (reads raw score files)
#   2. acoustic degradation  Section 4.4.2       (builds its view first)
#   3. TTS systems           Sections 4.4.3/3.2.3 (builds its view first)
#   4. verification, level 2  -- a SEPARATE step, see VERIFY below
#
# Only the last two build views. Main results reads the raw tree directly, so
# there is nothing to group it by.
#
# Step 4 is not folded into the analyses. Each analysis compares against
# nothing, and this script calls the verifier afterwards exactly as you would
# by hand -- so an analysis can never grade itself against the numbers it is
# trying to produce. Set VERIFY="no" and the three analyses are unaffected.
#
# Edit the settings block below, then run:   bin/analyze.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Which analyses to run:  all | main | degradation | tts
# Space-separated for a subset, e.g. WHICH="degradation tts".
WHICH="all"

# Where the tables and figures go. Each analysis writes a sub-directory under
# this: main_results/, degradation/, tts/. Empty = `outputs_root` in
# configs/paths.yaml, or the repo's outputs/ if that is unset.
OUT_ROOT=""

# Run level-2 verification after the analyses, comparing what was just written
# against reference/analysis/.
#
# "yes" is right when you are reproducing our results. Set it to "no" when you
# are analysing a NEW score tree that has no published reference yet -- the
# check would stop with "no reference analysis tables", which is true and
# unhelpful. Publish one with
#   python -m spoof_superb.tools.build_reference --from OUT_ROOT
# once you are satisfied with the run.
#
# It is skipped automatically when WHICH is a subset, because a partial run
# cannot answer the question the full check asks.
VERIFY="yes"

# --- reading a different tree ------------------------------------------------

# Score tree to analyse. Empty = the configured scores_root.
SCORES_ROOT_OVERRIDE=""

# Where the degradation and TTS views are written. Empty = beside the scores.
# The degradation view is ~4.5M rows per model, so point this at a disk with
# room if the score tree is on a small one.
VIEWS_ROOT=""

# "no" writes the CSVs and skips every figure. Useful when you only want the
# numbers, or when matplotlib is unavailable.
FIGURES="yes"

# ------------------------------------------------------------ END SETTINGS --

case "$WHICH" in
    all) STEPS="main degradation tts"; FULL="yes" ;;
    *)   STEPS="$WHICH";               FULL="no"  ;;
esac

# WHICH says "main"; the analysis sub-directory, and therefore the name the
# verifier's --tables filter matches on, is "main_results". Translate rather
# than print a command that would come back "matched nothing".
TABLE_NAMES=""
for s in $STEPS; do
    case "$s" in
        main) TABLE_NAMES="$TABLE_NAMES main_results" ;;
        *)    TABLE_NAMES="$TABLE_NAMES $s" ;;
    esac
done
TABLE_NAMES="${TABLE_NAMES# }"

for s in $STEPS; do
    case "$s" in
        main|degradation|tts) ;;
        *) echo "bin/analyze.sh: unknown analysis '$s'." >&2
           echo "  WHICH must be 'all', or any of: main degradation tts" >&2
           exit 2 ;;
    esac
done

# Flags shared by every analysis. Note that each analysis takes --out_dir as its
# OWN sub-directory, while verification takes the root above them -- so OUT_ROOT
# is threaded to the two of them differently on purpose.
COMMON=()
[ -n "$SCORES_ROOT_OVERRIDE" ] && COMMON+=(--scores_root "$SCORES_ROOT_OVERRIDE")

VIEW_ARGS=()
[ -n "$VIEWS_ROOT" ] && VIEW_ARGS+=(--out_root "$VIEWS_ROOT")
[ "$FIGURES" = "yes" ] || VIEW_ARGS+=(--no-figures)

# Run one analysis. Builds its argument list directly rather than through a
# helper that prints and a `mapfile` that reads back: that round trip needed
# bash 4, and an empty result tripped `set -u` on bash 4.3 and earlier.
run() {
    local label="$1" module="$2" sub="$3"; shift 3
    local args=("${COMMON[@]}" "$@")
    [ -n "$OUT_ROOT" ] && args+=(--out_dir "$OUT_ROOT/$sub")

    echo
    echo "=============================================================================="
    echo "  $label"
    echo "=============================================================================="
    echo "+ python -m $module ${args[*]}"
    "$PY" -m "$module" "${args[@]}" || {
        local rc=$?
        echo >&2
        echo "bin/analyze.sh: $label FAILED (exit $rc)." >&2
        echo "  The remaining analyses were not run, and nothing was verified:" >&2
        echo "  a partial set of tables cannot be checked against the reference." >&2
        exit "$rc"
    }
}

for s in $STEPS; do
    case "$s" in
      main)
        run "1/3  main results" \
            spoof_superb.analysis.recompute_main_results main_results
        ;;
      degradation)
        run "2/3  acoustic degradation (Section 4.4.2)" \
            spoof_superb.analysis.acoustic_degradation degradation \
            "${VIEW_ARGS[@]}"
        ;;
      tts)
        run "3/3  TTS systems (Sections 4.4.3 / 3.2.3)" \
            spoof_superb.analysis.tts_systems tts \
            "${VIEW_ARGS[@]}"
        ;;
    esac
done

if [ "$VERIFY" != "yes" ]; then
    echo
    echo "Analyses complete. VERIFY=\"no\", so nothing was checked."
    echo "  To check them:  bin/verify.sh   (set LEVEL=\"analysis\")"
    exit 0
fi

if [ "$FULL" != "yes" ]; then
    echo
    echo "Analyses complete. WHICH=\"$WHICH\" is a subset, so the level-2 check"
    echo "was skipped -- it grades all six tables, and the ones you did not"
    echo "produce would report as MISSING rather than as anything you can act on."
    echo "  To check just what you ran, e.g.:"
    echo "    python -m spoof_superb.verification analysis --tables $TABLE_NAMES"
    exit 0
fi

echo
echo "=============================================================================="
echo "  4/4  verification, level 2 -- do the paper's conclusions still hold?"
echo "=============================================================================="
VARGS=()
[ -n "$OUT_ROOT" ] && VARGS+=(--candidate "$OUT_ROOT")
echo "+ python -m spoof_superb.verification analysis ${VARGS[*]}"
exec "$PY" -m spoof_superb.verification analysis "${VARGS[@]}" "$@"
