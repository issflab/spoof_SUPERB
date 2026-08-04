#!/usr/bin/env bash
# ============================================================================
# Score EVERY model for one job, across GPUs, with resume and retry.
#
# Edit the settings block below, then run:   bin/orchestrate.sh
#   bin/orchestrate.sh --list     # show the tasks and exit, running nothing
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Which job. A job picks the back-ends; the three filters below narrow it
# further, and any combination is valid.
#   all          every system on every dataset
#   linear_head  every SSL head on every dataset
#   baselines    aasist_raw + lfcc_gmm on every dataset
# There is no per-dataset job: use DATASETS below. What the old mlaad /
# mailabs / spoofceleb jobs carried (grading policy, retry budget, model skips)
# now lives on the datasets themselves, so DATASETS="Multilingual" behaves the
# way JOB="mlaad" used to.
JOB="all"

# --- narrow the sweep. Leave empty to mean "all of them". -------------------

# Back-ends:  linear_head  aasist_raw  lfcc_gmm
SYSTEMS=""

# Datasets. Run `bin/score.sh --list_datasets` for the full set.
#   deepfake_eval_2024  wild  eval_2019  spoofceleb  asvspoof2021_LA
#   deepfake_eval_2024_segmented  Famous_Figures  MAILABS  asvspoof2021_DF
#   asvspoof5  Multilingual  asvspoofLD
DATASETS=""

# SSL upstreams, e.g. "xls_r_300m wavlm_large". Applies to linear_head only.
# Empty means "the models the paper reports" -- see PAPER_ONLY below, not "all 24".
MODELS=""

# "yes" scores only the 21 SSL upstreams the paper reports, out of the 24 trained
# heads on disk. Slugs are read from spoof_superb/scoring/paper_roster.json, so they cannot
# drift from the paper. "no" scores every head.
# Naming models in MODELS above overrides this either way.
PAPER_ONLY="yes"

# --- runtime ----------------------------------------------------------------

# GPUs to spread the work over.
GPUS="0 1 2"

# Parallel workers. 0 = one per GPU. Use 1 for a sequential run.
# (This is --workers. Not --job, which picks the job above.)
WORKERS=0

# Identity for this run. Empty = a timestamp. Two runs of one job no longer
# share a status file, so there is no reason to set this unless you want a
# memorable name in {scores_root}/_runs/{job}/.
RUN_NAME=""

# "yes" re-scores even when a complete, NaN-free output already exists.
FORCE="no"

# NOTE: this sweep does not compare anything. Scoring must not depend on a
# score file it did not just write, or the new tree can only ever reproduce the
# old one's coverage. To check a finished tree against the published reference,
# run  bin/verify.sh  -- it re-scores nothing.

# Live progress display.
#   auto   a redrawing bar on a terminal, one status line a minute when the
#          output is redirected. Correct in both cases; leave it here.
#   bar    force the redrawing bar
#   plain  force periodic status lines
#   none   only the per-task result lines
PROGRESS="auto"

# ------------------------------------------------------------ END SETTINGS --

ARGS=(--job "$JOB" --workers "$WORKERS" --gpus $GPUS --progress "$PROGRESS")
[ -n "$SYSTEMS" ]   && ARGS+=(--systems $SYSTEMS)
[ -n "$DATASETS" ]  && ARGS+=(--datasets $DATASETS)
[ -n "$MODELS" ]    && ARGS+=(--models $MODELS)
[ -n "$RUN_NAME" ]  && ARGS+=(--run-name "$RUN_NAME")
[ "$FORCE" = "yes" ]      && ARGS+=(--force)
[ "$PAPER_ONLY" != "yes" ] && ARGS+=(--all-models)

echo "+ python -m spoof_superb.orchestration.driver ${ARGS[*]} $*"
exec "$PY" -m spoof_superb.orchestration.driver "${ARGS[@]}" "$@"
