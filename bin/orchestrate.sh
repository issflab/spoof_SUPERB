#!/usr/bin/env bash
# ============================================================================
# Score EVERY model for one job, across GPUs, with resume and retry.
#
# Edit the settings block below, then run:   bin/orchestrate.sh
#   bin/orchestrate.sh --list     # show the tasks and exit, running nothing
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Which job:
#   mlaad       every linear head on MLAAD v10 fake
#   mailabs     every linear head on M-AILABS bonafide (writes to a staging dir)
#   spoofceleb  every linear head on the SpoofCeleb eval set
#   baselines   aasist_raw and lfcc_gmm on all 10 published sets
JOB="spoofceleb"

# GPUs to spread the work over.
GPUS="0 1 2"

# Parallel workers. 0 = one per GPU. Use 1 for a sequential run.
WORKERS=0

# Restrict to specific tasks: SSL model names, or dataset names for 'baselines'.
# Leave empty to run everything the job defines.
ONLY=""

# "yes" re-scores even when a complete, NaN-free output already exists.
FORCE="no"

# ------------------------------------------------------------ END SETTINGS --

ARGS=(--job "$JOB" --jobs "$WORKERS" --gpus $GPUS)
[ -n "$ONLY" ] && ARGS+=(--only $ONLY)
[ "$FORCE" = "yes" ] && ARGS+=(--force)

echo "+ python -m spoof_superb.orchestration.driver ${ARGS[*]} $*"
exec "$PY" -m spoof_superb.orchestration.driver "${ARGS[@]}" "$@"
