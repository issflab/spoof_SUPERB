#!/usr/bin/env bash
# ============================================================================
# Train the LFCC-GMM baseline (EM over two diagonal GMMs; CPU only, no torch).
#
# Edit the settings block below, then run:   bin/train_lfcc_gmm.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

N_JOBS=16          # parallel LFCC feature-extraction workers
N_COMPONENTS=512   # Gaussian components per class

# Leave empty to use database_path / protocols_path from configs/paths.yaml.
DATABASE_PATH=""
PROTOCOLS_PATH=""
OUT_DIR=""         # empty -> {baseline_models_root}/lfcc_gmm

# ------------------------------------------------------------ END SETTINGS --

ARGS=(--n_jobs "$N_JOBS" --ncomp "$N_COMPONENTS")
[ -n "$DATABASE_PATH" ]  && ARGS+=(--database_path "$DATABASE_PATH")
[ -n "$PROTOCOLS_PATH" ] && ARGS+=(--protocols_path "$PROTOCOLS_PATH")
[ -n "$OUT_DIR" ]        && ARGS+=(--out_dir "$OUT_DIR")

echo "+ python -m spoof_superb.train.lfcc_gmm ${ARGS[*]} $*"
exec "$PY" -m spoof_superb.train.lfcc_gmm "${ARGS[@]}" "$@"
