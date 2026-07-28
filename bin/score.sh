#!/usr/bin/env bash
# ============================================================================
# Score ONE model on ONE evaluation set -> one score file.
#
# Edit the settings block below, then run:   bin/score.sh
# Anything you pass on the command line is appended, so you can still do
#   bin/score.sh --limit 300          (quick smoke test)
#   bin/score.sh --list_datasets      (show the 10 published sets and exit)
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Which back-end:  linear_head | aasist_raw | lfcc_gmm
MODEL="linear_head"

# s3prl upstream name. Required for linear_head; ignored by the other two.
SSL_MODEL="xls_r_300m"

# Checkpoint: swa.pth for linear_head / aasist_raw, a GMM DIRECTORY for lfcc_gmm.
MODEL_PATH="$MODELS_ROOT/${LINEAR_HEAD_PREFIX}${SSL_MODEL}/swa.pth"

# Where the trial list comes from:
#   benchmark     one of the 10 published sets (uses the reference score file)
#   asvld         one ASVLD laundering condition (uses the ASVLD protocol)
#   walk          every wav under a directory   (MLAAD, M-AILABS)
#   protocol_csv  a file,speaker,attack CSV     (SpoofCeleb)
SOURCE="benchmark"

# --- for SOURCE=benchmark ---------------------------------------------------
# eval_2019 | asvspoof2021_LA | asvspoof2021_DF | asvspoof5 |
# deepfake_eval_2024 | wild | Famous_Figures | spoofceleb |
# Multilingual | asvspoofLD
DATASET="wild"

# --- for SOURCE=asvld -------------------------------------------------------
# Noise_Addition | Reverberation | Resampling | Recompression | Filtering
ASVLD_CONDITION="Noise_Addition"

# --- for SOURCE=walk --------------------------------------------------------
WALK_ROOT="$DATA_ROOT/MLAAD/fake"     # M-AILABS is $DATA_ROOT/MAILabs
WALK_LABEL="spoof"                    # 'bonafide' for M-AILABS

# --- output -----------------------------------------------------------------
# Which benchmark set this run represents. Used to place the output in the
# configured layout (see score_layout in configs/paths.yaml). For SOURCE=walk
# this is Multilingual for MLAAD and MAILABS for M-AILABS.
SCORED_DATASET="$DATASET"

# Leave empty to place the file automatically. Set it to write somewhere else.
OUTPUT_FILE=""

# --- runtime ----------------------------------------------------------------
CUDA_DEVICE="cuda:0"
BATCH_SIZE=0        # 0 = the back-end default (linear_head 32, aasist_raw 64)
NUM_WORKERS=6
N_JOBS=16           # lfcc_gmm worker processes (CPU)
USE_AMP="no"        # KEEP THIS "no". fp16 overflow is what produced the NaN.

# ------------------------------------------------------------ END SETTINGS --

FRONTEND="none"
[ "$MODEL" = "linear_head" ] && FRONTEND="$SSL_MODEL"
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE=$("$PY" -m spoof_superb.core.scorepath \
        --system "$MODEL" --dataset "$SCORED_DATASET" --frontend "$FRONTEND")
fi

ARGS=(--model "$MODEL" --model_path "$MODEL_PATH" --output_file "$OUTPUT_FILE"
      --source "$SOURCE" --cuda_device "$CUDA_DEVICE"
      --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --n_jobs "$N_JOBS")

[ "$MODEL" = "linear_head" ] && ARGS+=(--ssl_model "$SSL_MODEL")

case "$SOURCE" in
  benchmark)    ARGS+=(--dataset "$DATASET") ;;
  asvld)        ARGS+=(--asvld_condition "$ASVLD_CONDITION") ;;
  walk)         ARGS+=(--walk_root "$WALK_ROOT" --label "$WALK_LABEL") ;;
  protocol_csv) ;;   # protocol and audio base come from configs/paths.yaml
  *) echo "bin/score.sh: unknown SOURCE '$SOURCE'" >&2; exit 2 ;;
esac

[ "$USE_AMP" = "yes" ] && ARGS+=(--amp)

echo "+ python -m spoof_superb.scoring.driver ${ARGS[*]} $*"
exec "$PY" -m spoof_superb.scoring.driver "${ARGS[@]}" "$@"
