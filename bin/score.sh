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

# WHAT TO SCORE. This one setting decides the trial list, the audio root and
# where the output file is placed. Run `bin/score.sh --list_datasets` to see
# every dataset and where its trial list comes from.
#
#   from the corpus/protocol   spoofceleb  Multilingual  MAILABS  asvspoofLD
#   from a published score file (needs one present):
#     eval_2019  asvspoof2021_LA  asvspoof2021_DF  asvspoof5
#     deepfake_eval_2024  wild  Famous_Figures
DATASET="spoofceleb"

# Override where the trial list comes from. Leave EMPTY to let the dataset
# decide, which is almost always what you want.
#   benchmark | asvld | walk | protocol_csv
SOURCE=""

# Only for the asvld source: which laundering condition to score.
# Noise_Addition | Reverberation | Resampling | Recompression | Filtering
ASVLD_CONDITION="Noise_Addition"

# --- output -----------------------------------------------------------------
# Leave empty to place the file in the standard layout (see
# configs/paths.yaml). Set it to write somewhere else.
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
        --system "$MODEL" --dataset "$DATASET" --frontend "$FRONTEND")
fi

ARGS=(--model "$MODEL" --model_path "$MODEL_PATH" --output_file "$OUTPUT_FILE"
      --dataset "$DATASET" --cuda_device "$CUDA_DEVICE"
      --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" --n_jobs "$N_JOBS")

[ "$MODEL" = "linear_head" ] && ARGS+=(--ssl_model "$SSL_MODEL")
[ -n "$SOURCE" ] && ARGS+=(--source "$SOURCE")
[ "$SOURCE" = "asvld" ] && ARGS+=(--asvld_condition "$ASVLD_CONDITION")
[ "$USE_AMP" = "yes" ] && ARGS+=(--amp)

echo "+ python -m spoof_superb.scoring.driver ${ARGS[*]} $*"
exec "$PY" -m spoof_superb.scoring.driver "${ARGS[@]}" "$@"
