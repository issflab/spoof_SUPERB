#!/usr/bin/env bash
# ============================================================================
# Train a model.
#
# Edit the settings block below, then run:   bin/train.sh
# Corpus and output paths come from configs/paths.yaml.
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# aasist | linear_head | aasist_raw | lfcc_gmm
# NOTE: lfcc_gmm does not train here -- use bin/train_lfcc_gmm.sh.
MODEL_ARCH="linear_head"

# s3prl upstream. Ignored by aasist_raw.
SSL_MODEL="wavlm_large"

BATCH_SIZE=64
NUM_EPOCHS=50
LEARNING_RATE=0.000001
WEIGHT_DECAY=0.0001
LOSS="weighted_CCE"
SEED=1234

# Gradient accumulation: split BATCH_SIZE into micro-batches of this size.
# 0 disables it (one optimizer step per batch).
MICRO_BATCH=0

# RawBoost augmentation variant. 0 = none. See main.py --help for the list.
RAWBOOST_ALGO=5

# Tag recorded in the checkpoint directory name.
TRAIN_DATASET="ASV19"

# Free-text suffix on the checkpoint directory. Leave empty for none.
COMMENT=""

# ------------------------------------------------------------ END SETTINGS --

ARGS=(--model_arch "$MODEL_ARCH" --ssl_model "$SSL_MODEL"
      --batch_size "$BATCH_SIZE" --num_epochs "$NUM_EPOCHS"
      --lr "$LEARNING_RATE" --weight_decay "$WEIGHT_DECAY" --loss "$LOSS"
      --seed "$SEED" --micro_batch "$MICRO_BATCH" --algo "$RAWBOOST_ALGO"
      --train_dataset "$TRAIN_DATASET")
[ -n "$COMMENT" ] && ARGS+=(--comment "$COMMENT")

echo "+ python main.py ${ARGS[*]} $*"
exec "$PY" "$REPO/main.py" "${ARGS[@]}" "$@"
