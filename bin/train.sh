#!/usr/bin/env bash
# Train a model. Flags are passed through to main.py.
#
#   bin/train.sh --model_arch linear_head --ssl_model wavlm_large --num_epochs 50
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PY" "$REPO/main.py" "$@"
