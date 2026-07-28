#!/usr/bin/env bash
# Score one model on one set.  All flags are passed through to the driver.
#
#   bin/score.sh --model linear_head --ssl_model xls_r_300m \
#       --model_path .../swa.pth --dataset wild --output_file out.txt
#   bin/score.sh --list_datasets
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PY" -m spoof_superb.scoring.driver "$@"
