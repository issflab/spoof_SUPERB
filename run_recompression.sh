#!/bin/bash
# run_recompression.sh GPU MODEL [MODEL...]
# Scores the full Recompression protocol (427,422 utts) for each model on one GPU.
# No reference exists for Recompression -> generate-only, no --restrict_to.
set -u
GPU=$1; shift

PY=/home/alhashim/.conda/envs/ASD_SUPERB/bin/python
REPO=/home/alhashim/ASD_SUPERB/spoof_SUPERB
# `python -m spoof_superb...` needs the package importable; exporting
# PYTHONPATH rather than cd-ing keeps every relative path in this script
# resolving exactly as before.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PROTO=/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD/protocols
AUDIO=/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD
MODELS=/data/ssl_anti_spoofing/asd_superb_models/linear_head_models
OUT=/data/ssl_anti_spoofing/asd_superb_score_files/asvld_rerun/Recompression

for M in "$@"; do
  CK=$MODELS/model_weighted_CCE_50_64_linear_head_ASV19_${M}/swa.pth
  OF=$OUT/linear_head_Recompression_${M}.txt
  echo "=== $(date '+%F %T') START $M (gpu $GPU) ==="
  $PY -m spoof_superb.scoring.driver \
      --model linear_head --source asvld \
      --model_path "$CK" --ssl_model "$M" --asvld_condition Recompression \
      --output_file "$OF" --protocols_dir "$PROTO" --audio_base_dir "$AUDIO" \
      --cuda_device "cuda:$GPU"
  echo "=== $(date '+%F %T') END   $M rc=$? ==="
done
echo "########## GROUP DONE gpu$GPU $(date '+%F %T') ##########"
