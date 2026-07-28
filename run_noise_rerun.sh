#!/bin/bash
# run_noise_rerun.sh GPU MODEL [MODEL...]
# Re-scores the Noise_Addition condition in fp32 for each model on one GPU.
#
# Why: the archived Noise_Addition scores were produced in half precision and
# 53.93% of them overflowed to NaN (identical utterance set across the four
# s3prl transformer-family upstreams). eval_asvld.py runs fp32 and does not
# overflow -- verified on a 2001-utt pilot: 1101 archived NaN -> 0 NaN.
#
# Scope is held identical to the archive: --restrict_to the existing score file
# pins the exact 712,370-utterance subset (SNR 10 and 20; SNR 0 is not in the
# archive) and its ordering, so the new file is a drop-in replacement.
#
# Env: miniconda3/envs/spoof_SUPERB -- same torch 2.7.1+cu126 and s3prl 0.4.18
# as ASD_SUPERB, plus timm and espnet (needed by ssast_frame_base / wavlablm).
# Verified bit-identical to ASD_SUPERB on the pilot (max |delta| = 0.0).
#
# CUDA_VISIBLE_DEVICES pins the GPU and --cuda_device stays cuda:0, so upstreams
# that hardcode device 0 land on the intended physical GPU.
set -u
GPU=$1; shift

PY=/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python
REPO=/home/alhashim/ASD_SUPERB/spoof_SUPERB
# `python -m spoof_superb...` needs the package importable; exporting
# PYTHONPATH rather than cd-ing keeps every relative path in this script
# resolving exactly as before.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PROTO=/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD/protocols
AUDIO=/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD
MODELS=/data/ssl_anti_spoofing/asd_superb_models/linear_head_models
ROOT=/data/ssl_anti_spoofing/asd_superb_score_files/asvld_rerun
OLD=$ROOT/Noise_Addition
NEW=$ROOT/Noise_Addition_new

mkdir -p "$NEW"

for M in "$@"; do
  CK=$MODELS/model_weighted_CCE_50_64_linear_head_ASV19_${M}/swa.pth
  REF=$OLD/linear_head_Noise_Addition_${M}.txt
  OF=$NEW/linear_head_Noise_Addition_${M}.txt

  if [ ! -f "$CK" ];  then echo "[WARN] no checkpoint for $M ($CK) -- skipping"; continue; fi
  if [ ! -f "$REF" ]; then echo "[WARN] no archived score file for $M -- skipping"; continue; fi

  echo "=== $(date '+%F %T') START $M (gpu $GPU) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY -m spoof_superb.scoring.driver \
      --model linear_head --source asvld \
      --model_path "$CK" --ssl_model "$M" --asvld_condition Noise_Addition \
      --output_file "$OF" --protocols_dir "$PROTO" --audio_base_dir "$AUDIO" \
      --cuda_device cuda:0 --restrict_to "$REF" \
      --batch_size 32 --num_workers 8
  rc=$?
  echo "=== $(date '+%F %T') END   $M rc=$rc ==="
  if [ $rc -ne 0 ]; then echo "!!! $M FAILED rc=$rc -- continuing to next model"; fi
done
echo "########## GROUP DONE gpu$GPU $(date '+%F %T') ##########"
