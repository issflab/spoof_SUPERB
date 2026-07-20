#!/bin/bash
# run_asvld_model.sh MODEL GPU
# Runs all 5 ASVLD conditions for one model on one GPU (sequentially within the model).
# Verifiable conditions use --restrict_to the matching reference (reference-subset scoring);
# Recompression & Filtering have no reference -> full protocol.
set -u
MODEL=$1
GPU=$2

PY=/home/alhashim/.conda/envs/ASD_SUPERB/bin/python
REPO=/home/alhashim/ASD_SUPERB/spoof_SUPERB
PROTO=/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD/protocols
AUDIO=/data/Data/ASVSpoofLaunderedDatabase/ASVspoofLD
CKPT=/data/ssl_anti_spoofing/asd_superb_models/linear_head_models/model_weighted_CCE_50_64_linear_head_ASV19_${MODEL}/swa.pth
OUT=/data/ssl_anti_spoofing/asd_superb_score_files/asvld_rerun
REF=/data/ssl_anti_spoofing/asd_superb_score_files/scores_by_category

case $MODEL in
  xls_r_300m)          RN="XLS-R.txt";         RR="linear_head_resamp_xls_r_300m.txt";;
  unispeech_sat_large) RN="Unispeech-SAT.txt"; RR="linear_head_resamp_unispeech_sat_large.txt";;
  wav2vec2_large_ll60k)RN="wav2vec2_Large.txt";RR="linear_head_resamp_wav2vec2_large_ll60k.txt";;
  *) echo "unknown model $MODEL"; exit 2;;
esac

run() {  # COND  RESTRICT(optional)
  local COND=$1; local RESTRICT=${2:-}
  local OF=$OUT/$COND/linear_head_${COND}_${MODEL}.txt
  echo "=== $(date '+%F %T') START $MODEL / $COND (gpu $GPU) ==="
  local args=(--model_path "$CKPT" --ssl_model "$MODEL" --condition "$COND"
              --output_file "$OF" --protocols_dir "$PROTO" --audio_base_dir "$AUDIO"
              --cuda_device "cuda:$GPU")
  [ -n "$RESTRICT" ] && args+=(--restrict_to "$RESTRICT")
  $PY "$REPO/eval_asvld.py" "${args[@]}"
  echo "=== $(date '+%F %T') END   $MODEL / $COND rc=$? ==="
}

# Verifiable first (reference-subset), smallest -> largest; then unverifiable full protocols.
run Resampling     "$REF/Resampling/$RR"
run Reverberation  "$REF/Reverberation/$RN"
run Noise_Addition "$REF/Additive_Noise/$RN"
run Recompression
run Filtering
echo "########## ALL DONE $MODEL $(date '+%F %T') ##########"
