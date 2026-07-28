#!/usr/bin/env bash
# Waits out the recurring host CUDA fault, then scores the one remaining
# (aasist_raw, MLAAD v10) cell. Mirrors watch_and_run_spoofceleb.sh.
PY=/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python
cd /home/alhashim/ASD_SUPERB/spoof_SUPERB
OUT=/data/ssl_anti_spoofing/asd_superb_score_files/baselines/aasist_raw/aasist_raw_Multilingual.txt
while true; do
  if [ -s "$OUT" ]; then echo "$(date): output present, done"; break; fi
  if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "$(date): CUDA healthy, launching"
    $PY -u -m spoof_superb.orchestration.driver --job baselines --jobs 1 \
        --only Multilingual >> outputs/logs/phase3_aasist_mlaad.log 2>&1
    if [ -s "$OUT" ]; then echo "$(date): SUCCESS"; break; fi
    echo "$(date): attempt failed, will retry"
  else
    echo "$(date): CUDA down, waiting"
  fi
  sleep 300
done
