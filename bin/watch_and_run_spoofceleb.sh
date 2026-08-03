#!/usr/bin/env bash
# watch_and_run_spoofceleb.sh
# ---------------------------
# HISTORICAL, one-off. This ran once, against the LEGACY tree, and its canary
# check calls spoof_superb.verification.driver -- a module nothing else uses any
# more. Do not copy this pattern: scoring no longer compares itself against
# anything, and verification is a separate step over a finished tree
# (`bin/verify.sh`). Kept as a record of how the SpoofCeleb batch was launched.
#
# The SpoofCeleb batch is blocked only by a host-wide CUDA driver fault
# (cuInit -> error 3) that needs root to clear; see humanpending.md. Rather than
# lose the run to the wait, poll until CUDA comes back and then launch the full
# batch unattended: canary (xls_r_300m) first, and only if the canary passes
# verification do the remaining 23 models go out across the 3 GPUs.
#
# Kill with: pkill -f watch_and_run_spoofceleb
set -u

PY=/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python
REPO=/home/alhashim/ASD_SUPERB/spoof_SUPERB
OUT=/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_SpoofCeleb
REF=/data/ssl_anti_spoofing/asd_superb_score_files/linear_head
WATCHLOG="$OUT/watcher.log"
MAX_WAIT_S=$((24 * 3600))   # give up after 24h rather than poll forever

mkdir -p "$OUT/logs"
exec >>"$WATCHLOG" 2>&1
echo "=== watcher started $(date) (pid $$) ==="

waited=0
until "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; do
    if [ "$waited" -ge "$MAX_WAIT_S" ]; then
        echo "$(date): CUDA still down after ${MAX_WAIT_S}s; giving up. See humanpending.md."
        exit 1
    fi
    sleep 60
    waited=$((waited + 60))
    [ $((waited % 1800)) -eq 0 ] && echo "$(date): still waiting for CUDA (${waited}s)"
done

echo "$(date): CUDA is available after ${waited}s -> starting canary (xls_r_300m)"
cd "$REPO" || exit 1
"$PY" -m spoof_superb.orchestration.driver --job all --datasets spoofceleb --models xls_r_300m

canary="$OUT/linear_head_SpoofCeleb_xls_r_300m.txt"
lines=$( [ -f "$canary" ] && wc -l < "$canary" || echo 0 )
if [ "$lines" -ne 91130 ]; then
    echo "$(date): CANARY FAILED - $lines lines (expected 91130). Not batching a broken pipeline."
    exit 1
fi
if ! "$PY" -m spoof_superb.verification.driver --check spoofceleb --new "$canary" \
        --ref "$REF/linear_head_spoofceleb_xls_r_300m.txt"; then
    echo "$(date): CANARY FAILED verification (spearman < 0.99). Stopping; see log above."
    exit 1
fi

echo "$(date): canary PASSED ($lines lines) -> launching remaining 23 models on 3 GPUs"
"$PY" -m spoof_superb.orchestration.driver --job all --datasets spoofceleb
echo "$(date): batch complete. Summary: $OUT/SUMMARY.txt"
