#!/usr/bin/env bash
# ============================================================================
# Compare a freshly produced score file against its published reference.
# Exit 0 = pass, 1 = fail.
#
# Edit the settings block below, then run:   bin/verify.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Grade policy. These are NOT interchangeable:
#   mlaad / mailabs  Pearson >= 0.99 AND Spearman >= 0.99 AND sign@0 >= 0.999,
#                    tolerating up to 1% NaN in your output
#   spoofceleb       Spearman >= 0.99 alone, tolerating no NaN at all
CHECK="spoofceleb"

SSL_MODEL="xls_r_300m"

# The file you just produced, and the reference to compare it against.
NEW_FILE="$SCORES_ROOT/linear_head_SpoofCeleb/linear_head_SpoofCeleb_${SSL_MODEL}.txt"
REF_FILE="$SCORES_ROOT/linear_head/linear_head_spoofceleb_${SSL_MODEL}.txt"

# ------------------------------------------------------------ END SETTINGS --

echo "+ verify --check $CHECK --new $NEW_FILE --ref $REF_FILE $*"
exec "$PY" -m spoof_superb.verification.driver \
    --check "$CHECK" --new "$NEW_FILE" --ref "$REF_FILE" "$@"
