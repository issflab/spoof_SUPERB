#!/usr/bin/env bash
# Verify a freshly produced score file against its reference.
#
#   bin/verify.sh --check spoofceleb --new out.txt --ref reference.txt
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PY" -m spoof_superb.verification.driver "$@"
