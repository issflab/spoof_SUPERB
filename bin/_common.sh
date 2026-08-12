# ============================================================================
# Shared preamble for the bin/ scripts. Sourced, not executed.
#
# You should not need to edit this file. Settings live in configs/paths.yaml.
#
# It does three things:
#   1. puts the repo on PYTHONPATH, so `python -m spoof_superb...` resolves
#      from any working directory (without cd-ing, which would change how
#      relative paths in your arguments resolve)
#   2. picks the interpreter
#   3. exports the configured roots as shell variables, read from the SAME
#      configs/paths.yaml the Python code uses -- so the shell and the code
#      can never disagree about where the data is
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

PY="${SPOOF_SUPERB_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || {
    echo "bin: interpreter '$PY' not found. Activate the environment," >&2
    echo "     or set SPOOF_SUPERB_PYTHON=/path/to/python." >&2
    exit 127
}

# Read the resolved config once and expose it to the calling script.
eval "$("$PY" - <<'PYEOF'
from spoof_superb.config import cfg
for name in ("data_root", "scores_root", "models_root", "baseline_models_root",
             "save_dir", "linear_head_prefix", "reference_ssl",
             "release_dir", "release_scores_dir", "release_models_dir"):
    print(f'{name.upper()}="{getattr(cfg, name)}"')
PYEOF
)"
