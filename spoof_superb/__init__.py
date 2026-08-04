"""Spoof-SUPERB: a SUPERB-style benchmark of SSL speech models for audio deepfake detection."""

from pathlib import Path

# Single anchor for every repo-relative asset (configs/, outputs/, ...).
#
# Modules must resolve repo assets through this rather than through their own
# `__file__`: a module's depth inside the package is an implementation detail,
# and deriving paths from it silently relocates assets whenever a file moves.
REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = REPO_ROOT / "configs"
OUTPUTS_DIR = REPO_ROOT / "outputs"

__all__ = ["REPO_ROOT", "CONFIGS_DIR", "OUTPUTS_DIR"]
