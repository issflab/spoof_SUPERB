"""Configuration schema and loader.

THIS FILE IS CODE, NOT YOUR SETTINGS FILE.

To point the repo at your data, edit **configs/paths.yaml**. That is the only
file you need to touch, and it is loaded automatically. This module defines
what the settings *are* (their names, types and fallback values) and how they
are resolved; the values you actually use live in the YAML.

    configs/paths.yaml   <- edit this
    spoof_superb/config.py  <- this file: the schema and the loader

Resolution order, lowest priority first:

    1. the dataclass defaults below (fallbacks, so a fresh clone still imports)
    2. configs/paths.yaml, loaded automatically if it exists
    3. environment variables, for one-off overrides in a single command
    4. the CLI flags of whatever tool you are running

So `SPOOF_SUPERB_SCORES_ROOT=/tmp/x python -m ...` overrides the YAML for that
one command without editing anything. The env var SPOOF_SUPERB_CONFIG does NOT
hold settings -- it points at a *different* YAML file, for when you keep
several (e.g. one per machine).

Import has NO side effects. The previous version called ``prepare_dirs()`` at
import time, so merely importing the module created directories on /data;
callers that need them now ask.

    from spoof_superb.config import cfg
    print(cfg.scores_root)

Environment variables
---------------------
    SPOOF_SUPERB_CONFIG      use a different YAML instead of configs/paths.yaml
    SPOOF_SUPERB_DATA_ROOT   SPOOF_SUPERB_SCORES_ROOT
    SPOOF_SUPERB_MODELS_ROOT SPOOF_SUPERB_BASELINE_MODELS_ROOT
    SPOOF_SUPERB_PYTHON      interpreter the orchestrators launch
    SSL_MODEL_ARCH  SSL_SAVE_DIR  SSL_DATABASE_PATH  SSL_PROTOCOLS_PATH
    SSL_MODE  SSL_MODEL_NAME  SSL_DATASET  SSL_PRETRAINED_CHECKPOINT
    CUDA_DEVICE
"""

import os
import sys
from dataclasses import dataclass, fields
from typing import Literal, Optional

from spoof_superb import CONFIGS_DIR

__all__ = ["Config", "cfg", "load", "DEFAULT_CONFIG_FILE"]

#: Loaded automatically when present. This is the file users edit.
DEFAULT_CONFIG_FILE = CONFIGS_DIR / "paths.yaml"


@dataclass
class Config:
    # ---- model / run identity -------------------------------------------
    # 'aasist_raw' and 'lfcc_gmm' are the non-SSL reference baselines: they
    # take no s3prl upstream and ignore --ssl_model.
    model_arch: Literal['aasist', 'sls', 'linear_head', 'aasist_raw', 'lfcc_gmm'] = 'aasist'

    # Naming only: which datasets a model was TRAINED on, used to build the
    # checkpoint tag. e.g. 'ASV19', 'mlaad_spoofceleb_FF'.
    dataset: str = 'ASV19'

    mode: Literal['train', 'eval'] = 'train'
    model_name: str = 'run1'
    cuda_device: str = 'cuda:0'
    pretrained_checkpoint: Optional[str] = None

    # ---- training corpus -------------------------------------------------
    database_path: str = '/data/Data/ASVSpoofData_2019/train/LA/'
    protocols_path: str = '/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/'
    train_protocol: str = 'ASVspoof2019.LA.cm.train.trn.txt'
    dev_protocol: str = 'ASVspoof2019.LA.cm.dev.trl.txt'

    # ---- shared roots ----------------------------------------------------
    # These were hardcoded across ~20 modules before the reorganisation.
    data_root: str = '/data/Data'
    scores_root: str = '/data/ssl_anti_spoofing/asd_superb_score_files'
    models_root: str = '/data/ssl_anti_spoofing/asd_superb_models/linear_head_models'
    baseline_models_root: str = '/data/ssl_anti_spoofing/asd_superb_models/baselines'
    save_dir: str = '/data/ssl_anti_spoofing/asd_superb/'

    # Score-file directory layout. 'legacy' reproduces the pre-reorg paths;
    # 'v2' is raw/{system}/{dataset}/{frontend}.txt. Default is legacy so an
    # existing tree keeps working; set v2 in configs/paths.yaml for a new one.
    score_layout: Literal['legacy', 'v2'] = 'legacy'

    linear_head_prefix: str = 'model_weighted_CCE_50_64_linear_head_ASV19_'
    reference_ssl: str = 'xls_r_300m'

    # Interpreter the orchestrators launch scoring subprocesses with. Defaults
    # to the running one: four different absolute interpreter paths were
    # hardcoded across the old launchers, and two of them differed in soxr
    # version -- librosa's resampler. See humanpending.md.
    python: str = sys.executable

    # Which YAML actually supplied the values above. Set by load(); not a setting.
    config_file: Optional[str] = None

    # ---- derived ---------------------------------------------------------
    @property
    def train_protocol_path(self) -> str:
        return os.path.join(self.protocols_path, self.train_protocol)

    @property
    def dev_protocol_path(self) -> str:
        return os.path.join(self.protocols_path, self.dev_protocol)

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.save_dir, self.model_name)

    @property
    def reference_dir(self) -> str:
        return os.path.join(self.scores_root, 'linear_head')

    def prepare_dirs(self):
        """Create the output directories. Call explicitly; never on import."""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)


# ---------------------------------------------------------------------------
# Layered resolution
# ---------------------------------------------------------------------------

_ENV_MAP = {
    'model_arch': 'SSL_MODEL_ARCH',
    'dataset': 'SSL_DATASET',
    'mode': 'SSL_MODE',
    'model_name': 'SSL_MODEL_NAME',
    'cuda_device': 'CUDA_DEVICE',
    'pretrained_checkpoint': 'SSL_PRETRAINED_CHECKPOINT',
    'database_path': 'SSL_DATABASE_PATH',
    'protocols_path': 'SSL_PROTOCOLS_PATH',
    'save_dir': 'SSL_SAVE_DIR',
    'data_root': 'SPOOF_SUPERB_DATA_ROOT',
    'scores_root': 'SPOOF_SUPERB_SCORES_ROOT',
    'models_root': 'SPOOF_SUPERB_MODELS_ROOT',
    'baseline_models_root': 'SPOOF_SUPERB_BASELINE_MODELS_ROOT',
    'python': 'SPOOF_SUPERB_PYTHON',
    'score_layout': 'SPOOF_SUPERB_SCORE_LAYOUT',
}


def _apply_yaml(config, path):
    try:
        import yaml
    except ImportError:
        print(f"[config] {path} ignored: pyyaml not installed")
        return
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        print(f"[config] cannot read {path}: {exc}")
        return

    known = {f.name for f in fields(config)}
    for key, value in data.items():
        if key in known:
            setattr(config, key, value)
        else:
            print(f"[config] ignoring unknown key {key!r} in {path}")


def config_file_in_use():
    """Which YAML will be loaded, or None if the built-in defaults are all there is."""
    override = os.getenv('SPOOF_SUPERB_CONFIG')
    if override:
        return override
    if DEFAULT_CONFIG_FILE.is_file():
        return str(DEFAULT_CONFIG_FILE)
    return None


def load() -> Config:
    """Build a Config from defaults, then configs/paths.yaml, then environment."""
    config = Config()

    yaml_path = config_file_in_use()
    if yaml_path:
        _apply_yaml(config, yaml_path)
    config.config_file = yaml_path

    for field_name, env_name in _ENV_MAP.items():
        value = os.getenv(env_name)
        if value:
            setattr(config, field_name, value)

    return config


def describe() -> str:
    """Human-readable dump of the resolved settings and where they came from."""
    c = cfg
    src = c.config_file or "(built-in defaults only -- configs/paths.yaml not found)"
    lines = [f"config file : {src}", ""]
    for f in fields(c):
        if f.name == "config_file":
            continue
        env = _ENV_MAP.get(f.name)
        overridden = " [env]" if env and os.getenv(env) else ""
        lines.append(f"  {f.name:24s} = {getattr(c, f.name)}{overridden}")
    return "\n".join(lines)


cfg = load()


if __name__ == "__main__":
    print(describe())
