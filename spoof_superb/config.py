"""Configuration for Spoof-SUPERB.

Before the reorganisation this file was imported by 2 of 47 modules; every
other script carried its own hardcoded absolute paths. It is now the single
place the corpus roots, model roots and score roots are declared, and the
modules that used to hardcode them read from here.

Resolution order, lowest priority first:

    1. the dataclass defaults below
    2. a YAML file, if SPOOF_SUPERB_CONFIG points at one
    3. environment variables (SSL_*, CUDA_DEVICE, SPOOF_SUPERB_*)
    4. whatever the CLI of the tool you are running sets

Import has NO side effects. The previous version called ``prepare_dirs()`` at
import time, so merely importing the module created directories on /data;
callers that need them now ask.

    from spoof_superb.config import cfg
    print(cfg.scores_root)

Environment variables
---------------------
    SSL_MODEL_ARCH  SSL_SAVE_DIR  SSL_DATABASE_PATH  SSL_PROTOCOLS_PATH
    SSL_MODE  SSL_MODEL_NAME  SSL_DATASET  SSL_PRETRAINED_CHECKPOINT
    CUDA_DEVICE
    SPOOF_SUPERB_CONFIG      path to a YAML file with any of these keys
    SPOOF_SUPERB_DATA_ROOT   SPOOF_SUPERB_SCORES_ROOT
    SPOOF_SUPERB_MODELS_ROOT SPOOF_SUPERB_BASELINE_MODELS_ROOT
    SPOOF_SUPERB_PYTHON      interpreter the orchestrators launch
"""

import os
import sys
from dataclasses import dataclass, fields
from typing import Literal, Optional

__all__ = ["Config", "cfg", "load"]


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

    linear_head_prefix: str = 'model_weighted_CCE_50_64_linear_head_ASV19_'
    reference_ssl: str = 'xls_r_300m'

    # Interpreter the orchestrators launch scoring subprocesses with. Defaults
    # to the running one: four different absolute interpreter paths were
    # hardcoded across the old launchers, and two of them differed in soxr
    # version -- librosa's resampler. See humanpending.md.
    python: str = sys.executable

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
}


def _apply_yaml(config, path):
    try:
        import yaml
    except ImportError:
        print(f"[config] SPOOF_SUPERB_CONFIG={path} ignored: pyyaml not installed")
        return
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        print(f"[config] cannot read SPOOF_SUPERB_CONFIG={path}: {exc}")
        return

    known = {f.name for f in fields(config)}
    for key, value in data.items():
        if key in known:
            setattr(config, key, value)
        else:
            print(f"[config] ignoring unknown key {key!r} in {path}")


def load() -> Config:
    """Build a Config from defaults, then YAML, then environment."""
    config = Config()

    yaml_path = os.getenv('SPOOF_SUPERB_CONFIG')
    if yaml_path:
        _apply_yaml(config, yaml_path)

    for field_name, env_name in _ENV_MAP.items():
        value = os.getenv(env_name)
        if value:
            setattr(config, field_name, value)

    return config


cfg = load()
