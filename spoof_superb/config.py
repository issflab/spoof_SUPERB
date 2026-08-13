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
    SPOOF_SUPERB_BENCH_ROOT  the benchmark's data root
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
    model_arch: Literal['aasist', 'linear_head', 'aasist_raw', 'lfcc_gmm'] = 'aasist'

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

    # The benchmark's own data root -- everything it reads and writes that is
    # not a corpus and not source:
    #
    #     {bench_root}/scores/         raw/, views/, _runs/
    #     {bench_root}/models/         the published detector checkpoints
    #     {bench_root}/analysis/       main_results/, degradation/, tts/, ...
    #     {bench_root}/verification/   analysis/, scores/
    #
    # Only scores/raw/ and models/ come from the published release; everything
    # else is generated locally and can be rebuilt from them.
    #
    # Empty means the repo's own bench/ directory. Point it at a disk with room:
    # the score files alone are about 8 GB.
    #
    # scores_root, analysis_root and verification_root each follow this when
    # left empty, so a fresh clone sets one path rather than four. An explicit
    # value always wins. `models_root` deliberately does NOT follow it: that key
    # means "one directory per model, each holding swa.pth", the training
    # layout, whereas the download is flat `{slug}.pth` files.
    bench_root: str = ''

    # Empty means {bench_root}/scores, so deleting this key -- or leaving it
    # blank -- makes it follow the benchmark root rather than falling back to
    # a path that only exists on the machine this was developed on.
    scores_root: str = ''
    # Empty means {bench_root}/models and {bench_root}/models/baselines.
    # bin/fetch_release.sh writes the downloaded checkpoints in exactly this
    # layout -- one directory per model holding swa.pth -- so a fetched set is
    # usable for scoring and orchestration with no further configuration. Point
    # these at your own training output instead when you have trained models.
    models_root: str = ''
    baseline_models_root: str = ''
    save_dir: str = '/data/ssl_anti_spoofing/asd_superb/'

    # Where the analyses write their CSVs and figures, and where verification
    # writes its reports. Empty means {bench_root}/analysis and
    # {bench_root}/verification.
    #
    # Each analysis still takes --out_dir, and verification --out, which win
    # over these.
    #
    # `outputs_root` is the former name of analysis_root and is still accepted
    # in configs/paths.yaml so existing config files keep working.
    analysis_root: str = ''
    verification_root: str = ''


    linear_head_prefix: str = 'model_weighted_CCE_50_64_linear_head_ASV19_'
    reference_ssl: str = 'xls_r_300m'

    # Interpreter the orchestrators launch scoring subprocesses with. Defaults
    # to the running one: four different absolute interpreter paths were
    # hardcoded across the old launchers, and two of them differed in soxr
    # version -- librosa's resampler.
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

    @property
    def bench_dir(self) -> str:
        """The benchmark's data root. Falls back to the repo's own bench/."""
        from spoof_superb import REPO_ROOT
        return self.bench_root or str(REPO_ROOT / 'bench')

    @property
    def bench_scores_dir(self) -> str:
        """The scores tree. Valid as a ``scores_root``."""
        return os.path.join(self.bench_dir, 'scores')

    @property
    def bench_models_dir(self) -> str:
        """The published checkpoints: flat ``{slug}.pth``, plus ``non_ssl/``.

        Not interchangeable with ``models_root``, which is the training layout
        of one directory per model containing ``swa.pth``.
        """
        return os.path.join(self.bench_dir, 'models')

    @property
    def models_dir(self) -> str:
        """Checkpoint root. Follows bench_root unless set explicitly."""
        return self.models_root or self.bench_models_dir

    @property
    def baselines_dir(self) -> str:
        """aasist_raw and lfcc_gmm checkpoints."""
        return self.baseline_models_root or os.path.join(self.bench_models_dir, 'baselines')

    @property
    def analysis_dir(self) -> str:
        """Where the analyses write. The one place this is decided."""
        return self.analysis_root or os.path.join(self.bench_dir, 'analysis')

    @property
    def verification_dir(self) -> str:
        """Where verification writes. A sibling of analysis, not a child.

        Verification answers a different question from the analyses -- does the
        result still match ``reference/`` -- and a reader looking for that
        answer should not have to know it lives inside the analysis output.
        """
        return self.verification_root or os.path.join(self.bench_dir, 'verification')

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
    'bench_root': 'SPOOF_SUPERB_BENCH_ROOT',
    'analysis_root': 'SPOOF_SUPERB_ANALYSIS_ROOT',
    'verification_root': 'SPOOF_SUPERB_VERIFICATION_ROOT',
    'scores_root': 'SPOOF_SUPERB_SCORES_ROOT',
    'models_root': 'SPOOF_SUPERB_MODELS_ROOT',
    'baseline_models_root': 'SPOOF_SUPERB_BASELINE_MODELS_ROOT',
    'python': 'SPOOF_SUPERB_PYTHON',
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
    # Former names, still accepted so existing config files keep working.
    aliases = {"outputs_root": "analysis_root", "release_root": "bench_root"}
    for key, value in data.items():
        key = aliases.get(key, key)
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

    # An empty scores_root means "under the benchmark root", so a fresh clone
    # needs one setting instead of several. An explicit value always wins,
    # which keeps every existing config working unchanged.
    if not config.scores_root:
        config.scores_root = config.bench_scores_dir
    if not config.models_root:
        config.models_root = config.bench_models_dir
    if not config.baseline_models_root:
        config.baseline_models_root = config.baselines_dir

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
