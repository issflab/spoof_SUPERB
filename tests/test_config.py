"""
test_config.py
--------------
Contracts for the configuration layering.

There is ONE settings file -- `configs/paths.yaml` -- and these tests pin the
properties that make that claim true, because the previous arrangement (a code
file plus an env-var-only YAML) was genuinely confusing about which held what.

  G1  configs/paths.yaml is loaded automatically. No env var required.
  G2  A key absent from the YAML falls back to the dataclass default, so a
      partial file is valid and a fresh clone still imports.
  G3  An environment variable beats the YAML -- that is the one-off override.
  G4  SPOOF_SUPERB_CONFIG selects a DIFFERENT file; it never holds settings.
  G5  Derived paths follow their root, so overriding one root moves everything
      built on it.
  G6  Importing config creates nothing on disk.
  G7  An unknown key is reported, not silently ignored or fatal.

Run:  pytest tests/test_config.py
"""

import os
import subprocess
import sys

import pytest

from spoof_superb.config import DEFAULT_CONFIG_FILE, Config, load

REPO_ROOT = DEFAULT_CONFIG_FILE.parent.parent


def _load_with(env):
    """Load a fresh Config in a clean-ish environment."""
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return load()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_g1_default_yaml_is_loaded_without_any_env_var():
    assert DEFAULT_CONFIG_FILE.is_file(), (
        f"{DEFAULT_CONFIG_FILE} is the file users edit; it must ship with the repo"
    )
    cfg = _load_with({"SPOOF_SUPERB_CONFIG": None, "SPOOF_SUPERB_SCORES_ROOT": None})
    assert cfg.config_file == str(DEFAULT_CONFIG_FILE)


def test_g2_missing_key_falls_back_to_the_dataclass_default(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("scores_root: /tmp/only_this\n")
    cfg = _load_with({"SPOOF_SUPERB_CONFIG": str(partial),
                      "SPOOF_SUPERB_DATA_ROOT": None})
    assert cfg.scores_root == "/tmp/only_this"
    assert cfg.data_root == Config().data_root, "absent key must fall back, not blank out"


def test_g3_environment_beats_the_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("scores_root: /tmp/from_yaml\n")
    cfg = _load_with({"SPOOF_SUPERB_CONFIG": str(y),
                      "SPOOF_SUPERB_SCORES_ROOT": "/tmp/from_env"})
    assert cfg.scores_root == "/tmp/from_env"


def test_g4_config_env_var_selects_a_file_and_is_not_itself_a_setting(tmp_path):
    y = tmp_path / "other.yaml"
    y.write_text("scores_root: /tmp/other\n")
    cfg = _load_with({"SPOOF_SUPERB_CONFIG": str(y), "SPOOF_SUPERB_SCORES_ROOT": None})
    assert cfg.config_file == str(y)
    assert cfg.scores_root == "/tmp/other"
    assert "SPOOF_SUPERB_CONFIG" not in {f for f in vars(cfg)}


def test_g5_derived_paths_follow_their_root():
    cfg = _load_with({"SPOOF_SUPERB_SCORES_ROOT": "/tmp/root"})
    assert cfg.reference_dir == "/tmp/root/linear_head"


def test_g6_import_creates_nothing(tmp_path):
    """A fresh interpreter importing config must not touch the filesystem."""
    probe = tmp_path / "probe"
    code = (
        "import os, sys;"
        f"os.environ['SSL_SAVE_DIR'] = {str(probe)!r};"
        "import spoof_superb.config;"
        f"sys.exit(1 if os.path.exists({str(probe)!r}) else 0)"
    )
    env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}")
    rc = subprocess.call([sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env)
    assert rc == 0, "importing spoof_superb.config created save_dir as a side effect"


def test_g7_unknown_key_is_reported_not_fatal(tmp_path, capsys):
    y = tmp_path / "typo.yaml"
    y.write_text("scores_rooot: /tmp/typo\nscores_root: /tmp/good\n")
    cfg = _load_with({"SPOOF_SUPERB_CONFIG": str(y), "SPOOF_SUPERB_SCORES_ROOT": None})
    assert cfg.scores_root == "/tmp/good"
    assert "scores_rooot" in capsys.readouterr().out, (
        "a typo'd key must be reported, or it fails silently"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
