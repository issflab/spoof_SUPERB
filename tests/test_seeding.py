"""
test_seeding.py
---------------
Contracts for `core.utils.set_seed`, which replaced
`core_scripts.startup_config.set_random_seed` when the 38-file vendored copy of
project-NN-Pytorch-scripts was removed. One function of that copy was reachable;
this is it.

Seeding is the kind of code that breaks silently -- a run still completes, the
loss still falls, and nobody notices the numbers stopped being reproducible
until they try to reproduce them. So the contracts are about what a caller can
rely on, not about the implementation:

  S1  The same seed produces the same draws from torch, numpy AND random. All
      three, because the training path uses all three and seeding two of them
      is indistinguishable from seeding three until it matters.
  S2  Different seeds produce different draws -- the guard that makes S1 mean
      something rather than being satisfied by a no-op.
  S3  `config` may be an argparse Namespace, a mapping, or None, and the three
      are NOT interchangeable in the obvious way. This is the bug that kept two
      seeding functions alive in the repo: `str_to_bool` calls `.lower()`, so
      feeding it argparse's real booleans raises AttributeError. Only strings
      are converted.
  S4  `config=None` sets the safe cuDNN pair rather than raising. The local
      implementation used to raise, which is why nothing called it.
  S5  PYTHONHASHSEED is set, and the docstring's caveat is true: it cannot
      affect the running process. The test pins the observable part.
  S6  The cuDNN notices go to stdout only when a config was supplied. They are
      preserved from the vendored version -- and they are why a probe of the
      old function returned unparseable JSON.

Run:  pytest tests/test_seeding.py
"""

import argparse
import os
import random

import numpy as np
import pytest
import torch

from spoof_superb.core.utils import _toggle, set_seed, str_to_bool


def draws():
    """One sample from each RNG the training path touches."""
    return (torch.randn(3).tolist(),
            np.random.rand(3).tolist(),
            [random.random() for _ in range(3)])


DEFAULTS = argparse.Namespace(cudnn_deterministic_toggle=True,
                              cudnn_benchmark_toggle=False)


# --- S1 / S2: the point of the function ------------------------------------

def test_s1_same_seed_reproduces_all_three_generators():
    set_seed(1234, DEFAULTS)
    first = draws()
    set_seed(1234, DEFAULTS)
    assert draws() == first


def test_s2_different_seeds_diverge():
    """Without this, S1 would pass on a function that did nothing at all."""
    set_seed(1234, DEFAULTS)
    a = draws()
    set_seed(4321, DEFAULTS)
    b = draws()
    for x, y in zip(a, b):
        assert x != y


def test_s1b_seeding_is_independent_of_the_config_shape():
    """A Namespace and the equivalent mapping must seed identically."""
    set_seed(99, DEFAULTS)
    from_namespace = draws()
    set_seed(99, {"cudnn_deterministic_toggle": "true",
                  "cudnn_benchmark_toggle": "false"})
    assert draws() == from_namespace


# --- S3: the incompatibility that kept two implementations alive ------------

def test_s3_argparse_booleans_are_not_passed_through_str_to_bool():
    """The exact defect: str_to_bool('.lower()') raises on a real bool.

    The vendored function read attributes and used the value directly; the
    local one subscripted a dict through str_to_bool. Neither could accept the
    other's argument, so both survived. `_toggle` converts strings only.
    """
    with pytest.raises(AttributeError):
        str_to_bool(True)

    assert _toggle(argparse.Namespace(flag=True), "flag", False) is True
    assert _toggle(argparse.Namespace(flag=False), "flag", True) is False


def test_s3b_mapping_strings_are_converted():
    assert _toggle({"flag": "false"}, "flag", True) is False
    assert _toggle({"flag": "yes"}, "flag", False) is True


def test_s3c_an_absent_toggle_falls_back_to_the_default():
    assert _toggle(argparse.Namespace(), "flag", True) is True
    assert _toggle({}, "flag", False) is False


# --- S4: None is a valid config, not an error -------------------------------

def test_s4_none_config_is_accepted_and_picks_the_safe_pair():
    """It used to raise ValueError, which is why nothing called it."""
    set_seed(7, None)
    if torch.cuda.is_available():
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


def test_s4b_toggles_reach_the_backend():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: the cuDNN flags are not written without it")
    set_seed(7, argparse.Namespace(cudnn_deterministic_toggle=False,
                                   cudnn_benchmark_toggle=True))
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    set_seed(7, DEFAULTS)          # leave the process as we found it
    assert torch.backends.cudnn.deterministic is True


# --- S5 / S6: the two observable side effects -------------------------------

def test_s5_pythonhashseed_is_exported():
    os.environ.pop("PYTHONHASHSEED", None)
    set_seed(31337, DEFAULTS)
    assert os.environ["PYTHONHASHSEED"] == "31337"


def test_s6_notices_are_printed_only_when_a_config_asks_for_them(capsys):
    set_seed(1, DEFAULTS)
    assert capsys.readouterr().out == "", (
        "the default pair is not worth a line of output"
    )

    set_seed(1, argparse.Namespace(cudnn_deterministic_toggle=False,
                                   cudnn_benchmark_toggle=True))
    out = capsys.readouterr().out
    assert "cudnn_deterministic set to False" in out
    assert "cudnn_benchmark set to True" in out

    set_seed(1, None)
    assert capsys.readouterr().out == "", "None configures nothing, so it says nothing"
    set_seed(1, DEFAULTS)


def test_the_vendored_module_is_gone():
    """core_scripts was 38 files; one was reachable and it is reimplemented here."""
    with pytest.raises(ImportError):
        import core_scripts.startup_config  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
