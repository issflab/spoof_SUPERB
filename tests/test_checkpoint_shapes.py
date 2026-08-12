"""Both checkpoint shapes load, and broken ones are still rejected.

``swa.pth`` as written by training carries the whole module, frozen upstream
included. The published checkpoints carry only the five trained tensors. Scoring
must accept both and must not become permissive in the process: a downstream-only
file that is missing a trained tensor, or that carries a tensor the model does
not define, has to fail loudly rather than score with partially random weights.

These tests use a stub module rather than a real s3prl upstream so they stay
offline and fast; the contract under test is the loader's, not the encoder's.
"""
import pytest
import torch
from torch import nn

from spoof_superb.scoring.backends import _TRAINED_KEYS, _load_linear_head_state


class _Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(13))


class _Upstream(nn.Module):
    """Stands in for the frozen s3prl encoder: parameters under ssl_model.model."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)


class _SSL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Upstream()
        self.featurizer = _Featurizer()


class _PostNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


class _Model(nn.Module):
    """Same key layout as UtteranceLevel, small enough to build in a test."""

    def __init__(self):
        super().__init__()
        self.ssl_model = _SSL()
        self.projector = nn.Linear(8, 4)
        self.post_net = _PostNet()


def _full_state():
    m = _Model()
    for i, p in enumerate(m.parameters()):
        torch.nn.init.constant_(p, float(i + 1))
    return m.state_dict()


def _downstream_only(full):
    return {k: v for k, v in full.items() if not k.startswith("ssl_model.model.")}


def test_full_checkpoint_loads(tmp_path):
    full = _full_state()
    p = tmp_path / "swa.pth"
    torch.save(full, p)
    m = _Model()
    assert _load_linear_head_state(m, str(p), "cpu") == "full"
    assert all(torch.equal(m.state_dict()[k], full[k]) for k in full)


def test_downstream_only_loads_and_leaves_upstream_untouched(tmp_path):
    full = _full_state()
    p = tmp_path / "slim.pth"
    torch.save(_downstream_only(full), p)

    m = _Model()
    upstream_before = {k: v.clone() for k, v in m.state_dict().items()
                       if k.startswith("ssl_model.model.")}

    assert _load_linear_head_state(m, str(p), "cpu") == "downstream-only"

    after = m.state_dict()
    # every trained tensor came from the file
    for k in _TRAINED_KEYS:
        assert torch.equal(after[k], full[k]), k
    # and the frozen upstream was not disturbed
    for k, v in upstream_before.items():
        assert torch.equal(after[k], v), k


def test_missing_trained_tensor_is_rejected(tmp_path):
    slim = _downstream_only(_full_state())
    slim.pop("projector.bias")
    p = tmp_path / "incomplete.pth"
    torch.save(slim, p)
    with pytest.raises(RuntimeError, match="missing trained tensors"):
        _load_linear_head_state(_Model(), str(p), "cpu")


def test_unexpected_tensor_is_rejected(tmp_path):
    slim = _downstream_only(_full_state())
    slim["projector.not_a_real_parameter"] = torch.zeros(3)
    p = tmp_path / "extra.pth"
    torch.save(slim, p)
    with pytest.raises(RuntimeError, match="does not define"):
        _load_linear_head_state(_Model(), str(p), "cpu")
