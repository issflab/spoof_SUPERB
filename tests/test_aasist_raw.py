"""
test_aasist_raw.py
------------------
Contracts for the standalone AASIST baseline (aasist_raw_model.Model):

  C1  Forward accepts the benchmark's (B, 64600) crop and returns (B, 2)
      logits -- the shape main.py's train/eval loops index as batch_out[:, 1].
  C2  Parameter count equals the published AASIST model (297,866). This is the
      sharpest available check that the architecture was assembled correctly:
      a wrong filter count, a missing block, or a wrong gat_dim moves it.

      Provenance: this number is not from the paper. It was obtained by
      instantiating the published implementation at
      /home/alhashim/Decision_Boundary_Analysis/AASIST.py with THIS repo's
      configs/AASIST.conf["model_config"] and comparing to aasist_raw_model:
      identical parameter names, identical shapes, identical total. (The
      often-quoted 297,687 belongs to a different pool_ratios variant.)
  C3  The internal node counts match pos_S (23 spectral nodes). A mismatch
      would broadcast silently or crash depending on batch size.
  C4  Gradients flow to the graph back-end and the sinc filterbank stays fixed
      (it is a buffer, not a parameter).
  C5  Building the model does NOT instantiate an s3prl upstream -- this is the
      no-SSL baseline; if an SSL encoder crept in, the param count and the
      "baseline" claim are both void.

Run:  pytest tests/test_aasist_raw.py       (or: python tests/test_aasist_raw.py)
"""

import pytest
import torch

from aasist_raw_model import Model  # noqa: E402

PUBLISHED_NB_PARAMS = 297_866
CROP = 64600


@pytest.fixture
def model():
    """A fresh eval-mode model per test.

    Deliberately function-scoped: C4 runs backward(), and a shared instance
    would let that mutation leak into whichever test happened to run next.
    """
    torch.manual_seed(1234)
    m = Model(args=None, device="cpu").to("cpu")
    m.eval()
    return m


def test_c2_param_count_matches_published_aasist(model):
    nb = sum(p.numel() for p in model.parameters())
    assert nb == PUBLISHED_NB_PARAMS, (
        f"param count = {nb}, expected {PUBLISHED_NB_PARAMS} "
        f"(diff {nb - PUBLISHED_NB_PARAMS:+d})"
    )


def test_c5_no_ssl_upstream_present(model):
    ssl_like = [n for n, mod in model.named_modules()
                if "s3prl" in type(mod).__module__.lower() or n.endswith("ssl_model")]
    assert not ssl_like, f"SSL submodules found in the no-SSL baseline: {ssl_like}"


@pytest.mark.parametrize("bs", [1, 4])
def test_c1_forward_shape(model, bs):
    x = torch.randn(bs, CROP)
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (bs, 2), (
        f"forward({bs}, {CROP}) -> {tuple(out.shape)}, expected ({bs}, 2)"
    )


def test_c3_spectral_node_count_matches_pos_s(model):
    with torch.no_grad():
        x = torch.randn(2, CROP).unsqueeze(1)
        h = model.conv_time(x, mask=False).unsqueeze(1)
        h = torch.nn.functional.max_pool2d(torch.abs(h), (3, 3))
        h = model.selu(model.first_bn(h))
        e = model.encoder(h)
    n_spec = e.shape[2]
    assert n_spec == model.pos_S.shape[1] == 23, (
        f"encoder gives {n_spec} spectral nodes, pos_S has {model.pos_S.shape[1]}"
    )


def test_c4_gradients_reach_the_graph_backend(model):
    model.train()
    x = torch.randn(2, CROP)
    y = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 20, f"only {n_with_grad} parameter tensors received gradient"


def test_c4_sinc_filterbank_is_a_fixed_buffer(model):
    trainable = [n for n, _ in model.named_parameters() if n == "conv_time.band_pass"]
    assert not trainable, "sinc filterbank is a trainable parameter (should be a fixed buffer)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
