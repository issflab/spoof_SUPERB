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

Run:  python tests/test_aasist_raw.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aasist_raw_model import Model  # noqa: E402

PUBLISHED_NB_PARAMS = 297_866
CROP = 64600


def main():
    failures = []
    device = "cpu"
    torch.manual_seed(1234)

    model = Model(args=None, device=device).to(device)
    model.eval()

    # C2 -- parameter count
    nb = sum(p.numel() for p in model.parameters())
    if nb == PUBLISHED_NB_PARAMS:
        print(f"ok   C2 param count = {nb} (matches published AASIST)")
    else:
        print(f"FAIL C2 param count = {nb}, expected {PUBLISHED_NB_PARAMS} "
              f"(diff {nb - PUBLISHED_NB_PARAMS:+d})")
        failures.append("C2")

    # C5 -- no SSL upstream anywhere in the module tree
    ssl_like = [n for n, _ in model.named_modules()
                if "s3prl" in type(_).__module__.lower() or n.endswith("ssl_model")]
    if not ssl_like:
        print("ok   C5 no s3prl/SSL submodule present")
    else:
        print(f"FAIL C5 SSL submodules found: {ssl_like}")
        failures.append("C5")

    # C1 -- forward shape
    for bs in (1, 4):
        x = torch.randn(bs, CROP)
        with torch.no_grad():
            out = model(x)
        if tuple(out.shape) == (bs, 2):
            print(f"ok   C1 forward({bs}, {CROP}) -> {tuple(out.shape)}")
        else:
            print(f"FAIL C1 forward({bs}, {CROP}) -> {tuple(out.shape)}, expected ({bs}, 2)")
            failures.append("C1")

    # C3 -- spectral node count matches pos_S
    with torch.no_grad():
        x = torch.randn(2, CROP).unsqueeze(1)
        h = model.conv_time(x, mask=False).unsqueeze(1)
        h = torch.nn.functional.max_pool2d(torch.abs(h), (3, 3))
        h = model.selu(model.first_bn(h))
        e = model.encoder(h)
    n_spec = e.shape[2]
    if n_spec == model.pos_S.shape[1] == 23:
        print(f"ok   C3 encoder output {tuple(e.shape)} -> {n_spec} spectral nodes == pos_S")
    else:
        print(f"FAIL C3 encoder gives {n_spec} spectral nodes, pos_S has {model.pos_S.shape[1]}")
        failures.append("C3")

    # C4 -- gradients flow; sinc filterbank is fixed
    model.train()
    x = torch.randn(2, CROP)
    y = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    if n_with_grad > 20:
        print(f"ok   C4 gradients reach {n_with_grad} parameter tensors")
    else:
        print(f"FAIL C4 only {n_with_grad} parameter tensors received gradient")
        failures.append("C4")

    if any(n == "conv_time.band_pass" for n, _ in model.named_parameters()):
        print("FAIL C4 sinc filterbank is a trainable parameter (should be a fixed buffer)")
        failures.append("C4")
    else:
        print("ok   C4 sinc filterbank is a fixed buffer, not trainable")

    if failures:
        print(f"\nFAILED: {sorted(set(failures))}")
        return 1
    print("\nPASS: all aasist_raw contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
