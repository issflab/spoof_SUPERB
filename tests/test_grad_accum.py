"""
test_grad_accum.py
------------------
Contracts for the gradient-accumulation change in main.py::train_epoch.

That change touches code shared with the SSL architectures, so it must be
provably behaviour-preserving:

  A1  accum_steps=1 reproduces the ORIGINAL loop (zero_grad; backward; step)
      exactly -- same final weights. This is what every SSL run would execute,
      since --micro_batch defaults to 0.
  A2  accum_steps=4 over micro-batches of 16 produces the same WEIGHTS as a
      single un-accumulated pass at batch 64. This is the claim that lets
      aasist_raw be described as trained with the SSL recipe's batch size of
      64 rather than at a quietly reduced batch.

      Only the weights are asserted, not the returned running_loss. That value
      is a diagnostic accumulated as (per-batch mean loss x batch size) / total,
      which is a mean-of-means and therefore depends on how samples are grouped
      into batches -- it is batch-size-dependent in the ORIGINAL loop too, so
      requiring it to be invariant would be asserting something false.

  A2-guard  A naive micro-batch WITHOUT accumulation must NOT match batch 64.
      Without this, A2 could pass vacuously.

A toy linear model is used deliberately: the contract under test is the
optimizer-stepping logic in train_epoch, not any particular architecture.

Run:  pytest tests/test_grad_accum.py       (or: python tests/test_grad_accum.py)
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from main import train_epoch  # noqa: E402

N = 256
DIM = 32


class ToyDS(Dataset):
    def __init__(self, n=N, dim=DIM):
        g = torch.Generator().manual_seed(7)
        self.x = torch.randn(n, dim, generator=g)
        self.y = (torch.rand(n, generator=g) > 0.5).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], f"utt{i}", self.y[i]


def make_model():
    torch.manual_seed(1234)
    return nn.Sequential(nn.Linear(DIM, 16), nn.SELU(), nn.Linear(16, 2))


def original_train_epoch(loader, model, optimizer, device):
    """The pre-change loop, copied verbatim from main.py before accumulation."""
    running_loss, num_total = 0, 0.0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, utt_id, batch_y in loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    return running_loss / num_total


def weights(m):
    return [p.detach().clone() for p in m.parameters()]


def max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def run_case(batch, accum, use_original=False):
    ds = ToyDS()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=True)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if use_original:
        loss = original_train_epoch(loader, model, opt, "cpu")
    else:
        loss = train_epoch(loader, model, opt, "cpu", accum_steps=accum)
    return weights(model), loss


def test_a1_accum_one_is_identical_to_original_loop():
    w_orig, l_orig = run_case(64, 1, use_original=True)
    w_new, l_new = run_case(64, 1)
    d = max_diff(w_orig, w_new)
    assert d == 0.0, f"max|dW| = {d}, expected exactly 0"
    assert abs(l_orig - l_new) < 1e-12, f"loss {l_orig} vs {l_new}"


def test_a2_micro16_x_accum4_matches_batch64_weights():
    w_ref, _ = run_case(64, 1)
    w_acc, _ = run_case(16, 4)
    d = max_diff(w_ref, w_acc)
    assert d < 1e-6, f"max|dW| = {d:.3e} (float32 eps ~1.2e-7)"


def test_a2_guard_unaccumulated_micro_batch_must_differ():
    """Proves A2 is testing something real and not trivially satisfied."""
    w_ref, _ = run_case(64, 1)
    w_naive, _ = run_case(16, 1)
    d_naive = max_diff(w_ref, w_naive)
    assert d_naive > 1e-6, (
        f"batch16 without accumulation matched batch64 (max|dW| = {d_naive:.2e}) "
        f"-- the A2 contract is vacuous"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
