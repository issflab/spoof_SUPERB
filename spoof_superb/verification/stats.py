"""Score-file comparison statistics.

Every score-file verifier in this repo computed the same numbers -- Pearson,
Spearman, sign agreement at 0, mean offset, max delta -- and then applied a
different verdict to them. The statistics are here; the verdicts are in
policies.py, because they are not interchangeable.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["load_scores", "Comparison", "compare"]


def load_scores(path):
    """Parse '<utt_id> - <label> <score>' -> {utt_id: score}.

    utt_id may contain spaces (MLAAD v10 vendor dirs like 'OpenAI TTS-1 HD'),
    so peel the last three fields off from the right rather than splitting left.
    An unparseable score becomes NaN rather than dropping the row: a NaN that
    the producer wrote is exactly what these checks exist to catch.
    """
    d = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) < 4:
                continue
            utt, _dash, _label, score = parts
            try:
                d[utt] = float(score)   # may be nan/inf
            except ValueError:
                d[utt] = float("nan")
    return d


@dataclass
class Comparison:
    n_new: int
    n_ref: int
    n_shared: int
    n_both_finite: int
    new_nan: int
    ref_nan: int
    pearson: float = float("nan")
    spearman: float = float("nan")
    sign_agree: float = float("nan")
    offset: float = float("nan")
    offset_std: float = float("nan")
    max_delta: float = float("nan")

    def line(self):
        """The one-line report the orchestrators parse out of the log."""
        nan_note = f" ref_nan={self.ref_nan}" if self.ref_nan else ""
        return (f"new={self.n_new} ref={self.n_ref} shared={self.n_shared} "
                f"both={self.n_both_finite}{nan_note} "
                f"r={self.pearson:.4f} spearman={self.spearman:.4f} "
                f"sign@0={self.sign_agree:.4%} "
                f"offset={self.offset:+.3f}±{self.offset_std:.3f} "
                f"maxΔ={self.max_delta:.3f}")


def compare(new_path, ref_path):
    """Compare two score files on the intersection of their utt_ids."""
    new = load_scores(new_path)
    ref = load_scores(ref_path)
    shared = sorted(set(new) & set(ref))

    if not shared:
        return Comparison(len(new), len(ref), 0, 0, 0, 0)

    a = np.array([new[u] for u in shared])
    b = np.array([ref[u] for u in shared])
    both = np.isfinite(a) & np.isfinite(b)

    c = Comparison(
        n_new=len(new), n_ref=len(ref), n_shared=len(shared),
        n_both_finite=int(both.sum()),
        new_nan=int((~np.isfinite(a)).sum()),
        ref_nan=int((~np.isfinite(b)).sum()),
    )
    if c.n_both_finite < 2:
        return c

    a, b = a[both], b[both]
    c.pearson = float(np.corrcoef(a, b)[0, 1])
    # Spearman as Pearson over ranks, matching the original implementations.
    c.spearman = float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])
    c.sign_agree = float((np.sign(a) == np.sign(b)).mean())
    c.offset = float((a - b).mean())
    c.offset_std = float((a - b).std())
    c.max_delta = float(np.abs(a - b).max())
    return c
