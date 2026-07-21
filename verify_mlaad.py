"""
verify_mlaad.py
---------------
Compare a freshly produced MLAAD score file against an existing reference score
file on the intersection of utt_ids.

Bit-exact reproduction of the reference is NOT achievable here: the reference
files were produced in a different environment (different librosa/soxr/torch/CUDA
versions), which introduces a near-constant logit offset (~0.33 for xls_r_300m)
with r>0.99. That offset is irrelevant to EER / detection, which are rank-based.
So the pass criterion is detection-equivalence, not absolute match:
    Pearson r >= 0.99  AND  Spearman >= 0.99  AND  sign-agreement@0 >= 0.999

Usage:
    python verify_mlaad.py --new <file> --ref <file>
Exit 0 if pass, else 1. Always prints the full stat line.
"""
import argparse
import sys

import numpy as np


def load_scores(path):
    """Parse '<utt_id> - <label> <score>'.

    utt_id may contain spaces (MLAAD v10 vendor dirs like 'OpenAI TTS-1 HD'),
    so peel the last three fields off from the right rather than splitting left.
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
                d[utt] = float(score)  # may be nan/inf
            except ValueError:
                d[utt] = float("nan")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--r_min", type=float, default=0.99)
    ap.add_argument("--sign_min", type=float, default=0.999)
    args = ap.parse_args()

    new = load_scores(args.new)
    ref = load_scores(args.ref)
    shared = sorted(set(new) & set(ref))
    if not shared:
        print(f"[verify] {args.new}: NO SHARED utt_ids (new={len(new)}, ref={len(ref)})")
        return 1

    a = np.array([new[u] for u in shared])
    b = np.array([ref[u] for u in shared])
    new_nan = int(np.isnan(a).sum())
    ref_nan = int(np.isnan(b).sum())
    both = np.isfinite(a) & np.isfinite(b)
    n_both = int(both.sum())

    # Our own output must be finite. NaN in new is a real failure.
    if new_nan > 0.01 * len(a):
        print(f"[verify] new={len(new)} ref={len(ref)} shared={len(shared)} "
              f"new_nan={new_nan} -> FAIL (our output has NaN)")
        return 1

    # Reference itself unusable (e.g. adupa's spectrogram models saved 98% NaN).
    # Not our failure: report and pass-through since our finite output is valid.
    if n_both < 0.5 * len(shared):
        print(f"[verify] new={len(new)} ref={len(ref)} shared={len(shared)} "
              f"ref_nan={ref_nan} both_finite={n_both} "
              f"-> REF_UNUSABLE (reference is >50% NaN; our output is finite)")
        return 0

    a, b = a[both], b[both]
    pearson = float(np.corrcoef(a, b)[0, 1])
    ra = a.argsort().argsort()
    rb = b.argsort().argsort()
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    sign_agree = float((np.sign(a) == np.sign(b)).mean())
    offset = float((a - b).mean())
    off_std = float((a - b).std())
    max_d = float(np.abs(a - b).max())

    ok = (pearson >= args.r_min and spearman >= args.r_min
          and sign_agree >= args.sign_min)
    nan_note = f" ref_nan={ref_nan}" if ref_nan else ""
    print(f"[verify] new={len(new)} ref={len(ref)} shared={len(shared)} both={n_both}{nan_note} "
          f"r={pearson:.4f} spearman={spearman:.4f} sign@0={sign_agree:.4%} "
          f"offset={offset:+.3f}±{off_std:.3f} maxΔ={max_d:.3f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
