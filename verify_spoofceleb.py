"""
verify_spoofceleb.py
--------------------
Compare a freshly produced SpoofCeleb score file against the existing reference
on the intersection of utt_ids.

Differs from verify_mlaad.py in the GRADE, which is why it is a separate script
rather than a flag: EER is rank-based, so the verdict is decided by Spearman
alone (>= 0.99). Pearson is reported as a diagnostic only -- on the MLAAD run a
handful of tail outliers dragged Pearson to 0.92 on models whose Spearman was
0.996, which would have failed a Pearson-gated check for no detection-relevant
reason.

SpoofCeleb audio is natively 16 kHz, so no resampling happens and near-exact
reproduction is expected (r ~ 0.999-1.000), unlike MLAAD's ~0.33 resampler drift.

Four references (tera, mockingjay, mockingjay_960hr, audio_albert_960hr) are
~22.8% NaN. They remain >50% finite, so we grade on the both-finite subset
instead of declaring them unusable. NaN in OUR output is always a hard failure.

Usage:
    python verify_spoofceleb.py --new <file> --ref <file>
Exit 0 if pass, else 1. Always prints the full stat line.
"""
import argparse
import sys

import numpy as np


def load_scores(path):
    """Parse '<utt_id> - <label> <score>' (peel 3 fields off the right)."""
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
    ap.add_argument("--spearman_min", type=float, default=0.99)
    args = ap.parse_args()

    new = load_scores(args.new)
    ref = load_scores(args.ref)
    shared = sorted(set(new) & set(ref))
    if not shared:
        print(f"[verify] {args.new}: NO SHARED utt_ids (new={len(new)}, ref={len(ref)})")
        return 1

    a = np.array([new[u] for u in shared])
    b = np.array([ref[u] for u in shared])
    new_nan = int((~np.isfinite(a)).sum())
    ref_nan = int((~np.isfinite(b)).sum())
    both = np.isfinite(a) & np.isfinite(b)
    n_both = int(both.sum())

    # Our own output must be finite; any NaN is a real failure, flagged loudly.
    if new_nan:
        print(f"[verify] new={len(new)} ref={len(ref)} shared={len(shared)} "
              f"new_nan={new_nan} -> FAIL (our output contains NaN/inf)")
        return 1

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

    ok = spearman >= args.spearman_min  # rank correlation is the verdict
    nan_note = f" ref_nan={ref_nan}" if ref_nan else ""
    print(f"[verify] new={len(new)} ref={len(ref)} shared={len(shared)} both={n_both}{nan_note} "
          f"spearman={spearman:.4f} r={pearson:.4f} sign@0={sign_agree:.4%} "
          f"offset={offset:+.4f}+/-{off_std:.4f} maxdiff={max_d:.4f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
