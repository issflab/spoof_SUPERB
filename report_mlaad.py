"""
report_mlaad.py
---------------
Consolidated verification report for the MLAAD v10 score files.

Regrades every produced score file against its Multilingual reference using
rank correlation as the primary criterion, because that is what the benchmark
actually depends on (EER is rank-based). Absolute-score agreement is reported
as a diagnostic but is not a pass condition: the references were produced in a
different environment and carry a near-constant offset, and Pearson is easily
dragged down by a handful of tail outliers.

Verdicts:
  OK            - spearman >= 0.99 (rank/EER equivalent to the reference)
  WEAK          - spearman <  0.99 (real divergence; inspect before use)
  REF_UNUSABLE  - reference is >50% NaN (nothing valid to compare against;
                  our finite output supersedes it)
  NAN_OUTPUT    - our own output contains NaN (real failure)
  MISSING       - no score file produced
"""
import os
import sys

import numpy as np

OUT_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10"
REF_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head"
MODELS_ROOT = "/data/ssl_anti_spoofing/asd_superb_models/linear_head_models"
PREFIX = "model_weighted_CCE_50_64_linear_head_ASV19_"
EXPECTED_LINES = 456000


def load(path):
    """Parse '<utt_id> - <label> <score>'.

    utt_id may itself contain spaces: MLAAD v10 adds vendor directories such as
    'Cartesia.ai (Sonic-3)' and 'OpenAI TTS-1 HD'. Splitting from the left would
    truncate those ids and silently merge every file in such a directory, so the
    last three fields are peeled off from the right instead.
    """
    d = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) < 4:
                continue
            utt, _dash, _label, score = parts
            try:
                d[utt] = float(score)
            except ValueError:
                d[utt] = float("nan")
    return d


def main():
    ssls = sorted(n[len(PREFIX):] for n in os.listdir(MODELS_ROOT)
                  if n.startswith(PREFIX) and
                  os.path.isfile(os.path.join(MODELS_ROOT, n, "swa.pth")))

    rows = []
    for ssl in ssls:
        out = os.path.join(OUT_DIR, f"linear_head_MLAAD_v10_{ssl}.txt")
        ref = os.path.join(REF_DIR, f"linear_head_Multilingual_{ssl}.txt")
        if not os.path.isfile(out):
            rows.append((ssl, 0, "-", "-", "-", "-", "MISSING"))
            continue

        new = load(out)
        a_all = np.array(list(new.values()))
        n_nan = int(np.isnan(a_all).sum())
        n_lines = len(new)

        if n_nan > 0.01 * max(1, n_lines):
            rows.append((ssl, n_lines, "-", "-", "-", "-", f"NAN_OUTPUT({n_nan})"))
            continue
        if not os.path.isfile(ref):
            rows.append((ssl, n_lines, "-", "-", "-", "-", "NO_REF"))
            continue

        r_ = load(ref)
        shared = sorted(set(new) & set(r_))
        a = np.array([new[u] for u in shared])
        b = np.array([r_[u] for u in shared])
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < 0.5 * len(shared):
            rows.append((ssl, n_lines, len(shared), "-", "-", "-", "REF_UNUSABLE"))
            continue

        a, b = a[both], b[both]
        pe = float(np.corrcoef(a, b)[0, 1])
        sp = float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])
        off = float((a - b).mean())
        verdict = "OK" if sp >= 0.99 else "WEAK"
        rows.append((ssl, n_lines, int(both.sum()), f"{pe:.4f}", f"{sp:.4f}",
                     f"{off:+.3f}", verdict))

    w = max(len(r[0]) for r in rows) + 1
    print(f"{'model':<{w}} {'lines':>7} {'shared':>7} {'pearson':>8} {'spearman':>9} "
          f"{'offset':>8}  verdict")
    print("-" * (w + 55))
    for r in rows:
        print(f"{r[0]:<{w}} {r[1]:>7} {str(r[2]):>7} {str(r[3]):>8} {str(r[4]):>9} "
              f"{str(r[5]):>8}  {r[6]}")

    n_ok = sum(1 for r in rows if r[6] == "OK")
    n_weak = sum(1 for r in rows if r[6] == "WEAK")
    n_refu = sum(1 for r in rows if r[6] == "REF_UNUSABLE")
    n_bad = sum(1 for r in rows if r[6].startswith(("NAN", "MISSING")))
    n_full = sum(1 for r in rows if r[1] == EXPECTED_LINES)
    print()
    print(f"TOTAL {len(rows)} models | OK={n_ok} WEAK={n_weak} "
          f"REF_UNUSABLE={n_refu} MISSING/NAN={n_bad} | "
          f"{n_full} files with full {EXPECTED_LINES} utts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
