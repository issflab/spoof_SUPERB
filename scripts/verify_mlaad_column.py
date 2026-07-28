"""
Independent verification of the MLAAD column written into Table 5.

This deliberately does NOT reuse evaluation.compute_eer, the repo's own EER
routine, nor the recompute script's parsing.  It re-reads the MLAAD v10 score
files from scratch and computes EER via sklearn's ROC curve with a Brent root
find on fnr(t) - fpr(t).  If a bug lived in compute_det_curve or in the
recompute script's reader, this path would not reproduce it.

Three checks:
  1. TRANSCRIPTION -- the numbers printed in access.tex match the computed
     values exactly at the precision printed (3 decimals).
  2. INDEPENDENT EER -- sklearn/Brent EER agrees with the repo EER.
  3. RATIO INVARIANCE -- full pool vs balanced pool agree, recomputed here
     with the independent estimator rather than trusting the earlier run.

Usage
-----
    python3 scripts/verify_mlaad_column.py --tex ../spoof_SUPERB_IEEE_ACCESS/access.tex
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spoof_superb.core.metrics import compute_eer as repo_compute_eer  # type: ignore

V10 = Path("/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10")

# Table 5 display name -> score-file slug.  Non-960h Mockingjay is excluded:
# it has no v10 scores and its row is struck from the table.
MODELS = [
    ("FBANK", "fbank"), ("APC", "apc"), ("VQ-APC", "vq_apc"), ("NPC", "npc"),
    ("Mockingjay-960h", "mockingjay_960hr"), ("TERA", "tera"),
    ("DeCoAR 2.0", "decoar2"), ("wav2vec", "wav2vec"),
    ("wav2vec 2.0 Base", "wav2vec2_base_960"),
    ("wav2vec 2.0 Large", "wav2vec2_large_ll60k"),
    ("HuBERT Base", "hubert_base"), ("HuBERT Large", "hubert_large_ll60k"),
    ("MR-HuBERT", "multires_hubert_multilingual_large600k"),
    ("XLS-R", "xls_r_300m"), ("UniSpeech-SAT", "unispeech_sat_large"),
    ("Data2Vec", "data2vec_large_ll60k"), ("WAVLABLM", "wavlablm_ek_40k"),
    ("WavLM Large", "wavlm_large"), ("SSAST", "ssast_frame_base"),
    ("MAE-AST-FRAME", "mae_ast_frame"),
]

TOL_TRANSCRIPTION = 0.0005   # printed to 3 decimals
TOL_ESTIMATOR = 0.01         # pp, repo EER vs independent EER
TOL_BALANCE = 0.2            # pp, full pool vs balanced pool


def read_tsv(path):
    lab, sc = [], []
    with open(path) as fh:
        if fh.readline().rstrip("\n").split("\t") != ["utt_id", "label", "score"]:
            raise ValueError(f"{path}: unexpected header")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            a = line.split("\t")
            lab.append(a[1])
            sc.append(float(a[2]))
    return np.asarray(lab), np.asarray(sc, dtype=np.float64)


def read_balanced(path):
    lab, sc = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            _u, _d, l, s = line.rsplit(" ", 3)
            if l == "label":
                continue
            lab.append(l)
            sc.append(float(s))
    return np.asarray(lab), np.asarray(sc, dtype=np.float64)


def eer_independent(labels, scores):
    """EER via sklearn ROC + Brent root find on fnr - fpr. Bonafide = positive."""
    y = (labels == "bonafide").astype(np.int8)
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
    # fpr here is the rate at which spoof is accepted as bonafide;
    # 1 - tpr is the rate at which bonafide is rejected.
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return 100.0 * eer


def parse_tex_mlaad(tex_path):
    """Pull the MLAAD cell (9th data column) out of each live Table 5 row."""
    out = {}
    for line in Path(tex_path).read_text().split("\n"):
        s = line.strip()
        if s.startswith("%") or s.count("&") != 12 or "\\\\" not in s:
            continue
        cells = [c.strip() for c in s.split("&")]
        name = cells[0]
        if "\\sout" in name:          # struck row, deliberately excluded
            continue
        cell = cells[9]               # MLAAD
        m = re.search(r"(\d+\.\d+)", cell)
        if m:
            out[name] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(
        REPO_ROOT.parent / "spoof_SUPERB_IEEE_ACCESS" / "access.tex"))
    args = ap.parse_args()

    tex = parse_tex_mlaad(args.tex)
    print(f"Parsed {len(tex)} live MLAAD cells from {args.tex}\n")
    print(f"{'model':20s} {'tex':>8s} {'repo':>8s} {'indep':>8s} "
          f"{'bal':>8s} {'transcr':>8s} {'estim':>7s} {'ratio':>7s}")

    fails = []
    for disp, slug in MODELS:
        lab, sc = read_tsv(V10 / "tsv" / f"linear_head_MLAAD_v10_{slug}.tsv")
        e_repo = 100.0 * repo_compute_eer(sc[lab == "bonafide"],
                                          sc[lab == "spoof"])[0]
        e_ind = eer_independent(lab, sc)

        blab, bsc = read_balanced(
            V10 / "balanced" / f"linear_head_MLAAD_v10_balanced_{slug}.txt")
        e_bal = eer_independent(blab, bsc)

        t = tex.get(disp)
        d_tr = abs(t - e_repo) if t is not None else float("nan")
        d_es = abs(e_repo - e_ind)
        d_ra = abs(e_ind - e_bal)

        ok_tr = t is not None and d_tr <= TOL_TRANSCRIPTION
        ok_es = d_es <= TOL_ESTIMATOR
        ok_ra = d_ra <= TOL_BALANCE
        for ok, name in ((ok_tr, "transcription"), (ok_es, "estimator"),
                         (ok_ra, "ratio-invariance")):
            if not ok:
                fails.append(f"{disp}: {name}")

        print(f"{disp:20s} {t if t is not None else float('nan'):8.3f} "
              f"{e_repo:8.3f} {e_ind:8.3f} {e_bal:8.3f} "
              f"{'PASS' if ok_tr else 'FAIL':>8s} {'PASS' if ok_es else 'FAIL':>7s} "
              f"{'PASS' if ok_ra else 'FAIL':>7s}", flush=True)

    extra = set(tex) - {d for d, _ in MODELS}
    if extra:
        fails.append(f"tex rows with no verification source: {sorted(extra)}")

    print("\n" + "=" * 66)
    if fails:
        print("VERIFICATION FAILED:")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print(f"ALL CHECKS PASSED for {len(MODELS)} models")
    print(f"  transcription  : tex matches computed within {TOL_TRANSCRIPTION}")
    print(f"  estimator      : repo EER == sklearn/Brent EER within {TOL_ESTIMATOR} pp")
    print(f"  ratio-invariance: full pool == balanced pool within {TOL_BALANCE} pp")


if __name__ == "__main__":
    main()
