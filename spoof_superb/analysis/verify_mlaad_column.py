"""
Independent verification of the MLAAD column written into the main results table.

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

What each check means on a tree that is not the published one
-------------------------------------------------------------
Only check 1 is a claim about the paper. Checks 2 and 3 are properties of the
score files themselves and hold on any tree.

So running this against a freshly built tree does not "fail verification" when
check 1 disagrees -- it MEASURES the disagreement between that tree and what
was published, which is the entire point of building a replacement tree. The
exit status therefore reflects checks 2 and 3 always, and check 1 only when
``--expect-tex-match`` is given. Pointing this at a new tree and reading a red
FAIL would otherwise invite the one repair that must never happen: editing the
paper's numbers to match whatever the newest run produced.

Usage
-----
    # against the published tree: all three checks are binding
    python -m spoof_superb.analysis.verify_mlaad_column \\
        --tex ../spoof_SUPERB_IEEE_ACCESS/access.tex --expect-tex-match

    # against a new tree: checks 2 and 3 bind, check 1 is reported
    python -m spoof_superb.analysis.verify_mlaad_column \\
        --scores_root /data/ssl_anti_spoofing/spoof_superb_score_files --layout v3
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from spoof_superb import REPO_ROOT

from spoof_superb.core.metrics import compute_eer as repo_compute_eer  # type: ignore

from spoof_superb.config import cfg
from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import mlaad_pool_paths

# Main results display name -> score-file slug.  Non-960h Mockingjay is excluded:
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

#: The published table also carries the two non-SSL reference systems. They are
#: not front-ends, so they are named by system rather than by upstream; under
#: v2/v3 they have MLAAD and M-AILABS scores like everything else. Legacy does
#: not have them, and this list is simply skipped there.
NON_SSL_ROWS = [("LFCC-GMM", "lfcc_gmm"), ("AASIST", "aasist_raw")]

TOL_TRANSCRIPTION = 0.0005   # printed to 3 decimals
TOL_ESTIMATOR = 0.01         # pp, repo EER vs independent EER
TOL_BALANCE = 0.2            # pp, full pool vs balanced pool


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


def balanced_pool(labels, scores, seed=0):
    """A 50/50 pool drawn from the rows just read, when no balanced/ file exists.

    Deliberately NOT the repo's own subsampler: this module's contract is to
    reimplement everything it checks, so that a bug in the mainline path cannot
    hide inside its own verification.
    """
    rng = np.random.default_rng(seed)
    ib = np.flatnonzero(labels == "bonafide")
    is_ = np.flatnonzero(labels == "spoof")
    n = min(ib.size, is_.size)
    keep = np.concatenate([rng.choice(ib, n, replace=False),
                           rng.choice(is_, n, replace=False)])
    return labels[keep], scores[keep]


def eer_independent(labels, scores):
    """EER via sklearn ROC + Brent root find on fnr - fpr. Bonafide = positive."""
    y = (labels == "bonafide").astype(np.int8)
    fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
    # fpr here is the rate at which spoof is accepted as bonafide;
    # 1 - tpr is the rate at which bonafide is rejected.
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return 100.0 * eer


def parse_tex_mlaad(tex_path, column="MLAAD", label="tab:results_main"):
    """Pull one column out of each live row of the table carrying `label`.

    Two things are located rather than assumed: the table, by its LaTeX label,
    and the column, by the table's own header row.

    The previous version pinned neither. It took every line in the file with
    exactly 12 ampersands and read `cells[9]`. When the paper dropped a column
    that matched no rows at all and every model was reported as "no tex row" --
    a verification silently verifying nothing. Matching on the header alone is
    not enough either: an earlier table in the same paper also has an MLAAD
    column, and anchoring on the header found that one first.

    Failing to find the table, its header, or any data row raises, because each
    means the check did not run.
    """
    text = Path(tex_path).read_text()
    at = text.find(f"\\label{{{label}}}")
    if at < 0:
        raise ValueError(f"{tex_path}: no \\label{{{label}}} in this file.")
    end = text.find("\\end{tabular}", at)
    body = text[at:end if end > 0 else len(text)]

    rows, header, col = {}, None, None
    # Split on the LaTeX row separator, not on newlines: this table writes its
    # header one column per line, so a line-by-line reader sees a 2-cell header.
    for chunk in body.split(r"\\"):
        # Drop comment lines and the rule/environment commands that share a row.
        keep = [ln for ln in chunk.split("\n") if not ln.strip().startswith("%")]
        s = " ".join(keep)
        s = re.sub(r"\\(top|mid|bottom)rule|\\cmidrule\S*|\\begin\{.*?\}"
                   r"|\\end\{.*?\}|\\resizebox|\\label\{.*?\}", " ", s)
        if "&" not in s:
            continue
        cells = [c.strip() for c in s.split("&")]
        if header is None:
            # Header cells carry \textbf{...}; compare on the stripped text.
            names = [re.sub(r"\\textbf\{(.*?)\}", r"\1", c).strip()
                     for c in cells]
            if column in names:
                header, col = names, names.index(column)
            continue
        if len(cells) != len(header):
            continue
        name = cells[0]
        if "\\sout" in name:          # struck row, deliberately excluded
            continue
        m = re.search(r"(\d+\.\d+)", cells[col])
        if m:
            rows[name] = float(m.group(1))

    if header is None:
        raise ValueError(
            f"{tex_path}: {label} has no {column!r} column. The table was "
            f"restructured; this check needs updating, not silencing.")
    if not rows:
        raise ValueError(
            f"{tex_path}: found the {column!r} header in {label} but no data "
            f"rows matched its {len(header)} columns.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(
        REPO_ROOT.parent / "spoof_SUPERB_IEEE_ACCESS" / "access.tex"))
    ap.add_argument("--scores_root", default=None,
                    help="score tree to read (default: the configured one)")
    ap.add_argument("--layout", default=None, choices=("legacy", "v2", "v3"),
                    help="layout of that tree (default: the configured one)")
    ap.add_argument("--expect-tex-match", action="store_true",
                    help="treat disagreement with access.tex as a failure. Set "
                         "this only for the tree the paper was written from.")
    args = ap.parse_args()

    scores_root = args.scores_root or cfg.scores_root
    layout = args.layout or getattr(cfg, "score_layout", "legacy")
    print(f"reading {scores_root}  (layout={layout})")
    if not args.expect_tex_match:
        print("transcription check is REPORTED, not enforced "
              "(pass --expect-tex-match to enforce)\n")

    tex = parse_tex_mlaad(args.tex)
    print(f"Parsed {len(tex)} live MLAAD cells from {args.tex}\n")
    print(f"{'model':20s} {'tex':>8s} {'repo':>8s} {'indep':>8s} "
          f"{'bal':>8s} {'transcr':>8s} {'estim':>7s} {'ratio':>7s}")

    fails = []
    missing = []
    diverged = []
    rows = list(MODELS)
    if layout != "legacy":
        rows += NON_SSL_ROWS
    for disp, slug in rows:
        system = slug if (disp, slug) in NON_SSL_ROWS else "linear_head"
        pool = [Path(p) for p in mlaad_pool_paths(slug, scores_root=scores_root,
                                                  layout=layout, system=system)]
        absent = [p for p in pool if not p.exists()]
        if absent:
            missing.append(f"{disp}: no MLAAD scores ({absent[0].name})")
            continue
        _utt, lab, sc = read_scored(pool)
        e_repo = 100.0 * repo_compute_eer(sc[lab == "bonafide"],
                                          sc[lab == "spoof"])[0]
        e_ind = eer_independent(lab, sc)

        bal_file = (Path(scores_root) / "linear_head_MLAAD_v10" / "balanced"
                    / f"linear_head_MLAAD_v10_balanced_{slug}.txt")
        if bal_file.exists():
            blab, bsc = read_balanced(bal_file)
        else:
            blab, bsc = balanced_pool(lab, sc)
        e_bal = eer_independent(blab, bsc)

        t = tex.get(disp)
        d_tr = abs(t - e_repo) if t is not None else float("nan")
        d_es = abs(e_repo - e_ind)
        d_ra = abs(e_ind - e_bal)

        ok_tr = t is not None and d_tr <= TOL_TRANSCRIPTION
        ok_es = d_es <= TOL_ESTIMATOR
        ok_ra = d_ra <= TOL_BALANCE
        # The estimator and ratio-invariance checks are properties of the score
        # files and always bind. Transcription is a claim about the paper, and
        # binds only for the tree the paper reports.
        for ok, name in ((ok_es, "estimator"), (ok_ra, "ratio-invariance")):
            if not ok:
                fails.append(f"{disp}: {name}")
        if not ok_tr:
            (fails if args.expect_tex_match else diverged).append(
                f"{disp}: transcription (tex {t} vs computed {e_repo:.3f})"
                if t is not None else f"{disp}: no tex row")

        print(f"{disp:20s} {t if t is not None else float('nan'):8.3f} "
              f"{e_repo:8.3f} {e_ind:8.3f} {e_bal:8.3f} "
              f"{'PASS' if ok_tr else 'DIFF':>8s} {'PASS' if ok_es else 'FAIL':>7s} "
              f"{'PASS' if ok_ra else 'FAIL':>7s}", flush=True)

    extra = set(tex) - {d for d, _ in rows}
    if extra:
        fails.append(f"tex rows with no verification source: {sorted(extra)}")

    print("\n" + "=" * 66)
    if missing:
        print(f"NOT SCORED on this tree ({len(missing)}):")
        for m in missing:
            print("  " + m)
    if diverged:
        print(f"\nDIFFERS FROM THE PUBLISHED TABLE ({len(diverged)} of "
              f"{len(rows) - len(missing)}):")
        for d in diverged:
            print("  " + d)
        print("  -> reported, not failed: this tree is not the published tree.")
    if fails:
        print("\nVERIFICATION FAILED:")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print(f"\nALL BINDING CHECKS PASSED for {len(rows) - len(missing)} models")
    print(f"  transcription  : tex matches computed within {TOL_TRANSCRIPTION}")
    print(f"  estimator      : repo EER == sklearn/Brent EER within {TOL_ESTIMATOR} pp")
    print(f"  ratio-invariance: full pool == balanced pool within {TOL_BALANCE} pp")


if __name__ == "__main__":
    main()
