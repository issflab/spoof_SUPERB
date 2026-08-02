"""
Recompute the Spoof-SUPERB IEEE Access paper's two results tables:

    tab:results_main        the main results table  (EER per model per dataset)
    tab:top_ssl_lineage     the top-SSL-lineage table

Referred to by LaTeX label, never by number. The numbers move between drafts --
tab:results_main has been Table 5 and is currently Table 6 -- so encoding one in
a filename or a docstring guarantees the code goes stale against the paper.

This exists because the old MLAAD reference scores were corrupted and had to be
replaced with the freshly computed MLAAD v10 + M-AILABS score files.

Why this script exists
----------------------
The old MLAAD reference files
    /data/.../linear_head/linear_head_Multilingual_<model>.txt
contain NaN scores for four models (mockingjay, mockingjay_960hr, tera,
audio_albert_960hr): 151,051 of the 154,000 MLAAD spoof rows are NaN, i.e.
98.085% -- which is exactly the "EER" that was written into the paper. The
number was never a result; it was the NaN fraction.

The replacement pool lives in
    /data/.../linear_head_MLAAD_v10/tsv/linear_head_MLAAD_v10_<model>.tsv
and contains 1,040,006 rows per model (456,000 MLAAD v10 spoof +
584,006 M-AILABS bonafide), zero NaN.  Because the pool changed size and
composition (307,998 -> 1,040,006 rows), *every* model's MLAAD EER moves,
not just the four broken ones.  Mean and Pooled therefore move too.

Parsing contracts (both load-bearing -- do not "simplify" away)
--------------------------------------------------------------
1. MLAAD v10 vendor directories ("Cartesia.ai (Sonic-3)", "OpenAI TTS-1 HD")
   put SPACES inside ~8.6% of utt_ids.  The MLAAD v10 scores must therefore be
   read from the tab-separated `tsv/` copies, or -- for the `balanced/` files,
   which have no tsv companion -- split from the RIGHT with rsplit.
   Splitting on whitespace from the left silently mis-parses those rows.
2. The legacy per-dataset files are 4-column "utt_id - label score".  utt_ids
   there (e.g. Famous Figures absolute paths) are also parsed from the right.

Verification contract
---------------------
EER is computed from class-conditional error rates and is therefore invariant
to the bonafide:spoof ratio.  Each model's MLAAD EER is computed on BOTH the
full 1,040,006-row pool and the 912,000-row `balanced/` pool; they must agree
within 0.2 percentage points.  A larger gap indicates a parsing, labelling or
polarity bug and is reported as a FAIL rather than silently resolved.

Usage
-----
    python -m spoof_superb.analysis.recompute_main_results \
        --out_dir scripts/verification_out
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from spoof_superb.core.metrics import compute_eer  # type: ignore

from spoof_superb.config import cfg

LEGACY_DIR = Path(cfg.reference_dir)
V10_DIR = Path(f"{cfg.scores_root}/linear_head_MLAAD_v10")
RECOMP_DIR = Path(
    f"{cfg.scores_root}/asvld_rerun/Recompression")

# Main results column header -> legacy file prefix.  MLAAD is handled separately.
#
# ASVLD is TWO files, not one.  Commit 6bf39a0 ("Fold ASVLD recompression into
# main benchmark") folded the six-bitrate re-compression variants into this
# column, but they live in a separate directory (asvld_rerun/Recompression/).
# Pooling linear_head_asvspoofLD_<m>.txt (1,207,509 rows: noise x10, reverb x3,
# resample x4) with linear_head_Recompression_<m>.txt (427,422 rows = 71,237 x 6
# bitrates) reproduces the published ASVLD column to +/-0.000 for all 21 models.
# Reading only the first file silently reproduces a pre-6bf39a0 column.
DATASETS = [
    ("ASV19 LA",   "eval_2019"),
    ("ASV21 LA",   "asvspoof2021_LA"),
    ("ASV21 DF",   "asvspoof2021_DF"),
    ("ASV5 Eval",  "asvspoof5"),
    ("ITW",        "wild"),
    ("DFEval24",   "deepfake_eval_2024"),
    ("FF",         "Famous_Figures"),
    ("ASVLD",      "asvspoofLD"),
    ("MLAAD",      None),
    ("SpoofCeleb", "spoofceleb"),
]

# Models whose legacy scores are NaN-corrupted on these datasets.  The MLAAD v10
# re-run fixed MLAAD only; ASVLD (23.50% NaN) and SpoofCeleb (22.78% NaN) remain
# corrupt for the four masked-spectrogram-reconstruction models.  The published
# SpoofCeleb values for these models are reproduced exactly by a NaN-inclusive
# computation, confirming the corruption reached the paper.  Per author decision
# these cells are reported as [TODO-verify] rather than recomputed on the
# surviving finite subset, and Mean/Pooled are withheld for the same rows.
NAN_CORRUPT = {
    "Mockingjay":      ["ASVLD", "SpoofCeleb"],
    "Mockingjay-960h": ["ASVLD", "SpoofCeleb"],
    "TERA":            ["ASVLD", "SpoofCeleb"],
}

# Main results row order -> score-file model slug.
MODELS = [
    ("FBANK",             "fbank"),
    ("APC",               "apc"),
    ("VQ-APC",            "vq_apc"),
    ("NPC",               "npc"),
    ("Mockingjay",        "mockingjay"),
    ("Mockingjay-960h",   "mockingjay_960hr"),
    ("TERA",              "tera"),
    ("DeCoAR 2.0",        "decoar2"),
    ("wav2vec",           "wav2vec"),
    ("wav2vec 2.0 Base",  "wav2vec2_base_960"),
    ("wav2vec 2.0 Large", "wav2vec2_large_ll60k"),
    ("HuBERT Base",       "hubert_base"),
    ("HuBERT Large",      "hubert_large_ll60k"),
    ("MR-HuBERT",         "multires_hubert_multilingual_large600k"),
    ("XLS-R",             "xls_r_300m"),
    ("UniSpeech-SAT",     "unispeech_sat_large"),
    ("Data2Vec",          "data2vec_large_ll60k"),
    ("WAVLABLM",          "wavlablm_ek_40k"),
    ("WavLM Large",       "wavlm_large"),
    ("SSAST",             "ssast_frame_base"),
    ("MAE-AST-FRAME",     "mae_ast_frame"),
]

# tab:top_ssl_lineage rows, in the order the paper prints them.
LINEAGE = [
    ("wav2vec 2.0",   "wav2vec",      "Masked latent prediction"),
    ("HuBERT Base",   "wav2vec 2.0",  "Hidden-unit prediction"),
    ("HuBERT Large",  "HuBERT Base",  "Model scaling"),
    ("WAVLABLM",      "WavLM/XLS-R",  "Multilingual + denoising"),
    ("MR-HuBERT",     "HuBERT",       "Multi-resolution SSL"),
    ("WavLM Large",   "HuBERT",       "Denoising SSL"),
    ("UniSpeech-SAT", "HuBERT",       "Speaker-aware SSL"),
    ("XLS-R",         "wav2vec 2.0",  "Multilingual SSL"),
]
# In tab:top_ssl_lineage, "wav2vec 2.0" is the Base variant (paper text: 32.88 == wav2vec 2.0 Base).
LINEAGE_ROW_TO_MAIN = {"wav2vec 2.0": "wav2vec 2.0 Base"}

# Published main-results values (commit fa16daa), used purely as a regression gate:
# the eight columns this task does not touch, plus ASVLD, must reproduce to
# +/-0.01.  If they do not, the parsing or the score sources have drifted and
# no recomputed MLAAD/Mean/Pooled number can be trusted.
PUBLISHED = {
    "FBANK":             [42.828, 43.155, 44.789, 49.838, 48.393, 47.113, 48.427, 44.579, 53.627, 50.432],
    "APC":               [10.075, 16.335, 22.276, 33.311, 36.889, 42.662, 58.402, 17.599, 37.267, 40.985],
    "VQ-APC":            [12.155, 18.872, 20.217, 30.581, 34.860, 52.173, 58.544, 17.880, 39.747, 41.699],
    "NPC":               [15.243, 17.619, 25.239, 37.868, 40.986, 49.843, 51.979, 19.766, 42.830, 43.554],
    "Mockingjay":        [15.430, 19.798, 25.312, 40.217, 35.848, 49.800, 40.975, 37.872, 98.085, 40.503],
    "Mockingjay-960h":   [13.801, 25.525, 22.584, 37.866, 52.387, 52.130, 49.953, 39.042, 98.085, 46.362],
    "TERA":              [ 9.112, 26.572, 17.254, 35.656, 39.894, 54.251, 49.282, 34.107, 98.085, 38.999],
    "DeCoAR 2.0":        [ 7.628, 12.352, 18.990, 29.571, 35.029, 49.800, 54.452, 13.281, 38.204, 38.442],
    "wav2vec":           [ 8.812, 15.500, 14.761, 30.691, 42.239, 53.895, 51.048, 23.318, 54.451, 40.479],
    "wav2vec 2.0 Base":  [ 4.661, 11.452, 10.046, 18.698, 40.945, 56.981, 51.921, 23.886, 39.797, 32.601],
    "wav2vec 2.0 Large": [ 7.695, 18.887, 11.617, 19.956, 40.461, 55.764, 44.401, 24.711, 33.045, 34.149],
    "HuBERT Base":       [ 4.867, 12.562, 13.387, 23.990, 27.276, 53.747, 53.749, 13.848, 36.841, 29.671],
    "HuBERT Large":      [ 2.788, 10.049, 11.996, 21.252, 21.039, 52.991, 48.440, 12.593, 24.704, 25.721],
    "MR-HuBERT":         [ 2.478,  9.017, 11.635, 23.056, 23.799, 49.696, 52.720,  7.041, 20.790, 23.373],
    "XLS-R":             [ 1.985, 14.096,  4.314, 14.394, 20.073, 45.392, 29.598, 10.782,  9.203, 19.269],
    "UniSpeech-SAT":     [ 1.961,  8.818,  7.443, 14.996, 16.791, 49.800, 46.601,  8.965, 13.095, 18.073],
    "Data2Vec":          [ 7.695, 12.877, 16.511, 26.773, 29.249, 50.808, 53.092, 12.808, 20.351, 32.054],
    "WAVLABLM":          [ 3.631, 15.380,  9.847, 21.115, 23.402, 52.530, 52.660, 13.323, 24.992, 29.649],
    "WavLM Large":       [ 2.273, 11.636, 11.527, 17.549, 24.331, 49.696, 35.367, 10.951, 19.259, 24.922],
    "SSAST":             [11.693, 24.935, 22.909, 31.186, 47.113, 40.184, 36.885, 27.139, 13.100, 36.585],
    "MAE-AST-FRAME":     [ 7.685, 19.554, 17.001, 27.295, 43.645, 47.974, 35.214, 22.009,  9.120, 34.994],
}
# Columns expected to change; everything else is a regression gate.
CHANGED_COLS = {"MLAAD"}
REPRO_TOL = 0.01

MLAAD_FULL_ROWS = 1_040_006
MLAAD_BALANCED_ROWS = 912_000
BALANCE_TOL_PP = 0.2


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def read_legacy(path):
    """Legacy 4-column file: '<utt_id> - <label> <score>'.

    utt_ids may contain '/' and (Famous Figures) full absolute paths, so the
    three trailing fields are peeled from the RIGHT.  A leading header line, if
    present, is dropped.
    """
    labels, scores = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) != 4:
                raise ValueError(f"{path}: cannot parse line: {line!r}")
            _utt, _dash, label, score = parts
            if label == "label":  # header
                continue
            labels.append(label)
            scores.append(float(score))
    return np.asarray(labels), np.asarray(scores, dtype=np.float64)


def read_v10_tsv(path):
    """MLAAD v10 tab-separated file: 'utt_id\\tlabel\\tscore' with a header.

    Tab separation is mandatory: ~8.6% of v10 utt_ids contain literal spaces
    (vendor dirs such as 'Cartesia.ai (Sonic-3)' and 'OpenAI TTS-1 HD').
    """
    labels, scores = [], []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != ["utt_id", "label", "score"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}: cannot parse line: {line!r}")
            labels.append(parts[1])
            scores.append(float(parts[2]))
    return np.asarray(labels), np.asarray(scores, dtype=np.float64)


def read_v10_balanced(path):
    """Balanced MLAAD v10 file in the space-delimited reference format.

    There is no tsv companion for balanced/, so the three trailing fields are
    peeled from the RIGHT to survive spaces inside utt_ids.
    """
    labels, scores = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) != 4:
                raise ValueError(f"{path}: cannot parse line: {line!r}")
            _utt, _dash, label, score = parts
            if label == "label":
                continue
            labels.append(label)
            scores.append(float(score))
    return np.asarray(labels), np.asarray(scores, dtype=np.float64)


def eer_pct(labels, scores):
    """EER in percent.  Higher score == more bonafide (the repo convention:
    bonafide are the target class)."""
    bona = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("one class is empty")
    eer, _thr = compute_eer(bona, spoof)
    return 100.0 * eer


def nan_frac(scores):
    return float(np.isnan(scores).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "scripts" / "verification_out"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}   # display name -> dict
    problems = []

    for disp, slug in MODELS:
        row = {"slug": slug, "datasets": {}, "asserts": []}
        pooled_labels, pooled_scores = [], []

        # ---- legacy datasets -------------------------------------------
        withheld = NAN_CORRUPT.get(disp, [])
        for col, prefix in DATASETS:
            if prefix is None:
                continue
            paths = [LEGACY_DIR / f"linear_head_{prefix}_{slug}.txt"]
            if col == "ASVLD":
                paths.append(RECOMP_DIR / f"linear_head_Recompression_{slug}.txt")
            missing = [p for p in paths if not p.exists()]
            if missing:
                row["datasets"][col] = None
                problems.append(f"{disp}: MISSING legacy file {missing[0].name}")
                continue
            chunks = [read_legacy(p) for p in paths]
            lab = np.concatenate([c[0] for c in chunks])
            sc = np.concatenate([c[1] for c in chunks])
            nf = nan_frac(sc)
            if nf > 0:
                row["asserts"].append(f"{col}: NaN fraction {100*nf:.3f}%")
            if col in withheld:
                # NaN-corrupted beyond repair without a re-run: report, do not
                # substitute a finite-subset value, and drop from Mean/Pooled.
                row["datasets"][col] = {
                    "eer": None, "todo_verify": True,
                    "n": int(sc.size), "nan_frac": nf,
                }
                continue
            finite = ~np.isnan(sc)
            row["datasets"][col] = {
                "eer": eer_pct(lab[finite], sc[finite]),
                "n": int(sc.size),
                "nan_frac": nf,
            }
            pooled_labels.append(lab[finite])
            pooled_scores.append(sc[finite])

        # ---- MLAAD v10, full pool --------------------------------------
        tsv = V10_DIR / "tsv" / f"linear_head_MLAAD_v10_{slug}.tsv"
        bal = V10_DIR / "balanced" / f"linear_head_MLAAD_v10_balanced_{slug}.txt"
        if not tsv.exists():
            row["datasets"]["MLAAD"] = None
            problems.append(f"{disp}: MISSING MLAAD v10 file {tsv.name} "
                            f"-- no recomputed MLAAD/Mean/Pooled possible")
            results[disp] = row
            continue

        lab, sc = read_v10_tsv(tsv)
        # asserts
        if sc.size != MLAAD_FULL_ROWS:
            row["asserts"].append(f"MLAAD row count {sc.size} != {MLAAD_FULL_ROWS}")
        n_nan = int(np.isnan(sc).sum())
        if n_nan:
            row["asserts"].append(f"MLAAD has {n_nan} NaN")
        present = set(np.unique(lab))
        if present != {"bonafide", "spoof"}:
            row["asserts"].append(f"MLAAD labels present = {sorted(present)}")
        e_full = eer_pct(lab, sc)
        if e_full >= 50.0:
            row["asserts"].append(f"MLAAD EER {e_full:.3f} >= 50%")

        # ---- MLAAD v10, balanced pool (invariance check) ----------------
        if bal.exists():
            blab, bsc = read_v10_balanced(bal)
            if bsc.size != MLAAD_BALANCED_ROWS:
                row["asserts"].append(
                    f"balanced row count {bsc.size} != {MLAAD_BALANCED_ROWS}")
            n_b = int((blab == "bonafide").sum())
            n_s = int((blab == "spoof").sum())
            if n_b != n_s:
                row["asserts"].append(f"balanced not 50/50: {n_b} vs {n_s}")
            e_bal = eer_pct(blab, bsc)
            gap = abs(e_full - e_bal)
            if gap > BALANCE_TOL_PP:
                row["asserts"].append(
                    f"FULL-vs-BALANCED gap {gap:.3f}pp > {BALANCE_TOL_PP}pp")
        else:
            e_bal, gap = None, None
            row["asserts"].append("no balanced/ file")

        row["datasets"]["MLAAD"] = {
            "eer": e_full,
            "eer_balanced": e_bal,
            "balance_gap": gap,
            "n": int(sc.size),
            "n_bonafide": int((lab == "bonafide").sum()),
            "n_spoof": int((lab == "spoof").sum()),
            "nan_frac": 0.0,
        }
        pooled_labels.append(lab)
        pooled_scores.append(sc)

        # ---- Mean and Pooled -------------------------------------------
        eers = [row["datasets"][c]["eer"] for c, _ in DATASETS
                if row["datasets"].get(c) and row["datasets"][c]["eer"] is not None]
        complete = len(eers) == len(DATASETS)
        row["mean"] = float(np.mean(eers)) if complete else None
        if not complete:
            problems.append(f"{disp}: only {len(eers)}/{len(DATASETS)} dataset "
                            f"columns usable -- Mean and Pooled are [TODO-verify]")

        # Pooled re-pools raw scores across every dataset and computes EER once.
        # It is NOT an algebraic patch of the old pooled value.  A row missing
        # any column cannot be pooled honestly, so it is withheld too.
        if complete:
            pl = np.concatenate(pooled_labels)
            ps = np.concatenate(pooled_scores)
            row["pooled"] = eer_pct(pl, ps)
            row["pooled_n"] = int(ps.size)
        else:
            row["pooled"] = None
            row["pooled_n"] = None

        for a in row["asserts"]:
            problems.append(f"{disp}: {a}")
        results[disp] = row
        fmt = lambda v: "TODO" if v is None else f"{v:.3f}"
        print(f"{disp:20s} MLAAD full={e_full:7.3f} bal={fmt(e_bal):>7s} "
              f"mean={fmt(row['mean']):>7s} pooled={fmt(row['pooled']):>7s}",
              flush=True)

    # ---- bolding: best per dataset column, top-5 in Mean/Pooled ---------
    bold = {"columns": {}, "mean_top5": [], "pooled_top5": []}
    for col, _ in DATASETS:
        cand = [(d, r["datasets"][col]["eer"]) for d, r in results.items()
                if r["datasets"].get(col)
                and r["datasets"][col]["eer"] is not None]
        if cand:
            bold["columns"][col] = min(cand, key=lambda t: t[1])[0]
    for key in ("mean", "pooled"):
        cand = [(d, r[key]) for d, r in results.items() if r.get(key) is not None]
        bold[f"{key}_top5"] = [d for d, _ in sorted(cand, key=lambda t: t[1])[:5]]

    # ---- regression gate: untouched columns must reproduce the paper ----
    repro = []
    for disp, r in results.items():
        for i, (col, _) in enumerate(DATASETS):
            if col in CHANGED_COLS:
                continue
            cell = r["datasets"].get(col)
            if cell is None or cell.get("eer") is None:
                continue  # withheld [TODO-verify] cell, nothing to check
            d = abs(cell["eer"] - PUBLISHED[disp][i])
            if d > REPRO_TOL:
                repro.append(f"{disp}/{col}: published {PUBLISHED[disp][i]:.3f} "
                             f"vs recomputed {cell['eer']:.3f} (d={d:.3f})")
    print("\n=== REPRODUCTION GATE (untouched columns) ===")
    if repro:
        for x in repro:
            print("  FAIL " + x)
        print("  -> recomputed MLAAD/Mean/Pooled are NOT trustworthy")
    else:
        print("  PASS: every untouched published cell reproduced within "
              f"{REPRO_TOL} pp")

    payload = {"results": results, "bold": bold, "problems": problems,
               "reproduction_failures": repro}
    (out_dir / "main_results.json").write_text(json.dumps(payload, indent=2))

    print("\n=== PROBLEMS / ASSERT FAILURES ===")
    for p in problems:
        print("  " + p)
    if not problems:
        print("  none")
    print(f"\nWrote {out_dir / 'main_results.json'}")


if __name__ == "__main__":
    main()
