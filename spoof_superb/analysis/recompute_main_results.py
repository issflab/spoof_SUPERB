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

Internal consistency check (not a comparison against anything)
-------------------------------------------------------------
EER is computed from class-conditional error rates and is therefore invariant
to the bonafide:spoof ratio.  Each model's MLAAD EER is computed on BOTH the
full 1,040,006-row pool and the 912,000-row `balanced/` pool; they must agree
within 0.2 percentage points.  A larger gap indicates a parsing, labelling or
polarity bug and is reported as a FAIL rather than silently resolved.

This is a self-check on ONE run: it needs no second tree and no published
value, so it stays here.  Checking these numbers against a reference does not:
that is `python -m spoof_superb.verification analysis`, which compares this
script's CSV against the published one and grades on whether the paper's
claims survive.  This script used to carry a gate that compared its own output
against a dict of published values -- an analysis marking its own homework
against numbers it was simultaneously trying to reproduce, with no way to tell
"the code changed" from "the scores changed".

Outputs
-------
    main_results_table.csv   the paper's table, ready to read: the 19 SSL rows
                             in table order, the paper's columns, `*` on the
                             best in a column and on the Mean top five
    main_results.json        every computed row, with row counts, NaN
                             fractions and assertion results

Usage
-----
    python -m spoof_superb.analysis.recompute_main_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from spoof_superb import REPO_ROOT

from spoof_superb.core.metrics import compute_eer  # type: ignore

from spoof_superb.config import cfg
from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import (column_key, mlaad_pool_paths,
                                         score_path)

# Two columns are not the obvious single file, and the reasons are worth
# keeping now that the layouts they contrast with are gone:
#
#   ASVLD  ONE file, scored from the combined protocol built by
#          data.prep.build_protocols asvld, which already pools every condition
#          (2,065,873 rows). The retired legacy tree needed two files pooled --
#          the noise/reverb/resample set plus a separate Recompression re-run --
#          and reading only the first silently reproduced a pre-6bf39a0 column.
#
#   MLAAD  TWO single-class corpora pooled at read time via
#          scorepath.mlaad_pool_paths: 456,000 MLAAD v10 spoof rows and 584,006
#          M-AILABS bonafide rows, 1,040,006 together. Keeping them separate is
#          what lets either be counted on its own. (P8.)
#
#   DFEval24  reads deepfake_eval_2024_segmented -- every 4 s window of every
#          recording, 56,481 trials -- resolved through scorepath.column_key.
#          See the note there: this is a different measurement from the
#          unsegmented column an earlier draft printed, not a corrected one.
def column_paths(scores_root, dataset, slug, system="linear_head"):
    """Every score file that makes up one (dataset, model) cell, in pool order.

    `system` is "linear_head" for every SSL row. The two non-SSL reference rows
    pass their own system name, which is what puts them under raw/non_ssl/ with
    the system as the filename -- see core.scorepath.
    """
    return [Path(score_path(system, column_key(dataset), slug,
                            scores_root=scores_root))]


def mlaad_paths(scores_root, slug, system="linear_head"):
    """(pool_paths, balanced_path) for the MLAAD column.

    `pool_paths` is a list: the column is assembled from the two single-class
    corpora that compose it; see `scorepath.mlaad_pool_paths`. `balanced_path`
    is always None -- the pre-built 50/50 pool existed only in the retired
    legacy tree, and the invariance check subsamples instead, which is a
    stronger reference anyway (see `balanced_subsample`).
    """
    pool = [Path(p) for p in mlaad_pool_paths(slug, scores_root=scores_root,
                                              system=system)]
    return pool, None


def balanced_subsample(labels, scores, seed=0):
    """Down-sample the majority class to the minority size, deterministically.

    The published contract is that EER is invariant to the bonafide:spoof ratio,
    and the retired legacy tree checked it against a `balanced/` file built
    once, by hand, whose own provenance was never recorded. Constructing the
    balanced pool here tests the same property against a stronger reference: an
    exact 50/50 draw
    from the very rows the full-pool EER was computed on, so a disagreement can
    only be the estimator, never a difference in what was sampled.

    Fixed seed, so the check is reproducible run to run.
    """
    rng = np.random.default_rng(seed)
    idx_b = np.flatnonzero(labels == "bonafide")
    idx_s = np.flatnonzero(labels == "spoof")
    n = min(idx_b.size, idx_s.size)
    keep = np.concatenate([rng.choice(idx_b, n, replace=False),
                           rng.choice(idx_s, n, replace=False)])
    return labels[keep], scores[keep]

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
#: This list is now a RECORD of what the legacy tree had, not the rule. A cell
#: is withheld because its own scores are NaN-corrupted, measured per run --
#: see NAN_WITHHOLD_FRAC. Keeping the hardcoded list as the rule meant the v3
#: tree, where these same cells are 0.00% NaN because the re-scoring fixed them,
#: still had perfectly good numbers withheld.
NAN_CORRUPT = {
    "Mockingjay":      ["ASVLD", "SpoofCeleb"],
    "Mockingjay-960h": ["ASVLD", "SpoofCeleb"],
    "TERA":            ["ASVLD", "SpoofCeleb"],
}

#: A column is withheld as [TODO-verify] when this fraction of its scores is
#: NaN. Not a tuning knob: on the legacy tree the two populations are five
#: orders of magnitude apart -- the six corrupted cells are 22.78%-23.50% NaN,
#: and the only other cell with any NaN at all (NPC/SpoofCeleb) is 0.0011%,
#: which is a handful of rows and does not move an EER. Any threshold between
#: them selects the same six cells.
NAN_WITHHOLD_FRAC = 0.01

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

#: The table's top block: the two non-SSL reference systems, in printed order.
#: They are systems, not upstreams, so they resolve through a different path --
#: raw/non_ssl/{dataset}/{system}.txt -- and `system` is threaded through the
#: path helpers for them alone.
NON_SSL_MODELS = [
    ("LFCC-GMM", "lfcc_gmm"),
    ("AASIST",   "aasist_raw"),
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


def write_table_csv(results, bold, path):
    """The main results table as CSV, in the shape tab:results_main prints it.

    Columns are the paper's, in the paper's order, and rows are the paper's 19
    SSL models in table order -- so a cell here can be read straight against the
    published table without transcribing anything.

    The top block is the two non-SSL reference systems, exactly as the table
    prints them, then the SSL rows. `*` follows the caption: it marks the best
    SSL MODEL in a column, so the reference rows can carry a lower number
    without a star -- that is the comparison the table is making, not a
    formatting slip.

    `MODELS` here is deliberately WIDER than the paper's roster -- it carries
    FBANK and the non-960h Mockingjay because the regression baseline tracks
    them, and the gate guards more columns than the paper prints. Those rows
    stay in main_results.json and out of this file.

    A withheld cell is written as TODO, not blank: the distinction between "we
    did not compute this" and "this is zero" has to survive the CSV.
    """
    import csv as _csv
    from spoof_superb.scoring.models import paper_table_rows

    cols = [c for c, _ in DATASETS]
    fields = ["SSL Model"] + cols + ["Mean"]

    def cell(value, is_best):
        if value is None:
            return "TODO"
        return f"{value:.3f}*" if is_best else f"{value:.3f}"

    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(fields)
        order = [d for d, _ in NON_SSL_MODELS] + list(paper_table_rows())
        for name in order:
            row = results.get(name)
            if row is None:
                continue
            out = [name]
            for col in cols:
                c = row["datasets"].get(col)
                out.append(cell(c["eer"] if c else None,
                                bold["columns"].get(col) == name))
            out.append(cell(row.get("mean"), name in bold["mean_top5"]))
            w.writerow(out)
    return path


def default_out_dir(name):
    """Where an analysis writes, unless --out_dir says otherwise.

    `analysis_root` in configs/paths.yaml when set, {bench_root}/analysiss/ when not.
    """
    root = cfg.analysis_dir
    return os.path.join(root, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None,
                    help="default: analysis_root/main_results, or the repo's outputs/")
    ap.add_argument("--scores_root", default=None,
                    help="score tree to read (default: the configured scores_root)")
    ap.add_argument("--roster", default="paper", choices=("paper", "baseline"),
                    help="which models to compute. 'paper' is the 19 rows the "
                         "results table prints. 'baseline' adds FBANK and the "
                         "non-960h Mockingjay, which the regression baseline "
                         "tracks but the paper does not print -- the gate needs "
                         "them, a report does not.")
    args = ap.parse_args()
    scores_root = args.scores_root or cfg.scores_root
    print(f"reading {scores_root}", flush=True)
    out_dir = Path(args.out_dir or default_out_dir("main_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}   # display name -> dict
    problems = []

    if args.roster == "paper":
        from spoof_superb.scoring.models import paper_table_rows
        wanted = set(paper_table_rows())
        # The two non-SSL reference systems first, as the table prints them.
        roster = ([(d, s, s) for d, s in NON_SSL_MODELS]
                  + [(d, s, "linear_head") for d, s in MODELS if d in wanted])
    else:
        # The gate's superset: the rows the regression baseline tracks. It does
        # not include the non-SSL systems, which were never in that baseline.
        roster = [(d, s, "linear_head") for d, s in MODELS]
    print(f"roster     {args.roster} ({len(roster)} rows)")
    print()
    print(f"{'model':<20}" + "".join(c[:8].rjust(9) for c, _ in DATASETS)
          + "Mean".rjust(9))
    print("-" * (20 + 9 * (len(DATASETS) + 1)), flush=True)

    for disp, slug, system in roster:
        row = {"slug": slug, "system": system, "datasets": {},
               "asserts": [], "sources": {}}
        pooled_labels, pooled_scores = [], []

        # ---- legacy datasets -------------------------------------------
        # A model this tree never scored is reported once. Ten MISSING lines
        # per absent model buried the real problems in the v3 run's output.
        present = [c for c, prefix in DATASETS if prefix is not None
                   and all(p.exists() for p in
                           column_paths(scores_root, prefix, slug, system))]
        if not present:
            problems.append(f"{disp}: not scored on this tree -- no columns")
            results[disp] = row
            print(f"{disp:<20}{'not scored on this tree':>{9 * (len(DATASETS) + 1)}}",
                  flush=True)
            continue
        for col, prefix in DATASETS:
            if prefix is None:
                continue
            key = column_key(prefix)
            if key != prefix:
                row["sources"][col] = key
            paths = column_paths(scores_root, prefix, slug, system)
            missing = [p for p in paths if not p.exists()]
            if missing:
                row["datasets"][col] = None
                problems.append(f"{disp}: MISSING {col} file {missing[0]}")
                continue
            chunks = [read_legacy(p) for p in paths]
            lab = np.concatenate([c[0] for c in chunks])
            sc = np.concatenate([c[1] for c in chunks])
            nf = nan_frac(sc)
            if nf > 0:
                row["asserts"].append(f"{col}: NaN fraction {100*nf:.3f}%")
            if nf > NAN_WITHHOLD_FRAC:
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
        pool, bal = mlaad_paths(scores_root, slug, system)
        missing = [p for p in pool if not p.exists()]
        if missing:
            row["datasets"]["MLAAD"] = None
            problems.append(f"{disp}: MISSING MLAAD v10 file {missing[0].name} "
                            f"-- no recomputed MLAAD/Mean/Pooled possible")
            results[disp] = row
            continue
        if len(pool) > 1:
            row["sources"]["MLAAD"] = " + ".join(p.parent.name for p in pool)

        _utt, lab, sc = read_scored(pool)
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
        if bal is not None and bal.exists():
            blab, bsc = read_v10_balanced(bal)
            row["sources"]["MLAAD_balanced"] = "balanced/ file"
            if bsc.size != MLAAD_BALANCED_ROWS:
                row["asserts"].append(
                    f"balanced row count {bsc.size} != {MLAAD_BALANCED_ROWS}")
            n_b = int((blab == "bonafide").sum())
            n_s = int((blab == "spoof").sum())
            if n_b != n_s:
                row["asserts"].append(f"balanced not 50/50: {n_b} vs {n_s}")
        else:
            # No balanced/ file: build the 50/50 pool from the rows just read.
            blab, bsc = balanced_subsample(lab, sc)
            row["sources"]["MLAAD_balanced"] = "subsampled from the full pool"
        e_bal = eer_pct(blab, bsc)
        gap = abs(e_full - e_bal)
        if gap > BALANCE_TOL_PP:
            row["asserts"].append(
                f"FULL-vs-BALANCED gap {gap:.3f}pp > {BALANCE_TOL_PP}pp")

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
        # Every column, not just MLAAD. This line used to print MLAAD's full
        # and balanced EERs and nothing else -- a leftover from when the script
        # existed only to replace the corrupted MLAAD column. The other nine
        # were computed and written to the JSON and CSV, but never shown, so a
        # run looked like it had scored one dataset.
        #
        # The full-vs-balanced pair is not lost: a gap beyond tolerance is
        # already recorded as an assert and printed under PROBLEMS.
        fmt = lambda v: "TODO".rjust(9) if v is None else f"{v:9.3f}"
        print(f"{disp:<20}"
              + "".join(fmt(row["datasets"][c]["eer"]
                            if row["datasets"].get(c) else None)
                        for c, _ in DATASETS)
              + fmt(row["mean"]), flush=True)

    # ---- bolding: best per dataset column, top-5 in Mean/Pooled ---------
    # "Bold marks the best SSL MODEL in each dataset column and the top five
    # under Mean EER" -- the caption. The two non-SSL reference rows are the
    # thing being compared against, so they are excluded from the comparison.
    non_ssl = {d for d, _ in NON_SSL_MODELS}
    bold = {"columns": {}, "mean_top5": [], "pooled_top5": []}
    for col, _ in DATASETS:
        cand = [(d, r["datasets"][col]["eer"]) for d, r in results.items()
                if d not in non_ssl and r["datasets"].get(col)
                and r["datasets"][col]["eer"] is not None]
        if cand:
            bold["columns"][col] = min(cand, key=lambda t: t[1])[0]
    for key in ("mean", "pooled"):
        cand = [(d, r[key]) for d, r in results.items()
                if d not in non_ssl and r.get(key) is not None]
        bold[f"{key}_top5"] = [d for d, _ in sorted(cand, key=lambda t: t[1])[:5]]

    payload = {"results": results, "bold": bold, "problems": problems}
    (out_dir / "main_results.json").write_text(json.dumps(payload, indent=2))
    csv_path = write_table_csv(results, bold, out_dir / "main_results_table.csv")

    print("\n=== PROBLEMS / ASSERT FAILURES ===")
    for p in problems:
        print("  " + p)
    if not problems:
        print("  none")
    print(f"\nWrote {out_dir / 'main_results.json'}   (every computed row)")
    print(f"Wrote {csv_path}   (the paper's table: 19 SSL rows, "
          f"* marks the best in a column / the Mean top five)")


if __name__ == "__main__":
    main()
