"""Acoustic degradation with composition AND coverage matched (Section 5.2).

    python -m spoof_superb.analysis.acoustic_degradation \
        --out_dir outputs/degradation

Why the pools are matched
-------------------------
An earlier version of this analysis pooled each condition from partitions of
four corpora, keeping the corpora it did not degrade. Two things then varied
between the Baseline and a degraded condition, on top of the degradation
itself:

  COMPOSITION  which corpora make up the pool, in what proportion. ASVLD
               applies 15 noise settings to every ASV19 utterance, so the
               Additive Noise pool is 80.1% ASV19-derived against Baseline's
               21.2%, and 12.9% ASV5 against Baseline's 51.0%.

  COVERAGE     what fraction of the pool carries a degradation at all. Noise
               degrades 21.2% of its pool; Codec degrades 92.3% of its pool,
               because it swaps three of the four corpora at once.

That version has been removed; this module replaces it. Re-mixing the CLEAN
partitions in the confounded proportions, with no degradation applied at all,
moved the pooled EER by -18% to -44%, which is the size of the artifact it
reported as an effect.

Neither is held fixed, so a dEER computed against the Baseline mixes the
degradation with the re-weighting. Measured on the v3 tree, the composition
term alone moves the noise-column EER by -18% to -44% before any degradation
is applied, which is what makes eight models appear to IMPROVE under noise.

What this module does instead
-----------------------------
ONE corpus is degraded per cell, and every corpus carries a fixed share of the
metric regardless of how many rows it happens to contribute:

  1. Weights, not row counts. Every row of corpus c is weighted 1/n_c, so each
     of the four corpora contributes exactly 1/4 of the pooled EER. Nothing is
     subsampled and nothing is replicated -- see `weighted_eer`.
  2. One corpus per cell. A cell replaces corpus c's CLEAN partition with c's
     DEGRADED partition for one variant, and leaves the other three corpora
     exactly as the Baseline has them.
  3. The degraded partition inherits c's share. Its rows are weighted 1/n_v,
     so corpus c still contributes 1/4 whatever its degraded row count is.
  4. Average over variants (Option B). Each variant is scored separately and
     the EERs are averaged, so the spread across settings is measurable and a
     single global threshold is never asked to serve heterogeneous variants.

Composition is then 25/25/25/25 in every cell including the Baseline, and
coverage is 25% in every degraded cell. The only thing that differs between a
cell and the Baseline is the degradation.

Known limitation, deliberately not corrected
--------------------------------------------
`1/n_c` equalises each corpus's share of the ROWS, not of each CLASS. The
corpora are 79.5%-96.3% spoof, so ASV21 DF -- 2,513 bonafide trials out of
67,981 -- ends up with 8.3% of the bonafide mass and 27.1% of the spoof mass.
Since FRR is computed from bonafide scores alone, DF's degradation is
under-represented on the FRR side, and Codec:ASV21DF reads about 4.8 points
lower here than under per-class weighting (w[c,class] = 1/n[c,class]).

Per-class weighting was measured and rejected: it cuts the bonafide effective
sample size from 28,081 to 16,990 by giving two ~2,500-trial bonafide sets a
quarter of the FRR side each. Every conclusion is the same under both
(Kendall tau 0.833 on cell severity, same top-2 models), and the one column
that is sensitive -- Codec:ASV21DF -- is near zero under both. Do not make a
directional claim about that cell from this matrix.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.analysis.conditions import CLEAN, condition_of
from spoof_superb.config import cfg
from spoof_superb.scoring.models import (display_by_slug, paper_models,
                                         paper_table_rows)

#: The four corpora, in the order the Baseline pools them.
CORPORA = ["ASV19", "ASV21LA", "ASV21DF", "ASV5"]

#: Where each corpus's clean partition is read from, and how it is identified.
_CLEAN_SRC = {
    "ASV19":   ("asvspoof2019_la_eval", None),
    "ASV21LA": ("asvspoof2021_la", "asvspoof2021_LA"),
    "ASV21DF": ("asvspoof2021_df", "asvspoof2021_DF"),
    "ASV5":    ("asvspoof5", "asvspoof5"),
}

#: ASVLD's utt_id suffix -> the degradation family, as tab:acoustic_degradation
#: names them. ASVLD is the only corpus that carries all five.
_LD_FAMILY = [
    (re.compile(r"^(babble|cafe|street|volvo|white)_\d+$"), "Noise"),
    (re.compile(r"^RT_\d+_\d+$"),                           "Reverb"),
    (re.compile(r"^resample_\d+$"),                         "Resampling"),
    (re.compile(r"^recompression_\d+k$"),                   "Codec"),
    (re.compile(r"^lpf_\d+$"),                              "Channel"),
]
_LD_ID = re.compile(r"^LA_E_\d+_(?P<cond>.+)$")

#: (condition, corpus) -> column name. One corpus degraded per cell; a corpus
#: appears only where that corpus actually contains that degradation.
CELLS = [
    ("Codec", "ASV19",   "Codec:ASVLD"),
    ("Codec", "ASV21DF", "Codec:ASV21DF"),
    ("Codec", "ASV5",    "Codec:ASV5"),
    ("Noise", "ASV19",   "Noise:ASVLD"),
    ("Resampling", "ASV19", "Resampling:ASVLD"),
    ("Reverb", "ASV19",  "Reverb:ASVLD"),
    ("Channel", "ASV19", "Channel:ASVLD"),
    ("Channel", "ASV21LA", "Channel:ASV21LA"),
    ("Channel", "ASV5",  "Channel:ASV5"),
]


def read_scores(path):
    """(utt_ids, labels, scores) from a score file, as parallel arrays."""
    u, l, s = [], [], []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 4:
                u.append(f[0]); l.append(f[2]); s.append(f[3])
    return np.array(u), np.array(l), np.array(s, dtype=float)


def by_class(labels, scores):
    """(bonafide scores, spoof scores) with NaN scores dropped."""
    ok = ~np.isnan(scores)
    labels, scores = labels[ok], scores[ok]
    return scores[labels == "bonafide"], scores[labels == "spoof"]


def weighted_eer(bona, bona_w, spoof, spoof_w):
    """EER with a per-trial weight, swept over every observed score.

    FRR is the bonafide weight at or below the threshold as a fraction of all
    bonafide weight; FAR is the spoof weight above it as a fraction of all
    spoof weight. Both are normalised within their own class, so the EER does
    not depend on the bonafide:spoof ratio -- only on how each class is
    composed. That is why `1/n_c` weighting is what sets a corpus's influence,
    and why no resampling is needed to achieve it.
    """
    scores = np.concatenate([bona, spoof])
    is_spoof = np.concatenate([np.zeros(len(bona)), np.ones(len(spoof))])
    w = np.concatenate([bona_w, spoof_w])
    order = np.argsort(scores, kind="mergesort")
    is_spoof, w = is_spoof[order], w[order]
    total_b, total_s = bona_w.sum(), spoof_w.sum()
    frr = np.cumsum(np.where(is_spoof == 0, w, 0.0)) / total_b
    far = (total_s - np.cumsum(np.where(is_spoof == 1, w, 0.0))) / total_s
    i = int(np.nanargmin(np.abs(frr - far)))
    return 100.0 * (frr[i] + far[i]) / 2.0


def score_pool(members):
    """Weighted EER over a pool of four partitions, one per corpus slot.

    `members` is [(bona, spoof)] * 4. Each partition's rows are weighted
    1/n so that every corpus contributes an equal share of the metric.
    """
    bo, bw, sp, sw = [], [], [], []
    for bona, spoof in members:
        if len(bona) == 0 or len(spoof) == 0:
            return None
        w = 1.0 / (len(bona) + len(spoof))
        bo.append(bona); bw.append(np.full(len(bona), w))
        sp.append(spoof); sw.append(np.full(len(spoof), w))
    return weighted_eer(np.concatenate(bo), np.concatenate(bw),
                        np.concatenate(sp), np.concatenate(sw))


def load_model(slug, scores_root):
    """Every partition this analysis needs, for one model."""
    root = Path(scores_root) / "raw" / "linear_head"
    P = {"clean": {}, "deg": {}}

    for corpus, (dataset, protocol) in _CLEAN_SRC.items():
        u, l, s = read_scores(root / dataset / f"{slug}.txt")
        if protocol is None:                       # ASV19 eval: all of it is clean
            P["clean"][corpus] = by_class(l, s)
            continue
        cond = np.array([condition_of(protocol)(x) for x in u])
        keep = cond == CLEAN[protocol]
        P["clean"][corpus] = by_class(l[keep], s[keep])
        P[f"{corpus}_cond"] = (cond, l, s)

    u, l, s = read_scores(root / "asvspoof_ld" / f"{slug}.txt")
    cond = np.array([_LD_ID.match(x).group("cond") for x in u])
    for c in np.unique(cond):
        family = next(f for pat, f in _LD_FAMILY if pat.match(c))
        keep = cond == c
        P["deg"].setdefault((family, "ASV19"), {})[c] = by_class(l[keep], s[keep])

    def add(corpus, family, keys):
        cond, l, s = P[f"{corpus}_cond"]
        for k in keys:
            keep = cond == k
            P["deg"].setdefault((family, corpus), {})[k] = by_class(l[keep], s[keep])

    cond_df = P["ASV21DF_cond"][0]
    add("ASV21DF", "Codec",
        sorted(set(cond_df[cond_df != CLEAN["asvspoof2021_DF"]])))
    cond_la = P["ASV21LA_cond"][0]
    add("ASV21LA", "Channel",
        sorted(set(cond_la[cond_la != CLEAN["asvspoof2021_LA"]])))
    cond_a5 = P["ASV5_cond"][0]
    add("ASV5", "Codec",
        sorted(k for k in set(cond_a5) if k not in (CLEAN["asvspoof5"], "C11")))
    add("ASV5", "Channel", ["C11"])
    return P


def analyse(slug, scores_root):
    """Baseline plus every cell, for one model."""
    P = load_model(slug, scores_root)
    clean = P["clean"]
    baseline = score_pool([clean[c] for c in CORPORA])
    cells, per_variant = {}, []
    for condition, corpus, column in CELLS:
        variants = P["deg"].get((condition, corpus), {})
        if not variants:
            continue
        eers = []
        for name, part in sorted(variants.items()):
            members = [part if c == corpus else clean[c] for c in CORPORA]
            e = score_pool(members)
            if e is None:
                continue
            eers.append(e)
            per_variant.append({"Model": slug, "condition": condition,
                                "corpus": corpus, "column": column,
                                "variant": name, "EER": e})
        cells[column] = {
            "mean": float(np.mean(eers)),
            "sd": float(np.std(eers, ddof=1)) if len(eers) > 1 else 0.0,
            "min": float(np.min(eers)), "max": float(np.max(eers)),
            "n": len(eers),
            "delta": (float(np.mean(eers)) - baseline) / baseline * 100.0,
        }
    return baseline, cells, per_variant


def _write(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"  wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.acoustic_degradation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--scores_root", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args(argv)

    scores_root = args.scores_root or cfg.scores_root
    root = cfg.analysis_dir
    out_dir = Path(args.out_dir or os.path.join(root, "degradation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    slugs = args.models or sorted(paper_models())
    names = display_by_slug()
    columns = [col for _, _, col in CELLS]

    eer_rows, sd_rows, delta_rows, spread_rows, variant_rows = [], [], [], [], []
    for slug in slugs:
        baseline, cells, per_variant = analyse(slug, scores_root)
        model = names.get(slug, slug)
        eer_rows.append({"Model": model, "Baseline": baseline,
                         **{c: cells[c]["mean"] for c in columns if c in cells}})
        sd_rows.append({"Model": model,
                        **{c: cells[c]["sd"] for c in columns if c in cells}})
        delta_rows.append({"Model": model,
                           **{c: cells[c]["delta"] for c in columns if c in cells}})
        for c in columns:
            if c in cells:
                spread_rows.append({"Model": model, "column": c, **cells[c]})
        variant_rows.extend(per_variant)
        print(f"{model:<20} baseline {baseline:6.2f}   " +
              "  ".join(f"{c.split(':')[0][:5]}={cells[c]['mean']:5.2f}"
                        for c in columns if c in cells), flush=True)

    order = {n: i for i, n in enumerate(paper_table_rows())}
    for rows in (eer_rows, sd_rows, delta_rows):
        rows.sort(key=lambda r: order.get(r["Model"], len(order)))

    print()
    _write(out_dir / "eer_matrix.csv", ["Model", "Baseline"] + columns, eer_rows)
    _write(out_dir / "eer_sd.csv", ["Model"] + columns, sd_rows)
    _write(out_dir / "deer_matrix.csv", ["Model"] + columns, delta_rows)
    _write(out_dir / "cell_spread.csv",
           ["Model", "column", "mean", "sd", "min", "max", "n", "delta"],
           spread_rows)
    _write(out_dir / "per_variant_eer.csv",
           ["Model", "condition", "corpus", "column", "variant", "EER"],
           variant_rows)
    print(f"\nDone. {len(eer_rows)} models x {len(columns)} cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
