"""Score-file cell measurement: the primitives both tree comparisons share.

A "cell" is one (benchmark column, SSL model) pair. For most columns that is a
single score file; for MLAAD and ASVLD the published column is the POOL of two,
so a cell is a list of paths and not a path.

This module MEASURES. It does not grade -- `verification.scores` applies the
reproduction ladder, `tools.compare_trees` applies its own older vocabulary,
and keeping the verdict out of here is what lets the two coexist without one
quietly redefining the other's terms.

The measurement is built around one asymmetry that a single number cannot
express: two trees disagree either because they SCORED DIFFERENT UTTERANCES or
because they ASSIGNED DIFFERENT SCORES to the same ones. So four EERs are
reported, not one:

    eer_a          A's EER over A's own trials      -- what A would publish
    eer_b          B's EER over B's own trials      -- what B would publish
    eer_a_common   A's EER over the shared trials   -- coverage effect isolated
    eer_b_common   B's EER over the same shared set -- score effect isolated

`eer_a` vs `eer_a_common` is coverage alone. `eer_a_common` vs `eer_b_common`
is the scores alone, trial set held fixed. Only the second can indict a
pipeline, and only it grounds a reproduction claim.
"""

import os
from pathlib import Path

import numpy as np

from spoof_superb.core.metrics import compute_eer
from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import mlaad_pool_paths, score_path
from spoof_superb.verification.verdicts import EXACT_TOL

__all__ = ["DATASETS", "DATASET_KEY_BY_LAYOUT", "cell_paths", "compare_cell",
           "eer_pct", "layout_key", "rewrite_components",
           "strip_absolute_prefix", "EXACT_TOL"]

#: Benchmark columns, as (display name, registry key). Order is the paper's
#: column order, and the reports keep it.
DATASETS = [
    ("ASV19 LA",   "eval_2019"),
    ("ASV21 LA",   "asvspoof2021_LA"),
    ("ASV21 DF",   "asvspoof2021_DF"),
    ("ASV5 Eval",  "asvspoof5"),
    ("ITW",        "wild"),
    ("DFEval24",   "deepfake_eval_2024"),
    ("FF",         "Famous_Figures"),
    ("ASVLD",      "asvspoofLD"),
    ("SpoofCeleb", "spoofceleb"),
    ("MLAAD",      "Multilingual"),
]

#: Legacy has no per-dataset directory convention for linear_head, so its paths
#: are built the way the pre-reorganisation code built them.
LEGACY_ASVLD_EXTRA = ("asvld_rerun", "Recompression",
                      "linear_head_Recompression_{slug}.txt")

#: DFEval24 is a different measurement under v2/v3: every 4 s window rather than
#: one window per recording. Comparing them is meaningful only as coverage.
DATASET_KEY_BY_LAYOUT = {
    "deepfake_eval_2024": {"v2": "deepfake_eval_2024_segmented",
                           "v3": "deepfake_eval_2024_segmented"},
}


def layout_key(dataset, layout):
    """The registry key this benchmark column is stored under, in this layout.

    Only DFEval24 differs, and it differs because the measurement did: v2/v3
    score every 4 s window rather than one window per recording. Enumeration
    and path construction must agree on this or a column silently reports as
    unscored -- which is what happened when only `cell_paths` knew about it.
    """
    return DATASET_KEY_BY_LAYOUT.get(dataset, {}).get(layout, dataset)


def cell_paths(layout, root, dataset, slug):
    """Every score file composing one (dataset, model) cell, in pool order."""
    if dataset == "Multilingual":
        return [Path(p) for p in mlaad_pool_paths(slug, scores_root=root,
                                                  layout=layout)]
    if layout == "legacy":
        paths = [Path(root) / "linear_head" / f"linear_head_{dataset}_{slug}.txt"]
        if dataset == "asvspoofLD":
            a, b, c = LEGACY_ASVLD_EXTRA
            paths.append(Path(root) / a / b / c.format(slug=slug))
        return paths
    key = layout_key(dataset, layout)
    return [Path(score_path("linear_head", key, slug, scores_root=root,
                            layout=layout))]


def eer_pct(labels, scores):
    """EER in percent, or None if either class is empty or all-NaN."""
    finite = ~np.isnan(scores)
    labels, scores = labels[finite], scores[finite]
    bona = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    if bona.size == 0 or spoof.size == 0:
        return None
    return 100.0 * compute_eer(bona, spoof)[0]


def strip_absolute_prefix(utts):
    """Make ids corpus-relative by removing a shared absolute directory prefix.

    Famous Figures score files in the old tree record ids as ABSOLUTE paths
    (`/nfs/turbo/.../famousfigures/Anthony_Blinken/...`) while the new tree
    records them relative to the corpus root. They are the same utterances, but
    compared literally the intersection is empty -- so all 19 FF cells reported
    "no overlap", which reads as "these trees share no trials" when the truth is
    "one of them wrote the mount point into every id".

    Narrow on purpose: only a prefix that is ABSOLUTE and common to every id in
    the file is removed. Relative ids are never touched, so a genuine coverage
    difference can never be normalised away into a false match.

    Returns (ids, prefix_removed).
    """
    if utts.size == 0 or not str(utts[0]).startswith("/"):
        return utts, ""
    prefix = os.path.commonprefix(utts.tolist())
    prefix = prefix[:prefix.rindex("/") + 1] if "/" in prefix else ""
    if not prefix.startswith("/"):
        return utts, ""
    n = len(prefix)
    return np.asarray([u[n:] for u in utts.tolist()], dtype=object), prefix


def rewrite_components(utts, rules):
    """Rename whole path components in every id.  `rules` is {old: new}.

    Kept out of `strip_absolute_prefix` and off by default, because unlike a
    shared absolute prefix this cannot be detected -- it has to be asserted. The
    case it exists for is Famous Figures, where the old tree names the bonafide
    directory ``-`` and the new one names it ``Bonafide``:

        Anthony_Blinken/-/Anthony_Blinken_00001.wav
        Anthony_Blinken/Bonafide/Anthony_Blinken_00001.wav

    Same utterance, different convention. Without the rewrite the bonafide half
    of every FF cell fails to align, the intersection is all-spoof, and the cell
    reports as "not comparable" -- true, but it hides that the spoof half
    matched perfectly.

    Requiring it on the command line keeps the claim "these are the same
    utterances" a stated assumption rather than a silent one.
    """
    if not rules:
        return utts
    return np.asarray(
        ["/".join(rules.get(c, c) for c in u.split("/")) for u in utts.tolist()],
        dtype=object)


def _spearman(x, y):
    """Rank correlation, as Pearson over ranks. None when a side is constant."""
    if x.size < 2:
        return None
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def compare_cell(paths_a, paths_b, rewrite_a=None, rewrite_b=None):
    """Both comparison levels for one cell.  Returns a dict of measurements.

    Alignment is by utt_id. A duplicated utt_id within one tree would make the
    intersection ill-defined, so it is counted and reported rather than silently
    resolved by last-write-wins.

    No verdict is assigned here; callers grade the returned row.
    """
    missing_a = [p for p in paths_a if not p.exists()]
    missing_b = [p for p in paths_b if not p.exists()]
    if missing_a or missing_b:
        return {"status": "missing_a" if missing_a else "missing_b",
                "missing": str((missing_a or missing_b)[0])}

    utt_a, lab_a, sc_a = read_scored(paths_a)
    utt_b, lab_b, sc_b = read_scored(paths_b)
    utt_a, pre_a = strip_absolute_prefix(utt_a)
    utt_b, pre_b = strip_absolute_prefix(utt_b)
    utt_a = rewrite_components(utt_a, rewrite_a)
    utt_b = rewrite_components(utt_b, rewrite_b)

    dup_a = utt_a.size - len(set(utt_a.tolist()))
    dup_b = utt_b.size - len(set(utt_b.tolist()))

    # Index A, then walk B once.  Keeps one tree's worth of ids in memory
    # instead of two, which matters for the 2M-row ASVLD cells.
    index_a = {u: i for i, u in enumerate(utt_a.tolist())}
    idx_a, idx_b = [], []
    for j, u in enumerate(utt_b.tolist()):
        i = index_a.get(u)
        if i is not None:
            idx_a.append(i)
            idx_b.append(j)
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)

    row = {
        "status": "ok",
        "n_a": int(utt_a.size),
        "n_b": int(utt_b.size),
        "n_common": int(idx_a.size),
        "n_only_a": int(utt_a.size - idx_a.size),
        "n_only_b": int(utt_b.size - idx_b.size),
        "dup_a": int(dup_a),
        "dup_b": int(dup_b),
        "prefix_a": pre_a,
        "prefix_b": pre_b,
        "nan_a": int(np.isnan(sc_a).sum()),
        "nan_b": int(np.isnan(sc_b).sum()),
        "eer_a": eer_pct(lab_a, sc_a),
        "eer_b": eer_pct(lab_b, sc_b),
    }

    if idx_a.size == 0:
        row.update({"label_mismatch": None, "max_abs_diff": None, "corr": None,
                    "spearman": None, "frac_exact": None,
                    "eer_a_common": None, "eer_b_common": None})
        return row

    ca_lab, cb_lab = lab_a[idx_a], lab_b[idx_b]
    ca_sc, cb_sc = sc_a[idx_a], sc_b[idx_b]
    row["label_mismatch"] = int((ca_lab != cb_lab).sum())

    both = ~np.isnan(ca_sc) & ~np.isnan(cb_sc)
    x, y = ca_sc[both], cb_sc[both]
    row["n_both_finite"] = int(x.size)
    if x.size:
        d = np.abs(x - y)
        row["max_abs_diff"] = float(d.max())
        row["frac_exact"] = float((d <= EXACT_TOL).mean())
        row["mean_offset"] = float((y - x).mean())
        row["offset_std"] = float((y - x).std())
        # A constant column has zero variance and no defined correlation; that
        # is a real condition (a dead model), not an error.
        row["corr"] = (float(np.corrcoef(x, y)[0, 1])
                       if x.std() > 0 and y.std() > 0 else None)
        row["spearman"] = _spearman(x, y)
    else:
        row.update({"max_abs_diff": None, "frac_exact": None, "corr": None,
                    "spearman": None, "mean_offset": None, "offset_std": None})

    row["eer_a_common"] = eer_pct(ca_lab, ca_sc)
    row["eer_b_common"] = eer_pct(cb_lab, cb_sc)
    return row
