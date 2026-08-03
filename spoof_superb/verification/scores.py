"""Level 1 -- verify score files against the published reference.

    python -m spoof_superb.verification scores --manifest
    python -m spoof_superb.verification scores --ref-root /path/to/reference/tree

Two reference modes, because they answer the same question at different prices.

**Manifest mode (default).** `reference/manifest.json` is ~100 KB and lives in
the repo. Per score file it records a sha256, row and class counts, the
non-finite count, the EER, and a DIGEST OF THE SORTED TRIAL LIST. That last one
is what makes offline verification meaningful rather than suggestive: matching
row counts prove nothing (two different 71,237-trial sets are still different
trial sets), but a matching utt_id digest proves the two runs scored exactly the
same utterances, which is the precondition for comparing their EERs at all.

So manifest mode can establish, without downloading 6 GB:

    * the file is byte-identical to the reference          (sha256)
    * or it scored exactly the same trials                 (utt digest)
    * and its EER agrees / disagrees, by how much          (eer)

What it cannot do is separate SENSITIVE from SCORES_DIFFER, because that needs
rank agreement between the two score vectors and the manifest does not carry
per-utterance scores. When an EER disagreement is found, the report says so and
names the command that resolves it.

**Tree mode.** The reference score files are on disk, so every utterance is
compared. This is the full ladder, and it is what a provenance claim should
cite.

Neither mode writes to either tree.
"""

import hashlib
import json

import numpy as np

from spoof_superb.core.metrics import compute_eer
from spoof_superb.core.scorefile import read_scored
from spoof_superb.verification.cells import (DATASETS, cell_paths, compare_cell,
                                             eer_pct, rewrite_components,
                                             strip_absolute_prefix)
from spoof_superb.verification.verdicts import (EER_TOL_PP, NONFINITE_TOL,
                                                RANK_TOL)

__all__ = ["grade_tree_cell", "grade_manifest_cell", "cell_summary",
           "utt_digest", "sha256", "verify_scores"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def utt_digest(utts):
    """A stable fingerprint of a trial set.

    Sorted, so it does not depend on the order rows were written -- two runs
    that scored the same utterances in a different order have the same trial
    set and must not be reported as differing. Hashed rather than stored, so
    the manifest stays ~100 KB instead of ~500 MB.
    """
    h = hashlib.sha256()
    for u in sorted(utts.tolist() if hasattr(utts, "tolist") else utts):
        h.update(u.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def cell_summary(paths):
    """Everything the manifest records about one cell, computed from disk."""
    utts, labels, scores = read_scored(paths)
    utts, _prefix = strip_absolute_prefix(utts)
    finite = ~np.isnan(scores)
    bona = scores[(labels == "bonafide") & finite]
    spoof = scores[(labels == "spoof") & finite]
    eer = (100.0 * compute_eer(bona, spoof)[0]
           if bona.size and spoof.size else None)
    return {
        "n_rows": int(utts.size),
        "n_bonafide": int((labels == "bonafide").sum()),
        "n_spoof": int((labels == "spoof").sum()),
        "n_nonfinite": int((~finite).sum()),
        "eer_percent": None if eer is None else round(float(eer), 6),
        "utt_digest": utt_digest(utts),
        "sha256": [sha256(p) for p in paths],
    }


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

def grade_tree_cell(row):
    """Place a fully measured cell on the reproduction ladder.

    Order matters and is the contract. Label disagreement is checked before any
    score statistic, because comparing scores across disagreeing ground truth
    produces a confident answer to the wrong question. Candidate NaN is checked
    before agreement, because no statistic about an invalid vector means
    anything.
    """
    if row.get("status", "ok") != "ok":
        return "MISSING", row.get("missing", "no candidate file")

    if row["n_common"] == 0:
        return "COVERAGE_DIFFERS", (
            f"no shared utt_ids (reference {row['n_a']}, candidate {row['n_b']})")

    if row.get("label_mismatch"):
        return "LABELS_DIFFER", (
            f"{row['label_mismatch']} shared utt_id(s) carry a different key -- "
            f"the two runs are not scoring the same protocol")

    n_cand = row["n_b"] or 1
    if row["nan_b"] > NONFINITE_TOL * n_cand:
        return "CANDIDATE_INVALID", (
            f"{row['nan_b']} non-finite score(s) "
            f"({row['nan_b'] / n_cand:.2%} > {NONFINITE_TOL:.0%})")

    same_trials = row["n_only_a"] == 0 and row["n_only_b"] == 0

    if row.get("frac_exact") == 1.0 and same_trials:
        return "IDENTICAL", "every shared score is bit-identical"

    if not same_trials:
        return "COVERAGE_DIFFERS", (
            f"{row['n_only_a']} reference-only and {row['n_only_b']} "
            f"candidate-only trial(s); EERs below are on the "
            f"{row['n_common']} shared")

    a, b = row.get("eer_a_common"), row.get("eer_b_common")
    if a is None or b is None:
        return "COVERAGE_DIFFERS", "the shared trials do not contain both classes"

    d = abs(a - b)
    if d <= EER_TOL_PP:
        return "EQUIVALENT", (
            f"EER agrees to {d:.4f} pp on identical trials "
            f"(<= {EER_TOL_PP} pp)")

    sp = row.get("spearman")
    if sp is not None and sp >= RANK_TOL:
        return "SENSITIVE", (
            f"scores agree in rank (spearman {sp:.5f}) but the EER moved "
            f"{d:.3f} pp -- a near-chance operating point where the DET curve "
            f"is flat, not a scoring difference")

    return "SCORES_DIFFER", (
        f"EER moved {d:.3f} pp on identical trials"
        + (f" (spearman {sp:.5f})" if sp is not None else ""))


def grade_manifest_cell(ref, cand):
    """Grade a cell against the manifest entry for it.

    Strictly less discriminating than `grade_tree_cell`: with no per-utterance
    scores there is no rank agreement, so SENSITIVE cannot be distinguished
    from SCORES_DIFFER. The reason string says so rather than guessing.
    """
    if cand is None:
        return "MISSING", "no candidate score file"
    if ref is None:
        return "MISSING", "the reference manifest has no entry for this cell"

    if ref.get("sha256") and ref["sha256"] == cand["sha256"]:
        return "IDENTICAL", "byte-identical to the reference"

    n_cand = cand["n_rows"] or 1
    if cand["n_nonfinite"] > NONFINITE_TOL * n_cand:
        return "CANDIDATE_INVALID", (
            f"{cand['n_nonfinite']} non-finite score(s) "
            f"({cand['n_nonfinite'] / n_cand:.2%} > {NONFINITE_TOL:.0%})")

    if ref.get("utt_digest") and ref["utt_digest"] != cand["utt_digest"]:
        return "COVERAGE_DIFFERS", (
            f"different trial set (reference {ref['n_rows']} rows, candidate "
            f"{cand['n_rows']}); the manifest cannot intersect them -- "
            f"rerun with --ref-root to compare the shared trials")

    if ref.get("eer_percent") is None or cand["eer_percent"] is None:
        return "COVERAGE_DIFFERS", "one side has no EER (a class is missing)"

    d = abs(ref["eer_percent"] - cand["eer_percent"])
    if d <= EER_TOL_PP:
        return "EQUIVALENT", (
            f"same trial set, EER agrees to {d:.4f} pp (<= {EER_TOL_PP} pp)")

    return "SCORES_DIFFER", (
        f"same trial set, EER moved {d:.3f} pp; rank agreement is not in the "
        f"manifest, so this cannot yet be separated from a flat-DET SENSITIVE "
        f"cell -- rerun with --ref-root to decide")


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

def _wanted(datasets):
    if not datasets:
        return DATASETS
    keep = set(datasets)
    return [d for d in DATASETS if d[0] in keep or d[1] in keep]


def verify_scores(candidate_root, candidate_layout, models, datasets=None,
                  ref_root=None, ref_layout=None, manifest_path=None,
                  rewrite_ref=None, rewrite_cand=None, progress=print):
    """One row per (dataset, model). Tree mode if `ref_root`, else manifest."""
    wanted = _wanted(datasets)
    manifest = None
    if not ref_root:
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    rows = []
    for disp, key in wanted:
        for slug in models:
            row = {"dataset": disp, "model": slug,
                   "mode": "tree" if ref_root else "manifest"}
            try:
                cand_paths = cell_paths(candidate_layout, candidate_root, key, slug)
                if ref_root:
                    row.update(compare_cell(
                        cell_paths(ref_layout, ref_root, key, slug), cand_paths,
                        rewrite_ref, rewrite_cand))
                    verdict, reason = grade_tree_cell(row)
                    if (row.get("eer_a_common") is not None
                            and row.get("eer_b_common") is not None):
                        row["d_eer"] = abs(row["eer_a_common"] - row["eer_b_common"])
                else:
                    cand = (cell_summary(cand_paths)
                            if all(p.exists() for p in cand_paths) else None)
                    ref = (manifest.get("cells", {}).get(key, {}).get(slug))
                    verdict, reason = grade_manifest_cell(ref, cand)
                    row.update({
                        "n_a": (ref or {}).get("n_rows"),
                        "n_b": (cand or {}).get("n_rows"),
                        "nan_b": (cand or {}).get("n_nonfinite"),
                        "eer_a": (ref or {}).get("eer_percent"),
                        "eer_b": (cand or {}).get("eer_percent"),
                    })
                    if row["eer_a"] is not None and row["eer_b"] is not None:
                        row["d_eer"] = abs(row["eer_a"] - row["eer_b"])
            except Exception as exc:        # one bad cell must not kill the sweep
                verdict, reason = "ERROR", f"{type(exc).__name__}: {exc}"
            row["verdict"], row["reason"] = verdict, reason
            rows.append(row)
            progress(f"  {disp:<11} {slug:<40} {verdict}")
    return rows
