"""
build_release_manifest.py
-------------------------
Publisher-side tool: index a finished score tree so others can verify against it.

The score files are ~6 GB and do not belong in git. They are published as a
per-file archive (a release asset or a DOI record) and fetched on demand by
`bin/fetch_scores.sh`. This writes the small index that makes that work:

    reference/manifest.json     ~100 KB, versioned in the repo

Two blocks, because two questions are being answered.

`files` -- per score file: relative path, sha256, byte size, row and class
counts, non-finite count, EER, score quantiles. This is what a DOWNLOAD needs:
what exists, where to put it, and whether it arrived intact.

`cells` -- per (benchmark column, model): the pooled row and class counts, the
pooled EER, and a DIGEST OF THE SORTED TRIAL LIST. This is what VERIFICATION
needs, and it is not the same thing. Two published columns are the pool of two
files (MLAAD, ASVLD), so a per-file EER is not the number the paper prints.
And the trial digest is what lets `spoof_superb.verification scores` establish
offline that a candidate scored exactly the same utterances -- matching row
counts prove nothing, since two different 71,237-trial sets are still different
trial sets, and without that precondition comparing two EERs is meaningless.

Nothing else is shipped. Trial lists and score subsamples were considered and
rejected: both are derived from the score files, so committing them would put
two copies of the same information under version control with no record of
which is authoritative -- the exact pattern that produced the duplication in
the score directory.

Run this once when publishing a score tree, not as part of normal work.

Usage
-----
    python -m spoof_superb.tools.build_release_manifest --dry-run
    python -m spoof_superb.tools.build_release_manifest --datasets ITW --limit 2
"""

import argparse
import json
import os
from datetime import date

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.core.metrics import compute_eer
from spoof_superb.core.scorepath import available_frontends
from spoof_superb.verification.cells import DATASETS, cell_paths, column_key
from spoof_superb.verification.scores import cell_summary, sha256, utt_digest

QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0]


def discover_models(dataset_key, scores_root):
    """Model slugs with a score file for this dataset, by inverting the layout.

    Filenames are never parsed here: model names contain underscores and cannot
    be split out reliably, which is the ambiguity that helped the old eval
    scripts diverge in the first place. `available_frontends` inverts the
    the layout's own naming rule instead.
    """
    try:
        return available_frontends("linear_head", column_key(dataset_key),
                                   scores_root=scores_root)
    except KeyError:
        return []


def read_score_file(path):
    """({utt_id: label}, {utt_id: score}) from a 4-column score file."""
    labels, scores = {}, {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) != 4:
                continue
            utt, _, label, raw = parts
            labels[utt] = label
            try:
                scores[utt] = float(raw)
            except ValueError:
                scores[utt] = float("nan")
    return labels, scores


def file_entry(path, scores_root):
    labels, scores = read_score_file(path)
    values = np.array(list(scores.values()), dtype=float)
    finite = np.isfinite(values)

    bona = np.array([scores[u] for u in scores if labels[u] == "bonafide"], dtype=float)
    spoof = np.array([scores[u] for u in scores if labels[u] == "spoof"], dtype=float)
    bona, spoof = bona[np.isfinite(bona)], spoof[np.isfinite(spoof)]

    eer = None
    if len(bona) and len(spoof):
        eer = round(float(compute_eer(bona, spoof)[0] * 100), 6)

    q = {}
    if finite.any():
        for p, v in zip(QUANTILES, np.quantile(values[finite], QUANTILES)):
            q[str(p)] = round(float(v), 6)

    return {
        "path": os.path.relpath(path, scores_root),
        "bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "n_rows": len(scores),
        "n_bonafide": int(sum(1 for u in scores if labels[u] == "bonafide")),
        "n_spoof": int(sum(1 for u in scores if labels[u] == "spoof")),
        "n_nonfinite": int((~finite).sum()),
        "eer_percent": eer,
        "utt_digest": utt_digest(list(scores)),
        "score_quantiles": q,
    }


def main(argv=None):
    from spoof_superb.config import cfg

    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.build_release_manifest",
        description="Index a finished score tree for release and verification")
    ap.add_argument("--out", default=os.path.join(str(REPO_ROOT), "reference",
                                                  "manifest.json"))
    ap.add_argument("--scores_root", default=None,
                    help="default: the configured scores_root")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="benchmark column display names, e.g. ITW MLAAD")
    ap.add_argument("--archive_url", default=None,
                    help="base URL the files are published under; recorded in "
                         "the manifest so fetch_scores.sh knows where to look")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N models per dataset (smoke test)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args(argv)

    scores_root = args.scores_root or cfg.scores_root

    wanted = DATASETS
    if args.datasets:
        keep = set(args.datasets)
        wanted = [d for d in DATASETS if d[0] in keep or d[1] in keep]
        if not wanted:
            print(f"[ERROR] no benchmark column matches {args.datasets}")
            return 2

    plan = {}
    for disp, key in wanted:
        models = discover_models(key, scores_root)
        if args.limit:
            models = models[:args.limit]
        plan[key] = models
        print(f"  {disp:12s} ({key:28s}) {len(models):3d} model(s)")

    if args.dry_run:
        print(f"\nwould index {sum(len(m) for m in plan.values())} cells "
              f"under {scores_root}")
        print(f"would write {args.out}")
        return 0

    manifest = {
        "generated": date.today().isoformat(),
        "scores_root_at_build": scores_root,
        "archive_url": args.archive_url,
        "files": {},
        "cells": {},
    }
    n_files = n_cells = 0
    for disp, key in wanted:
        models = plan.get(key) or []
        if not models:
            continue
        manifest["files"][key] = {}
        manifest["cells"][key] = {}
        for model in models:
            paths = cell_paths(scores_root, key, model)
            if any(not p.exists() for p in paths):
                continue
            # A pooled column is several files; index each so they fetch
            # individually, then summarise the pool as the cell.
            entries = [file_entry(str(p), scores_root) for p in paths]
            manifest["files"][key][model] = (entries if len(entries) > 1
                                             else entries[0])
            summary = cell_summary(paths)
            manifest["cells"][key][model] = summary
            n_files += len(entries)
            n_cells += 1
            eer = summary["eer_percent"]
            print(f"    {disp:12s} {model:34s} n={summary['n_rows']:>9} "
                  f"EER={'-' if eer is None else f'{eer:.3f}':>8}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, args.out)

    size = os.path.getsize(args.out) / 1024
    print(f"\nindexed {n_cells} cells over {n_files} files -> {args.out} "
          f"({size:.0f} KB)")
    if not args.archive_url:
        print("[note] no --archive_url recorded; bin/fetch_scores.sh will need "
              "SPOOF_SUPERB_SCORES_URL set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
