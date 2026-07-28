"""
build_release_manifest.py
-------------------------
Publisher-side tool: index a finished score tree for release.

The score files are ~6 GB and do not belong in git. They are published as a
per-file archive (a release asset or a DOI record) and fetched on demand by
`bin/fetch_scores.sh`. This writes the small index that makes that work:

    reference/manifest.json     ~100 KB, versioned in the repo

Per (dataset, model) it records the relative path, sha256, row and class
counts, non-finite count, and the EER. That is enough for three things without
downloading anything:

  * `fetch_scores.sh` knows what exists and where to put it
  * a download can be checked for corruption or tampering
  * "what should my EER be?" is answerable offline

Nothing else is shipped. Trial lists and score subsamples were considered and
rejected: both are derived from the score files, so committing them would put
two copies of the same information under version control with no record of
which is authoritative -- the exact pattern that produced the duplication in
the score directory.

Run this once when publishing a score tree, not as part of normal work.

Usage
-----
    python -m spoof_superb.tools.build_release_manifest --dry-run
    python -m spoof_superb.tools.build_release_manifest
    python -m spoof_superb.tools.build_release_manifest --datasets wild --limit 2
"""

import argparse
import glob
import hashlib
import json
import os
from datetime import date

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.core.metrics import compute_eer
from spoof_superb.scoring.datasets import DATASETS, has_reference, reference_paths

QUANTILES = [0.0, 0.25, 0.5, 0.75, 1.0]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def discover_models(dataset, placeholder="\x00"):
    """SSL model names that have a score file for this dataset.

    Splits the registry's own filename template on the {ssl} slot and globs.
    Filenames are never parsed: model names contain underscores and cannot be
    split out reliably, which is the ambiguity that helped the old eval scripts
    diverge in the first place.
    """
    if not has_reference(dataset):
        return []          # no published score file: nothing to index
    template = reference_paths(dataset, placeholder)[0]
    prefix, suffix = template.split(placeholder)
    names = []
    for path in sorted(glob.glob(prefix + "*" + suffix)):
        name = path[len(prefix):len(path) - len(suffix)] if suffix else path[len(prefix):]
        if name:
            names.append(name)
    return names


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
        "score_quantiles": q,
    }


def main(argv=None):
    from spoof_superb.config import cfg

    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.build_release_manifest",
        description="Index a finished score tree for release")
    ap.add_argument("--out", default=os.path.join(str(REPO_ROOT), "reference",
                                                  "manifest.json"))
    ap.add_argument("--scores_root", default=None,
                    help="default: the configured scores_root")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--archive_url", default=None,
                    help="base URL the files are published under; recorded in "
                         "the manifest so fetch_scores.sh knows where to look")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N models per dataset (smoke test)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args(argv)

    scores_root = args.scores_root or cfg.scores_root
    datasets = args.datasets or list(DATASETS)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        print(f"[ERROR] unknown dataset(s): {', '.join(unknown)}")
        return 2

    plan = {}
    for ds in datasets:
        models = discover_models(ds)
        if args.limit:
            models = models[:args.limit]
        plan[ds] = models
        print(f"  {ds:22s} {len(models):3d} model(s)")

    if args.dry_run:
        print(f"\nwould index {sum(len(m) for m in plan.values())} files "
              f"under {scores_root}")
        print(f"would write {args.out}")
        return 0

    manifest = {
        "generated": date.today().isoformat(),
        "scores_root_at_build": scores_root,
        "archive_url": args.archive_url,
        "datasets": {},
    }
    n = 0
    for ds, models in plan.items():
        if not models:
            continue
        manifest["datasets"][ds] = {}
        for model in models:
            paths = reference_paths(ds, model)
            if any(not os.path.isfile(p) for p in paths):
                continue
            # A pooled column is several files; index each so they fetch
            # individually.
            entries = [file_entry(p, scores_root) for p in paths]
            manifest["datasets"][ds][model] = entries if len(entries) > 1 else entries[0]
            n += 1
            eer = entries[0]["eer_percent"]
            print(f"    {model:34s} n={entries[0]['n_rows']:>9} "
                  f"EER={'-' if eer is None else f'{eer:.3f}':>8}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, args.out)

    size = os.path.getsize(args.out) / 1024
    print(f"\nindexed {n} files -> {args.out} ({size:.0f} KB)")
    if not args.archive_url:
        print("[note] no --archive_url recorded; bin/fetch_scores.sh will need "
              "SPOOF_SUPERB_SCORES_URL set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
