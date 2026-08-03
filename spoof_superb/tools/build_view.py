"""Materialise an analysis view under {scores_root}/views/.

A view is a grouping of raw score rows, declared in `analysis.views`. Analysis
does not need it on disk -- `load_view` returns the same grouping in memory, and
the MLAAD figures already work that way. Build one to browse it, to feed a tool
that wants directories, or to publish a subset.

    python -m spoof_superb.tools.build_view --view mlaad_tts \\
        --scores_root /data/ssl_anti_spoofing/spoof_superb_score_files \\
        --layout v3

Writes, per P11:

    views/{view}/{group}/[{subgroup}/]{frontend}.txt   the grouped rows
    views/{view}/_bonafide/{frontend}.txt              shared reference pool
    views/{view}/_manifest.json                        what produced this

The manifest exists because a materialised view can go stale against the raw
files it came from, and the legacy tree demonstrates the consequence: it holds
`scores_by_category_augmented` and `scores_by_acoustic_degradation`, the same
view built twice from different runs, and the documentation has to tell you
which one to trust. Recording the source files, their sizes and mtimes, and the
per-group row counts makes "is this current" a question with an answer.

Writes only under `views/`. Never touches `raw/`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from spoof_superb.analysis.views import VIEW_SPECS, load_view, view_dir
from spoof_superb.config import cfg
from spoof_superb.core.scorepath import (available_frontends, mlaad_pool_paths,
                                         score_path)

#: The canonical 4-column line. Views hold score files, so they hold the same
#: format raw does -- a view that invented its own would need its own reader.
LINE = "{utt} - {key} {score}\n"


def source_paths(spec, frontend, scores_root, layout):
    """Every raw file this view reads for one model."""
    if spec.bonafide_dataset:
        return [Path(p) for p in mlaad_pool_paths(frontend, scores_root=scores_root,
                                                  layout=layout)]
    return [Path(score_path("linear_head", spec.dataset, frontend,
                            scores_root=scores_root, layout=layout,
                            ext=spec.ext))]


def write_group(rows, path):
    """Write one group's rows atomically, as the canonical format."""
    utts, labels, scores = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w") as fh:
        for u, k, s in zip(utts.tolist(), labels.tolist(), scores.tolist()):
            fh.write(LINE.format(utt=u, key=k, score=s))
    os.replace(tmp, path)
    return len(utts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.build_view",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, choices=sorted(VIEW_SPECS),
                    help="which view to materialise")
    ap.add_argument("--scores_root", default=None,
                    help="tree to read from (default: the configured one)")
    ap.add_argument("--layout", default=None, choices=("legacy", "v2", "v3"))
    ap.add_argument("--out_root", default=None,
                    help="where to write views/ (default: --scores_root, i.e. "
                         "beside raw/). Point elsewhere to build without "
                         "touching the score tree.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="score-file slugs (default: every model present)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the groups and row counts, write nothing")
    args = ap.parse_args(argv)

    spec = VIEW_SPECS[args.view]
    scores_root = args.scores_root or cfg.scores_root
    layout = args.layout or getattr(cfg, "score_layout", "legacy")
    out_root = args.out_root or scores_root
    out = Path(view_dir(spec.name, out_root))

    models = args.models or available_frontends(
        "linear_head", spec.dataset, scores_root=scores_root, layout=layout,
        ext=spec.ext)
    if not models:
        sys.exit(f"FATAL: no {spec.dataset} score files under {scores_root} "
                 f"({layout})")

    print(f"view       {spec.name}")
    print(f"source     {spec.dataset}"
          + (f" + {spec.bonafide_dataset}" if spec.bonafide_dataset else ""))
    print(f"reading    {scores_root}  (layout={layout})")
    print(f"writing    {out}" + ("  [DRY RUN -- nothing written]"
                                 if args.dry_run else ""))
    print(f"models     {len(models)}\n", flush=True)

    manifest = {
        "view": spec.name,
        "doc": spec.doc,
        "source_dataset": spec.dataset,
        "bonafide_dataset": spec.bonafide_dataset,
        "scores_root": str(scores_root),
        "layout": layout,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": list(models),
        "sources": {},
        "group_rows": {},
        "bonafide_rows": {},
    }

    groups_seen = None
    for frontend in models:
        srcs = source_paths(spec, frontend, scores_root, layout)
        missing = [p for p in srcs if not p.exists()]
        if missing:
            print(f"  {frontend:<40} SKIP (missing {missing[0].name})")
            continue
        manifest["sources"][frontend] = [
            {"path": str(p), "bytes": p.stat().st_size,
             "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                    time.localtime(p.stat().st_mtime))}
            for p in srcs
        ]

        groups, bonafide = load_view(spec, frontend, scores_root, layout)
        total = 0
        for key, rows in sorted(groups.items()):
            path = out.joinpath(*key, f"{frontend}.txt")
            total += (len(rows[0]) if args.dry_run else write_group(rows, path))
        if bonafide is not None:
            n_b = (len(bonafide[0]) if args.dry_run
                   else write_group(bonafide, out / "_bonafide" / f"{frontend}.txt"))
            manifest["bonafide_rows"][frontend] = n_b
        manifest["group_rows"][frontend] = total

        # Every model must group identically -- they score the same trials. A
        # difference means one model's file is not the set the others scored,
        # which is worth failing on rather than writing a ragged view.
        keys = set(groups)
        if groups_seen is None:
            groups_seen = keys
        elif keys != groups_seen:
            sys.exit(f"FATAL: {frontend} has groups {sorted(keys ^ groups_seen)} "
                     f"that other models do not. The view would be ragged.")

        print(f"  {frontend:<40} {len(groups):>4} groups  {total:>9} rows"
              + (f"  + {manifest['bonafide_rows'][frontend]} bonafide"
                 if bonafide is not None else ""), flush=True)

    if args.dry_run:
        print(f"\nDRY RUN -- {len(groups_seen or ())} groups per model, "
              f"nothing written.")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nBuilt {out}")
    print(f"  {len(groups_seen or ())} groups x {len(manifest['group_rows'])} models")
    print(f"  manifest: {out / '_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
