"""Materialise an analysis view under {scores_root}/views/.

The analysis entry points build their own view and then report over it, so a
number and the grouping it was computed over cannot disagree. This module is
that build step, usable either as a library:

    for frontend, groups, bonafide in build(spec, models, ...):
        ...                       # rows already written; use them and move on

or on its own, to produce a view for browsing without running an analysis:

    python -m spoof_superb.tools.build_view --view tts_systems \\
        --scores_root /data/ssl_anti_spoofing/spoof_superb_score_files \\
        --layout v3

Writes, per P11:

    views/{view}/{group}/[{subgroup}/]{frontend}.txt   the grouped rows
    views/{view}/_bonafide/{frontend}.txt              shared reference pool
    views/{view}/_manifest.json                        what produced this

`build` is a GENERATOR, and that is not incidental. The acoustic degradation
view is about 4.5 million rows per model across its six conditions; holding
nineteen models at once would cost tens of gigabytes for no reason, since every
consumer reduces one model to a handful of EERs before moving to the next.

The manifest exists because a materialised view can go stale against the raw
files behind it, and the legacy tree shows the cost: it holds
`scores_by_category_augmented` and `scores_by_acoustic_degradation`, the same
view built twice from different runs, with the documentation left to say which
one to trust. Recording the sources, their sizes and mtimes, and the per-group
row counts makes "is this current" answerable.

Writes only under `views/`. Never touches `raw/`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from spoof_superb.analysis.conditions import ASV21_DF_CODES, ASV21_LA_CODES
from spoof_superb.analysis.views import (VIEW_SPECS, CompositeView,
                                          load_view, view_dir)
from spoof_superb.config import cfg
from spoof_superb.core.scorepath import (available_frontends, mlaad_pool_paths,
                                         score_path)

#: Views hold score files, so they hold the same canonical 4-column format raw
#: does. A view that invented its own would need its own reader.
LINE = "{utt} - {key} {score}\n"


def source_paths(spec, frontend, scores_root, layout):
    """Every raw file this view reads for one model."""
    if isinstance(spec, CompositeView):
        datasets = {p.dataset for parts in spec.groups.values() for p in parts}
        return [Path(score_path("linear_head", d, frontend,
                                scores_root=scores_root, layout=layout))
                for d in sorted(datasets)]
    if spec.bonafide_dataset:
        return [Path(p) for p in mlaad_pool_paths(frontend, scores_root=scores_root,
                                                  layout=layout)]
    return [Path(score_path("linear_head", spec.dataset, frontend,
                            scores_root=scores_root, layout=layout,
                            ext=spec.ext))]


def write_group(rows, path):
    """Write one group's rows atomically, in the canonical format."""
    utts, labels, scores = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w") as fh:
        for u, k, s in zip(utts.tolist(), labels.tolist(), scores.tolist()):
            fh.write(LINE.format(utt=u, key=k, score=s))
    os.replace(tmp, path)
    return len(utts)


def _describe_part(part):
    """One dataset's contribution to a group, in words."""
    if part.family is not None:
        return f"{part.dataset} -- the {part.family} conditions"
    if part.conditions is not None:
        codes = ", ".join(_code(part.dataset, c) for c in part.conditions)
        return f"{part.dataset} -- {codes} only"
    if part.exclude is not None:
        codes = ", ".join(_code(part.dataset, c) for c in part.exclude)
        return f"{part.dataset} -- every condition EXCEPT {codes}"
    return f"{part.dataset} -- the whole set"


def _code(dataset, token):
    """Show a protocol token alongside the corpus's own condition code."""
    table = {"asvspoof2021_LA": ASV21_LA_CODES,
             "asvspoof2021_DF": ASV21_DF_CODES}.get(dataset)
    if table and token in table:
        return f"{table[token]} ({token!r})"
    if dataset == "asvspoof5" and token == "-":
        return "C00 ('-')"
    return repr(token)


def write_readme(spec, manifest, out):
    """A human-readable companion to _manifest.json.

    The JSON is for tools; it is a wall of paths and counts and nobody reading
    it can answer "what is in the Codec condition?" without decoding the
    selector fields. This says it in sentences.
    """
    lines = [f"# {spec.name}", "", spec.doc.strip(), "",
             f"Built {manifest['built_at']} from `{manifest['scores_root']}` "
             f"(layout `{manifest['layout']}`).", ""]

    n_models = len(manifest["group_rows"])
    plural = "model" if n_models == 1 else "models"
    if manifest["skipped"]:
        lines += [f"{n_models} {plural}. Not scored on this tree, so absent "
                  f"from the view: {', '.join(manifest['skipped'])}.", ""]
    else:
        lines += [f"{n_models} {plural}.", ""]

    if isinstance(spec, CompositeView):
        rows = next(iter(manifest["group_rows"].values()), None)
        lines += ["## Conditions", "",
                  "Each condition POOLS partitions of several corpora. The ones "
                  "it does not degrade are kept unchanged from the reference, so "
                  "its EER moves only for the degradation under study.", ""]
        if spec.reference:
            lines += [f"Reference condition: **{spec.reference}**.", ""]
        for group, parts in spec.groups.items():
            mark = "  (reference)" if group == spec.reference else ""
            lines.append(f"### {group}{mark}")
            lines.append("")
            for part in parts:
                lines.append(f"* {_describe_part(part)}")
            lines.append("")
        if rows:
            lines += [f"Every condition is written per model, "
                      f"{rows:,} rows per model across all conditions.", ""]
    else:
        counts = manifest["group_rows"]
        first = next(iter(counts.values()), 0)
        lines += ["## Groups", "",
                  f"One directory per group, `{{group}}/{{subgroup}}/{{model}}.txt`. "
                  f"{first:,} rows per model across all groups.", ""]
        if manifest["bonafide_rows"]:
            b = next(iter(manifest["bonafide_rows"].values()))
            lines += [f"`_bonafide/` holds the shared reference pool "
                      f"({b:,} rows per model). Every group is scored against "
                      f"it, never against a subset of its own.", ""]

    lines += ["## Files", "",
              "```",
              f"{spec.name}/{{group}}/[{{subgroup}}/]{{model}}.txt   the grouped rows",
              f"{spec.name}/_manifest.json                     machine-readable record",
              f"{spec.name}/README.md                          this file",
              "```", "",
              "Rows are the canonical 4-column score format, `utt_id - key score`, "
              "the same as `raw/`.", "",
              "Rebuild with:", "",
              "```bash",
              f"python -m spoof_superb.tools.build_view --view {spec.name} \\",
              f"    --scores_root {manifest['scores_root']} "
              f"--layout {manifest['layout']}",
              "```", ""]
    (out / "README.md").write_text("\n".join(lines))


def build(spec, models, scores_root=None, layout=None, out_root=None,
          dry_run=False, verbose=True):
    """Materialise `spec` for each model, yielding (frontend, groups, bonafide).

    Rows are written before each yield, so a consumer that stops early leaves a
    partial but internally consistent view rather than a half-written file.

    Models with no score file are skipped and named, not guessed at. The set of
    group keys must agree across models -- they scored the same trials, so a
    difference means one file is not the set the others are, which is worth
    failing on rather than writing a ragged view.
    """
    scores_root = scores_root or cfg.scores_root
    layout = layout or getattr(cfg, "score_layout", "legacy")
    out = Path(view_dir(spec.name, out_root or scores_root))

    manifest = {
        "view": spec.name,
        "doc": spec.doc,
        "kind": "composite" if isinstance(spec, CompositeView) else "partition",
        "scores_root": str(scores_root),
        "layout": layout,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": {},
        "group_rows": {},
        "bonafide_rows": {},
        "skipped": [],
    }
    if isinstance(spec, CompositeView):
        manifest["reference_group"] = spec.reference
        manifest["composition"] = {
            g: [{"dataset": p.dataset, "conditions": p.conditions,
                 "exclude": p.exclude, "family": p.family} for p in parts]
            for g, parts in spec.groups.items()
        }

    keys_seen = None
    for frontend in models:
        srcs = source_paths(spec, frontend, scores_root, layout)
        missing = [p for p in srcs if not p.exists()]
        if missing:
            manifest["skipped"].append(f"{frontend} ({missing[0].name})")
            if verbose:
                print(f"  {frontend:<40} SKIP (missing {missing[0].name})",
                      flush=True)
            continue

        manifest["sources"][frontend] = [
            {"path": str(p), "bytes": p.stat().st_size,
             "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                    time.localtime(p.stat().st_mtime))}
            for p in srcs
        ]

        groups, bonafide = load_view(spec, frontend, scores_root, layout)

        keys = set(groups)
        if keys_seen is None:
            keys_seen = keys
        elif keys != keys_seen:
            sys.exit(f"FATAL: {frontend} has groups {sorted(keys ^ keys_seen)} "
                     f"that other models do not. The view would be ragged.")

        total = 0
        for key, rows in sorted(groups.items()):
            if dry_run:
                total += len(rows[0])
            else:
                total += write_group(rows, out.joinpath(*key, f"{frontend}.txt"))
        manifest["group_rows"][frontend] = total
        if bonafide is not None:
            manifest["bonafide_rows"][frontend] = (
                len(bonafide[0]) if dry_run
                else write_group(bonafide, out / "_bonafide" / f"{frontend}.txt"))

        if verbose:
            extra = (f"  + {manifest['bonafide_rows'][frontend]:,} bonafide"
                     if bonafide is not None else "")
            print(f"  {frontend:<40} {len(groups):>4} groups  {total:>10,} rows"
                  + extra, flush=True)

        yield frontend, groups, bonafide

    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                       default=str))
        write_readme(spec, manifest, out)
        if verbose:
            print(f"\n  view -> {out}")
            print(f"  what is in it -> {out / 'README.md'}")
            print(f"  machine record -> {out / '_manifest.json'}", flush=True)
    build.last_manifest = manifest


def default_models(spec, scores_root, layout):
    """Every model this tree scored for the view's source dataset."""
    if isinstance(spec, CompositeView):
        dataset = next(iter(next(iter(spec.groups.values())))).dataset
        ext = ".txt"
    else:
        dataset, ext = spec.dataset, spec.ext
    return available_frontends("linear_head", dataset, scores_root=scores_root,
                               layout=layout, ext=ext)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.build_view",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, choices=sorted(VIEW_SPECS))
    ap.add_argument("--scores_root", default=None)
    ap.add_argument("--layout", default=None, choices=("legacy", "v2", "v3"))
    ap.add_argument("--out_root", default=None,
                    help="where views/ is written (default: --scores_root). "
                         "Point elsewhere to build without touching the tree.")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="report groups and row counts, write nothing")
    args = ap.parse_args(argv)

    spec = VIEW_SPECS[args.view]
    scores_root = args.scores_root or cfg.scores_root
    layout = args.layout or getattr(cfg, "score_layout", "legacy")
    models = args.models or default_models(spec, scores_root, layout)
    if not models:
        sys.exit(f"FATAL: no score files for {spec.name} under {scores_root} "
                 f"({layout})")

    print(f"view      {spec.name}")
    print(f"reading   {scores_root}  (layout={layout})")
    print(f"writing   {view_dir(spec.name, args.out_root or scores_root)}"
          + ("  [DRY RUN -- nothing written]" if args.dry_run else ""))
    print(f"models    {len(models)}\n", flush=True)

    n = 0
    for _frontend, _groups, _bonafide in build(
            spec, models, scores_root, layout,
            args.out_root or scores_root, args.dry_run):
        n += 1

    if args.dry_run:
        print(f"\nDRY RUN -- {n} models, nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
