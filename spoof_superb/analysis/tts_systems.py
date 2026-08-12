"""TTS systems analysis (Sections 4.4.3 and 3.2.3).

Builds the `tts_systems` view and reports over it, in one command:

    python -m spoof_superb.analysis.tts_systems --out_dir outputs/tts

Four groupings, each a heatmap of SSL models against groups, plus its CSV:

    eer_by_tts_system       the 91 canonical systems
    eer_by_architecture     the 11 mutually exclusive architecture groups
    eer_by_generation_mode  AR / NAR / Closed-Undisclosed
    eer_by_vocoder_family   the vocoder family, not the verbatim vocoder

Only the finest grouping -- the system -- is in the view. Architecture,
generation mode and vocoder family are FUNCTIONS of the system, recorded in
`mlaad_v10_tts_architecture_groups.csv`, so they are grouped up here rather
than fixed into the directory tree. A tree that hard-coded them would have to
be rebuilt whenever the taxonomy was revised, and would let the tree and the
table disagree in the meantime.

Every group, at every level, is scored against the SAME pooled M-AILABS
bonafide reference (584,006 utterances). That is the reason this analysis is
restricted to MLAAD: every MLAAD system synthesises from the same bonafide
source, so a per-system EER measures the detectability of the synthesis system
rather than the difficulty of its source corpus. Scoring a system against a
bonafide subset of its own would reintroduce exactly the confound the
restriction removes.

Bonafide is the target class -- a higher score means more genuine -- matching
`core.metrics.compute_eer`.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.analysis import metadata_csv
from spoof_superb.analysis.create_mlaad_tts_eer_heatmaps import (
    auc_bonafide_vs_spoof, plot_heatmap)
from spoof_superb.analysis.views import VIEW_SPECS
from spoof_superb.config import cfg
from spoof_superb.core.metrics import compute_eer
from spoof_superb.analysis.create_mlaad_group_ranked_figures import (
    main as ranked_group_figures)
from spoof_superb.analysis.create_mlaad_tts_system_ranked_figure import (
    main as ranked_system_figure)
from spoof_superb.scoring.models import (display_by_slug, paper_models,
                                         paper_table_order, paper_table_rows)
from spoof_superb.tools.build_view import build

#: Generation modes, in the order the figures present them.
MODE_ORDER = ["AR", "NAR", "closed_undisclosed"]
MODE_LABEL = {"AR": "AR", "NAR": "NAR",
              "closed_undisclosed": "Closed / Undisclosed"}

FIGURES = [
    ("system",       "eer_by_tts_system",      "EER (%) by TTS system",
     dict(figwidth=26, figheight=9, annot_size=4.2, xtick_size=6, xtick_rotation=90)),
    ("architecture", "eer_by_architecture",    "EER (%) by architecture group",
     dict(figwidth=13, figheight=8, annot_size=7.5, xtick_size=9, xtick_rotation=45)),
    ("mode",         "eer_by_generation_mode", "EER (%) by generation mode",
     dict(figwidth=8, figheight=8, annot_size=9, xtick_size=10, xtick_rotation=0)),
    ("vocoder",      "eer_by_vocoder_family",  "EER (%) by vocoder family",
     dict(figwidth=18, figheight=8, annot_size=6.5, xtick_size=8, xtick_rotation=60)),
]


def taxonomy():
    """system -> {architecture, mode, vocoder}, from the curated CSV."""
    import pandas as pd
    arch = pd.read_csv(metadata_csv("mlaad_v10_tts_architecture_groups.csv"))
    return {
        "architecture": dict(zip(arch["tts_system"], arch["architecture_group"])),
        "mode": dict(zip(arch["tts_system"], arch["ar_nar"].map(
            {"AR": "AR", "NAR": "NAR", "unknown": "closed_undisclosed"}))),
        "vocoder": dict(zip(arch["tts_system"], arch["vocoder_family"])),
    }, arch


def eers_for_model(groups, bonafide_scores, tax, model):
    """EER of every group at all four levels, against the shared bonafide pool.

    The coarse levels are recomputed by POOLING the systems' spoof scores, not
    by averaging their EERs. Averaging EERs would weight a 1,000-utterance
    system the same as a 34,000-utterance one and would not be an EER of
    anything.
    """
    per_system = {}
    for (_mode, system), (_u, _l, scores) in groups.items():
        per_system.setdefault(system, []).append(scores)
    per_system = {s: np.concatenate(v) for s, v in per_system.items()}

    out = {"system": {}}
    for system, scores in per_system.items():
        out["system"][system] = _eer(bonafide_scores, scores, system, model)

    for level in ("architecture", "mode", "vocoder"):
        pooled = {}
        for system, scores in per_system.items():
            group = tax[level].get(system)
            if group is None:
                raise KeyError(f"{system!r} has no {level} in the taxonomy CSV")
            pooled.setdefault(group, []).append(scores)
        out[level] = {g: _eer(bonafide_scores, np.concatenate(v), g, model)
                      for g, v in pooled.items()}
    return out


def _eer(bonafide, spoof, group, model):
    eer = compute_eer(bonafide, spoof)[0] * 100.0
    if not np.isfinite(eer) or not (0.0 <= eer <= 100.0):
        sys.exit(f"FATAL: EER {eer} out of range for {group}/{model}")
    if eer > 50.0:
        auc = auc_bonafide_vs_spoof(bonafide, spoof)
        print(f"    [EER>50] {group} {model}: EER={eer:.2f}% "
              f"AUC(bonafide>spoof)={auc:.4f} n={len(spoof)}", flush=True)
    return eer



def default_out_dir(name):
    """Where an analysis writes, unless --out_dir says otherwise.

    `analysis_root` in configs/paths.yaml when set, {bench_root}/analysis when not.
    """
    root = cfg.analysis_dir
    return os.path.join(root, name)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.tts_systems",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=None,
                    help="default: analysis_root/tts")
    ap.add_argument("--scores_root", default=None)
    ap.add_argument("--out_root", default=None,
                    help="where views/ is written (default: --scores_root)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    import pandas as pd

    scores_root = args.scores_root or cfg.scores_root
    spec = VIEW_SPECS["tts_systems"]
    roster = set(args.models or paper_models())
    models = paper_table_order(roster)
    display = display_by_slug()

    out_dir = Path(args.out_dir or default_out_dir("tts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tax, arch = taxonomy()

    print("=" * 78)
    print("STEP 1/2 -- build the tts_systems view, and score it")
    print("=" * 78, flush=True)

    results = {level: {} for level, _s, _t, _k in FIGURES}
    support = {}
    for model, groups, bonafide in build(
            spec, models, scores_root=scores_root,
            out_root=args.out_root or scores_root):
        bona_scores = bonafide[2]
        eers = eers_for_model(groups, bona_scores, tax, model)
        name = display.get(model, model)
        for level in results:
            results[level][name] = eers[level]
        if not support:
            for (_m, system), (u, _l, _s) in groups.items():
                support[system] = support.get(system, 0) + len(u)

    if not support:
        sys.exit("FATAL: no model produced a TTS system EER.")

    print()
    print("=" * 78)
    print("STEP 3 -- figures")
    print("=" * 78, flush=True)

    defined = {
        "system": sorted(support),
        "architecture": sorted(set(arch["architecture_group"])),
        "mode": [m for m in MODE_ORDER],
        "vocoder": sorted(set(arch["vocoder_family"])),
    }
    summary = []
    for level, stem, title, style in FIGURES:
        df = pd.DataFrame(results[level]).T
        df.index.name = "Model"
        df = df.reindex([n for n in paper_table_rows() if n in df.index])
        plotted = list(df.columns)
        df = df[[c for c in defined[level] if c in plotted]
                + [c for c in plotted if c not in defined[level]]]
        if level == "mode":
            df.columns = [MODE_LABEL.get(c, c) for c in df.columns]

        df.to_csv(out_dir / f"{stem}.csv", float_format="%.4f")
        print(f"  {stem}: {len(df)} models x {df.shape[1]} groups"
              f"  (protocol defines {len(defined[level])})", flush=True)
        if not args.no_figures:
            plot_heatmap(df, title, out_dir / f"{stem}.png", **style)
        summary.append({"figure": stem, "n_models": len(df),
                        "groups_plotted": df.shape[1],
                        "groups_in_protocol": len(defined[level]),
                        "cells_eer_gt_50": int((df > 50.0).to_numpy().sum())})

    pd.DataFrame(summary).to_csv(out_dir / "figure_summary.csv", index=False)
    print("\n" + pd.DataFrame(summary).to_string(index=False))

    if not args.no_figures:
        # The paper's TTS figures: six representative models, columns and
        # systems ranked by their Mean, the 91 systems split across two panels.
        # These read the CSVs just written, which is why those carry display
        # names in table order rather than slugs.
        print()
        print("=" * 78)
        print("STEP 4 -- the paper's ranked figures")
        print("=" * 78, flush=True)
        ranked_group_figures(["--out_dir", str(out_dir)])
        ranked_system_figure(["--out_dir", str(out_dir)])

    skipped = build.last_manifest["skipped"]
    if skipped:
        print(f"\nNot scored on this tree, omitted ({len(skipped)}): "
              f"{', '.join(skipped)}")
    print(f"\nDone. {len(results['system'])} models, {len(support)} systems, "
          f"{sum(support.values()):,} spoof utts vs the shared bonafide pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
