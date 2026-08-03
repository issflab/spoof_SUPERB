"""Acoustic degradation analysis (Section 4.4.2, tab:acoustic_degradation).

Builds the `acoustic_degradation` view and reports over it, in one command:

    python -m spoof_superb.analysis.acoustic_degradation --out_dir outputs/degradation

Six conditions -- one clean reference and five degraded -- each pooled from
partitions of four corpora. For every SSL model this reports the absolute EER
per condition and the relative change against the Baseline:

    dEER = (EER_deg - EER_clean) / EER_clean

The view is built as part of the analysis rather than as a prerequisite step,
so the numbers and the grouping they were computed over cannot disagree. It is
written under `{scores_root}/views/acoustic_degradation/` unless `--out_root`
sends it elsewhere.

Why every condition carries corpora it does not degrade
------------------------------------------------------
Each degraded condition retains the partitions the paper does not degrade -- so
Additive Noise is ASVLD's noise-augmented set POOLED WITH clean ASV21 LA:C1,
ASV21 DF:C1 and ASV5:C00. That is deliberate and load-bearing: the EER moves
only for the degradation under study, against an otherwise identical pool. A
per-degradation EER computed on the degraded corpus alone would confound the
degradation with the corpus's own difficulty.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.analysis.views import VIEW_SPECS
from spoof_superb.config import cfg
from spoof_superb.core.metrics import compute_eer

from spoof_superb.scoring.models import paper_models
from spoof_superb.tools.build_view import build

#: Column order in the report: the reference first, then the degraded
#: conditions in the order tab:acoustic_degradation lists them.
CONDITIONS = ["Baseline", "Codec_Compression", "Bandwidth", "Additive_Noise",
              "Reverberation", "Channel_Distortions"]

#: Display names, kept out of the directory names so the tree stays greppable.
DISPLAY = {
    "Baseline": "Baseline",
    "Codec_Compression": "Codec & Compression",
    "Bandwidth": "Bandwidth",
    "Additive_Noise": "Additive Noise",
    "Reverberation": "Reverberation",
    "Channel_Distortions": "Channel Distortions",
}

REFERENCE = "Baseline"


def eer_pct(labels, scores):
    """EER in percent, or None if a class is missing or every score is NaN."""
    finite = ~np.isnan(scores)
    labels, scores = labels[finite], scores[finite]
    bona = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    if bona.size == 0 or spoof.size == 0:
        return None
    return 100.0 * compute_eer(bona, spoof)[0]


def delta_eer(degraded, clean):
    """Relative change in EER against the clean reference.

    Returns None when the reference EER is 0 -- the ratio is undefined there,
    and reporting an infinity would put a number in a table that means "we
    divided by zero".
    """
    if degraded is None or clean is None or clean == 0:
        return None
    return (degraded - clean) / clean


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.acoustic_degradation",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "degradation"),
                    help="where the CSVs and figures go")
    ap.add_argument("--scores_root", default=None,
                    help="score tree to read (default: the configured one)")
    ap.add_argument("--layout", default=None, choices=("legacy", "v2", "v3"))
    ap.add_argument("--out_root", default=None,
                    help="where views/ is written (default: --scores_root)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="score-file slugs (default: the paper roster)")
    ap.add_argument("--no-figures", action="store_true",
                    help="write the CSVs only")
    args = ap.parse_args(argv)

    scores_root = args.scores_root or cfg.scores_root
    layout = args.layout or getattr(cfg, "score_layout", "legacy")
    spec = VIEW_SPECS["acoustic_degradation"]
    models = args.models or sorted(paper_models())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 1/2 -- build the acoustic_degradation view, and score it")
    print("=" * 78, flush=True)

    # One model at a time: `build` writes each model's groups, hands them over,
    # and moves on. This view is ~4.5M rows per model, so holding the whole
    # roster to score it afterwards would cost tens of GB to no purpose.
    rows = []
    for model, groups, _bonafide in build(
            spec, models, scores_root=scores_root, layout=layout,
            out_root=args.out_root or scores_root):
        eers = {}
        for cond in CONDITIONS:
            key = (cond,)
            eers[cond] = (eer_pct(*groups[key][1:]) if key in groups else None)
        ref = eers.get(REFERENCE)
        row = {"Model": model}
        for cond in CONDITIONS:
            row[DISPLAY[cond]] = eers[cond]
            if cond != REFERENCE:
                row[f"dEER {DISPLAY[cond]}"] = delta_eer(eers[cond], ref)
        rows.append(row)
        cells = "  ".join(
            f"{c[:9]}={eers[c]:6.2f}" if eers[c] is not None else f"{c[:9]}=   n/a"
            for c in CONDITIONS)
        print(f"      {cells}", flush=True)

    skipped = [s.split(" ")[0] for s in build.last_manifest["skipped"]]

    if not rows:
        sys.exit("FATAL: no model produced a condition EER.")

    eer_csv = out_dir / "eer_by_condition.csv"
    fields = ["Model"] + [DISPLAY[c] for c in CONDITIONS] + [
        f"dEER {DISPLAY[c]}" for c in CONDITIONS if c != REFERENCE]
    with open(eer_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nWrote {eer_csv}")

    if not args.no_figures:
        _figures(rows, out_dir)

    if skipped:
        print(f"\nNot scored on this tree, omitted ({len(skipped)}): "
              f"{', '.join(sorted(skipped))}")
    print(f"\nSTEP 3 -- done. {len(rows)} models x {len(CONDITIONS)} conditions.")
    return 0


def _figures(rows, out_dir):
    """Absolute-EER and dEER heatmaps, models x conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    absolute = pd.DataFrame(
        {r["Model"]: {DISPLAY[c]: r[DISPLAY[c]] for c in CONDITIONS}
         for r in rows}).T[[DISPLAY[c] for c in CONDITIONS]]
    relative = pd.DataFrame(
        {r["Model"]: {DISPLAY[c]: r[f"dEER {DISPLAY[c]}"]
                      for c in CONDITIONS if c != REFERENCE}
         for r in rows}).T

    # These names must not collide with the report CSV main() writes. They did:
    # the absolute matrix was called `eer_by_condition` here too, so the figure
    # step silently overwrote the full table -- same filename, but without the
    # Model column or any of the dEER columns.
    for df, name, title, fmt, cmap, center in (
        (absolute, "figure_eer_by_condition", "EER (%) by acoustic condition",
         ".2f", "Reds", None),
        (relative, "figure_delta_eer_by_condition",
         "Relative EER change vs Baseline", ".2f", "RdBu_r", 0.0),
    ):
        fig, ax = plt.subplots(figsize=(2 + 1.6 * df.shape[1], 1 + 0.34 * len(df)))
        sns.heatmap(df.astype(float), annot=True, fmt=fmt, cmap=cmap,
                    center=center, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.6})
        ax.set_title(title, fontsize=12, pad=10)
        ax.tick_params(axis="x", labelsize=9, rotation=30)
        ax.tick_params(axis="y", labelsize=8, rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close()
        df.to_csv(out_dir / f"{name}.csv", float_format="%.4f")
        print(f"Wrote {out_dir / (name + '.png')}")


if __name__ == "__main__":
    raise SystemExit(main())
