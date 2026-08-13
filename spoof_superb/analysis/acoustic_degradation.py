"""Acoustic degradation analysis -- SUPERSEDED, kept for reference.

    Use `spoof_superb.analysis.acoustic_degradation_matched` instead.
    `bin/analyze.sh` runs that one; nothing runs this.

This version substitutes degraded partitions into the pool without holding the
corpus mixture fixed. Because the corpora differ widely in difficulty and in how
much degraded material each contributes -- ASVLD carries fifteen noise settings
per utterance while the others contribute one copy each -- the pool's
composition changes along with the degradation, and the reported change mixes
the two. Re-mixing the CLEAN partitions in those same proportions, with no
degradation applied at all, moves the pooled EER by -18% to -44%.

The matched analysis fixes the composition, degrades exactly one corpus per
cell, and reports nine cells rather than five conditions. Section 4.4.2 of the
paper reports that one. This module is retained so the earlier numbers can
still be regenerated and compared, not because it should be used.

Original documentation follows.

Acoustic degradation analysis (Section 4.4.2, tab:acoustic_degradation).

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
import os
import sys
from pathlib import Path

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.analysis.views import VIEW_SPECS
from spoof_superb.config import cfg
from spoof_superb.core.metrics import compute_eer

from spoof_superb.analysis.create_heatmap import (load_matrix, plot_absolute,
                                                  plot_relative)
from spoof_superb.scoring.models import (display_by_slug, paper_models,
                                         paper_table_rows)
from spoof_superb.tools.build_view import build

#: Column order in the report: the reference first, then the degraded
#: conditions in the order tab:acoustic_degradation lists them.
CONDITIONS = ["Baseline", "Codec_Compression", "Additive_Noise", "Bandwidth",
              "Reverberation", "Channel_Distortions"]

#: View group -> the column name the paper's figures use. These are the names
#: `create_heatmap` expects, because that is the module that drew the published
#: degradation figures and this analysis feeds it rather than re-plotting.
DISPLAY = {
    "Baseline": "Baseline",
    "Codec_Compression": "Codec",
    "Additive_Noise": "Noise",
    "Bandwidth": "Resampling",
    "Reverberation": "Reverb",
    "Channel_Distortions": "Channel",
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



def default_out_dir(name):
    """Where an analysis writes, unless --out_dir says otherwise.

    `analysis_root` in configs/paths.yaml when set, {bench_root}/analysis when not.
    """
    root = cfg.analysis_dir
    return os.path.join(root, name)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.acoustic_degradation",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=None,
                    help="where the CSVs and figures go")
    ap.add_argument("--scores_root", default=None,
                    help="score tree to read (default: the configured one)")
    ap.add_argument("--out_root", default=None,
                    help="where views/ is written (default: --scores_root)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="score-file slugs (default: the paper roster)")
    ap.add_argument("--no-figures", action="store_true",
                    help="write the CSVs only")
    args = ap.parse_args(argv)

    scores_root = args.scores_root or cfg.scores_root
    spec = VIEW_SPECS["acoustic_degradation"]
    models = args.models or sorted(paper_models())

    out_dir = Path(args.out_dir or default_out_dir("degradation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 1/2 -- build the acoustic_degradation view, and score it")
    print("=" * 78, flush=True)

    # One model at a time: `build` writes each model's groups, hands them over,
    # and moves on. This view is ~4.5M rows per model, so holding the whole
    # roster to score it afterwards would cost tens of GB to no purpose.
    rows = []
    for model, groups, _bonafide in build(
            spec, models, scores_root=scores_root,
            out_root=args.out_root or scores_root):
        eers = {}
        for cond in CONDITIONS:
            key = (cond,)
            eers[cond] = (eer_pct(*groups[key][1:]) if key in groups else None)
        row = {"Model": display_by_slug().get(model, model), "slug": model}
        for cond in CONDITIONS:
            row[DISPLAY[cond]] = eers[cond]
        rows.append(row)
        cells = "  ".join(
            f"{c[:9]}={eers[c]:6.2f}" if eers[c] is not None else f"{c[:9]}=   n/a"
            for c in CONDITIONS)
        print(f"      {cells}", flush=True)

    skipped = [s.split(" ")[0] for s in build.last_manifest["skipped"]]

    if not rows:
        sys.exit("FATAL: no model produced a condition EER.")

    # Ordered as the paper's table prints them, so the figures read in the same
    # sequence as Table 6 and the family rules land in the right places.
    order = {name: i for i, name in enumerate(paper_table_rows())}
    rows.sort(key=lambda r: order.get(r["Model"], len(order)))

    eer_csv = out_dir / "eer_matrix.csv"
    fields = ["Model"] + [DISPLAY[c] for c in CONDITIONS]
    with open(eer_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nWrote {eer_csv}")

    if not args.no_figures:
        print()
        print("=" * 78)
        print("STEP 3 -- figures, via create_heatmap (the published ones)")
        print("=" * 78, flush=True)
        df = load_matrix(str(eer_csv))
        plot_absolute(df, str(out_dir /
                              "acoustic_eer_heatmap_absolute_eer_categorized.png"))
        plot_relative(df, str(out_dir /
                              "acoustic_eer_heatmap_relative_eer_categorized.png"))

    if skipped:
        print(f"\nNot scored on this tree, omitted ({len(skipped)}): "
              f"{', '.join(sorted(skipped))}")
    print(f"\nDone. {len(rows)} models x {len(CONDITIONS)} conditions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
