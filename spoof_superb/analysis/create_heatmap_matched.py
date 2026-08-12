"""Figures 1 and 2 from the composition- and coverage-matched matrix.

    python -m spoof_superb.analysis.create_heatmap_matched

Reads `{outputs_root}/degradation_matched/{eer_matrix,eer_sd}.csv` and writes
the two figures beside them. `create_heatmap.py` is left alone: it draws the
published five-column figures and is what regenerates them.

What differs from the five-column figures
-----------------------------------------
Nine columns, not five. A condition is split by the corpus that carries it,
because collapsing them hides the largest effect in the data -- codec costs
+68% on ASVspoof 5 and about nothing on ASVLD or ASV21 DF, and one "Codec"
column reports neither.

The Mean is condition-balanced: corpora are averaged within a condition first,
then the five conditions are averaged. A flat mean over the nine cells would
give Codec and Channel three slots each and Noise one, making it two-thirds a
statement about the two ASV5-dominated conditions.

Cells carry mean +- sd across that cell's variants. Two cells have a single
variant -- Channel:ASVLD (one 7 kHz low-pass) and Channel:ASV5 (C11) -- and
their sd is omitted rather than printed as 0.0, which would read as agreement
between settings that were never measured.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spoof_superb import REPO_ROOT
from spoof_superb.config import cfg
from spoof_superb.scoring.models import family_separator_rows, paper_table_rows

#: (column, condition, corpus label). Column order is the figure's column order.
COLUMNS = [
    ("Codec:ASVLD",      "Codec",      "ASVLD"),
    ("Codec:ASV21DF",    "Codec",      "ASV21 DF"),
    ("Codec:ASV5",       "Codec",      "ASV5"),
    ("Noise:ASVLD",      "Noise",      "ASVLD"),
    ("Resampling:ASVLD", "Resampling", "ASVLD"),
    ("Reverb:ASVLD",     "Reverb",     "ASVLD"),
    ("Channel:ASVLD",    "Channel",    "ASVLD"),
    ("Channel:ASV21LA",  "Channel",    "ASV21 LA"),
    ("Channel:ASV5",     "Channel",    "ASV5"),
]
COLS = [c for c, _, _ in COLUMNS]
CONDITIONS = ["Codec", "Noise", "Resampling", "Reverb", "Channel"]

#: Cells built from one variant: no spread exists to report.
SINGLE_VARIANT = {"Channel:ASVLD", "Channel:ASV5"}


def condition_balanced_mean(frame):
    """Average within a condition, then across the five conditions."""
    per = pd.DataFrame(index=frame.index)
    for cond in CONDITIONS:
        members = [c for c, k, _ in COLUMNS if k == cond]
        per[cond] = frame[members].mean(axis=1)
    return per[CONDITIONS].mean(axis=1)


def group_edges():
    """Column indices where the condition changes."""
    conds = [k for _, k, _ in COLUMNS]
    return [i for i in range(1, len(conds)) if conds[i] != conds[i - 1]]


def annotate_groups(ax):
    """Condition names above the columns, with rules between the groups."""
    conds = [k for _, k, _ in COLUMNS]
    for edge in group_edges():
        ax.vlines(edge, *ax.get_ylim(), colors="black", linewidth=1.5)
    start = 0
    for i in range(len(conds) + 1):
        if i == len(conds) or conds[i] != conds[start]:
            ax.text((start + i) / 2, -0.28, conds[start], ha="center",
                    va="bottom", fontsize=13, fontweight="bold")
            start = i


def labels(values, sd=None, fmt="{:.1f}"):
    """Cell text: value, plus '+-sd' where a spread was actually measured."""
    out = []
    for model in values.index:
        row = []
        for col in values.columns:
            txt = fmt.format(values.loc[model, col])
            if sd is not None and col not in SINGLE_VARIANT:
                txt += "\n$\\pm$" + f"{sd.loc[model, col]:.1f}"
            row.append(txt)
        out.append(row)
    return np.array(out)


def draw_separators(axes, rows):
    for ax in axes:
        xlim = ax.get_xlim()
        for y in rows:
            ax.hlines(y, *xlim, colors="black", linewidth=1.5)


def plot_absolute(eer, sd, out_path):
    df = eer.copy()
    df["Mean"] = condition_balanced_mean(df)

    sns.set(style="white", font_scale=1.0)
    side = sns.light_palette("gray", as_cmap=True)

    # Baseline | cells | Mean | colourbar. The colourbar gets its own slot so
    # seaborn cannot steal width from the cells and open a gap before Mean.
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 9.0, 0.95, 0.22],
                          wspace=0.02)
    ax0, ax1, ax2, cax = (fig.add_subplot(gs[0, i]) for i in range(4))

    sns.heatmap(df[["Baseline"]], annot=True, fmt=".1f", annot_kws={"fontsize": 11},
                cmap=side, cbar=False,
                linewidths=0.5, linecolor="white", ax=ax0, yticklabels=df.index,
                xticklabels=False)
    # Values only. The per-variant spread moved to the appendix: a cell already
    # carries a colour and a number, and a third quantity made it unreadable.
    sns.heatmap(df[COLS], annot=True, fmt=".1f", annot_kws={"fontsize": 11},
                cmap="YlGnBu", cbar=True, cbar_ax=cax,
                cbar_kws={"label": "EER (%)"},
                linewidths=0.5, linecolor="white", ax=ax1, yticklabels=False,
                xticklabels=[c for _, _, c in COLUMNS])
    sns.heatmap(df[["Mean"]], annot=True, fmt=".1f", annot_kws={"fontsize": 11},
                cmap=side, cbar=False,
                linewidths=0.5, linecolor="white", ax=ax2, yticklabels=False,
                xticklabels=False)

    ax0.set_title("Baseline", fontsize=13, fontweight="bold", pad=6)
    ax2.set_title("Mean", fontsize=13, fontweight="bold", pad=6)
    annotate_groups(ax1)

    for ax in (ax0, ax1, ax2):
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax0.tick_params(axis="y", labelsize=12, rotation=0)
    cax.tick_params(labelsize=10)
    cax.yaxis.label.set_size(11)
    draw_separators([ax0, ax1, ax2], family_separator_rows(list(df.index)))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_relative(eer, out_path):
    base = eer["Baseline"]
    rel = pd.DataFrame(index=eer.index)
    for c in COLS:
        rel[c] = (eer[c] - base) / base * 100
    rel["Mean"] = condition_balanced_mean(rel)

    sns.set(style="white", font_scale=1.0)
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[9.0, 0.95, 0.22], wspace=0.02)
    ax1, ax2, cax = (fig.add_subplot(gs[0, i]) for i in range(3))

    sns.heatmap(rel[COLS], annot=True, fmt=".1f", annot_kws={"fontsize": 11},
                # No cell falls below -9, so a wide negative limit would render
                # every improvement as indistinguishable white.
                cmap="RdBu_r", center=0, vmin=-20, vmax=120,
                linewidths=0.5, linecolor="white",
                cbar=True, cbar_ax=cax,
                cbar_kws={"label": "Relative Change (%)"},
                ax=ax1, yticklabels=rel.index,
                xticklabels=[c for _, _, c in COLUMNS])
    sns.heatmap(rel[["Mean"]], annot=True, fmt=".1f", annot_kws={"fontsize": 11},
                cmap=sns.light_palette("gray", as_cmap=True), cbar=False,
                linewidths=0.5, linecolor="white", ax=ax2, yticklabels=False,
                xticklabels=False)

    ax2.set_title("Mean", fontsize=13, fontweight="bold", pad=6)
    annotate_groups(ax1)

    for ax in (ax1, ax2):
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=11, rotation=0)
    ax1.tick_params(axis="y", labelsize=12, rotation=0)
    cax.tick_params(labelsize=10)
    cax.yaxis.label.set_size(11)
    draw_separators([ax1, ax2], family_separator_rows(list(rel.index)))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.create_heatmap_matched",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)

    root = getattr(cfg, "outputs_root", "") or str(REPO_ROOT / "outputs")
    in_dir = Path(args.in_dir or os.path.join(root, "degradation_matched"))
    out_dir = Path(args.out_dir or in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eer = pd.read_csv(in_dir / "eer_matrix.csv", index_col="Model")
    sd = pd.read_csv(in_dir / "eer_sd.csv", index_col="Model")
    order = [m for m in paper_table_rows() if m in eer.index]
    missing = [m for m in paper_table_rows() if m not in eer.index]
    if missing:
        print(f"  [WARN] models missing from the matrix: {missing}")
    eer, sd = eer.loc[order], sd.loc[order]

    plot_absolute(eer, sd, str(out_dir / "fig1_acoustic_eer_absolute_matched.png"))
    plot_relative(eer, str(out_dir / "fig2_acoustic_eer_relative_matched.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
