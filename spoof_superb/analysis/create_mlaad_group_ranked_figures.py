"""
Ranked architecture-group and vocoder-family MLAAD heatmaps for the paper.

Consistency with the per-system figure (create_mlaad_tts_system_ranked_figure.py):
both figures use the SAME six representative SSL models and rank the grouping
columns by the Mean row, easiest to hardest. The six models are a DISPLAY
sample; the Mean is over every model scored, because it is a claim about the
TTS group rather than about those six.

Orientation is kept as SSL models on rows and groups on columns (the groups are
few enough to be columns, unlike the 91 systems), with a Mean row at the bottom
that is also the sort key. Colour saturates at 50% (chance) to match the sibling
figures; cell annotations show the true, unclipped EER. Each column label carries
its system count in parentheses so single-system groups are visible as such.

Reads the raw EER matrices already written by create_mlaad_tts_eer_heatmaps.py
(eer_by_architecture.csv, eer_by_vocoder_family.csv); it does not recompute EERs.

Usage
-----
    python -m spoof_superb.analysis.create_mlaad_group_ranked_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from spoof_superb.analysis import metadata_csv
from spoof_superb.config import cfg

SCORE_ROOT = Path(cfg.scores_root)
#: Corpus metadata, resolved from the repo (see analysis.metadata_csv).
ARCH_NAME = "mlaad_v10_tts_architecture_groups.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures_mlaad_tts"

# Same six representative models as the ranked per-system figure, weakest->best.
REPRESENTATIVE = [
    "APC",
    "wav2vec 2.0 Large",
    "HuBERT Large",
    "XLS-R",
    "WavLM Large",
    "SSAST",
]
VMAX = 50.0
EXCLUDED = {"Dual-AR", "RVC", "Voxtral"}  # excluded/merged from the 91-system set


def group_sizes(col: str) -> dict[str, int]:
    """System count per group (col in {architecture_group, vocoder_family})."""
    arch = pd.read_csv(metadata_csv(ARCH_NAME))
    arch = arch[~arch["tts_system"].isin(EXCLUDED)]
    return arch[col].value_counts().to_dict()


def ranked_frame(csv_name: str, sizes: dict[str, int]) -> pd.DataFrame:
    """6 representative models x groups, sorted by the Mean row, Mean appended.

    The Mean is over EVERY model in the CSV, not over the six displayed. The
    six are a display sample -- one per performance tier, so the reader sees the
    spread without 19 rows of heatmap -- but the Mean is a claim about how
    detectable a TTS group is, and that should use all the evidence there is.

    Taking it over the six biased it low by 7.3 pp on the architecture groups
    and 7.7 pp on the vocoder families, because three of the six (XLS-R, WavLM
    Large, HuBERT Large) are among the strongest models in the roster. It also
    moved the ranking this figure exists to show: 7 of 11 architecture groups
    and 21 of 27 vocoder families change rank between the two.

    What that cost, and why it is worth paying: sorting by a Mean over rows the
    reader cannot see means the ordering is no longer checkable against the
    cells in front of them. The row is labelled with its model count so the
    figure says where the number came from.
    """
    df = pd.read_csv(OUT_DIR / csv_name, index_col="Model")
    missing = [m for m in REPRESENTATIVE if m not in df.index]
    if missing:
        sys.exit(f"FATAL: {csv_name} missing models {missing}")
    mean = df.mean(axis=0)
    order = mean.sort_values().index
    sub = df.loc[REPRESENTATIVE, order]
    sub.loc[f"Mean (all {len(df)})"] = mean[order]
    sub.columns = [f"{g} ({sizes.get(g, 0)})" for g in sub.columns]
    return sub


def plot(df: pd.DataFrame, out_png: Path, figwidth: float, xtick_rot: float,
         annot_size: float) -> None:
    fig, ax = plt.subplots(figsize=(figwidth, 3.2))
    sns.heatmap(
        df.clip(upper=VMAX),
        annot=df.round(1).values,
        fmt="",
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        vmin=0,
        vmax=VMAX,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "EER (%, colour capped at 50)"},
        ax=ax,
        annot_kws={"size": annot_size},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=xtick_rot, ha="right")
    ax.hlines(len(df) - 1, *ax.get_xlim(), colors="black", linewidth=1.4)  # Mean row
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main(argv=None) -> None:
    import argparse
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(OUT_DIR),
                    help="directory holding the EER CSVs, and where the ranked "
                         "figures are written")
    args = ap.parse_args(argv)
    OUT_DIR = Path(args.out_dir)

    arch = ranked_frame("eer_by_architecture.csv", group_sizes("architecture_group"))
    voc = ranked_frame("eer_by_vocoder_family.csv", group_sizes("vocoder_family"))

    arch.to_csv(OUT_DIR / "eer_by_architecture_ranked.csv", float_format="%.4f")
    voc.to_csv(OUT_DIR / "eer_by_vocoder_family_ranked.csv", float_format="%.4f")

    print("architecture, ranked by the all-model Mean:")
    print(arch.iloc[-1].round(1).to_string())
    print("\nvocoder family, ranked by the all-model Mean:")
    print(voc.iloc[-1].round(1).to_string())

    plot(arch, OUT_DIR / "mlaad_tts_eer_by_architecture_ranked.png",
         figwidth=11, xtick_rot=35, annot_size=8)
    plot(voc, OUT_DIR / "mlaad_tts_eer_by_vocoder_family_ranked.png",
         figwidth=18, xtick_rot=55, annot_size=7)


if __name__ == "__main__":
    raise SystemExit(main())
