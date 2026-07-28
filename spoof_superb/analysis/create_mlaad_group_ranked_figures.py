"""
Ranked architecture-group and vocoder-family MLAAD heatmaps for the paper.

Consistency with the per-system figure (create_mlaad_tts_system_ranked_figure.py):
both figures use the SAME six representative SSL models and rank the grouping
columns by the Mean row (mean EER over those six models), easiest to hardest.

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

SCORE_ROOT = Path("/data/ssl_anti_spoofing/asd_superb_score_files")
ARCH_CSV = SCORE_ROOT / "mlaad_v10_tts_architecture_groups.csv"
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
    arch = pd.read_csv(ARCH_CSV)
    arch = arch[~arch["tts_system"].isin(EXCLUDED)]
    return arch[col].value_counts().to_dict()


def ranked_frame(csv_name: str, sizes: dict[str, int]) -> pd.DataFrame:
    """6 representative models x groups, sorted by the Mean row, Mean appended.

    Returns rows = [6 models, "Mean"], columns = groups ordered easiest->hardest,
    column labels annotated with "(n)".
    """
    df = pd.read_csv(OUT_DIR / csv_name, index_col="Model")
    missing = [m for m in REPRESENTATIVE if m not in df.index]
    if missing:
        sys.exit(f"FATAL: {csv_name} missing models {missing}")
    sub = df.loc[REPRESENTATIVE]
    mean = sub.mean(axis=0)
    order = mean.sort_values().index
    sub = sub[order]
    sub.loc["Mean"] = mean[order]
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


def main() -> None:
    arch = ranked_frame("eer_by_architecture.csv", group_sizes("architecture_group"))
    voc = ranked_frame("eer_by_vocoder_family.csv", group_sizes("vocoder_family"))

    arch.to_csv(OUT_DIR / "eer_by_architecture_ranked.csv", float_format="%.4f")
    voc.to_csv(OUT_DIR / "eer_by_vocoder_family_ranked.csv", float_format="%.4f")

    print("architecture, ranked by 6-model Mean:")
    print(arch.loc["Mean"].round(1).to_string())
    print("\nvocoder family, ranked by 6-model Mean:")
    print(voc.loc["Mean"].round(1).to_string())

    plot(arch, OUT_DIR / "mlaad_tts_eer_by_architecture_ranked.png",
         figwidth=11, xtick_rot=35, annot_size=8)
    plot(voc, OUT_DIR / "mlaad_tts_eer_by_vocoder_family_ranked.png",
         figwidth=18, xtick_rot=55, annot_size=7)


if __name__ == "__main__":
    main()
