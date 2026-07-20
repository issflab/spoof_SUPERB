"""
Generate acoustic-degradation EER heatmaps from a pre-computed CSV.

Expected CSV layout (produced by compute_eer_matrix.py):
    Model, Baseline, Codec, Noise, Resampling, Reverb, Channel

Two output figures:
  1. acoustic_eer_heatmap_absolute_eer_categorized.png
  2. acoustic_eer_heatmap_relative_eer_categorized.png

Usage
-----
    python3 scripts/create_heatmap.py \\
        --csv /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_category_augmented/eer_matrix.csv \\
        --out_dir outputs/figures
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Model display order and category boundaries
# ---------------------------------------------------------------------------
ORDERED_MODELS = [
    "FBANK",
    "APC", "NPC", "Mockingjay-960h", "TERA", "DeCoAR 2.0",
    "wav2vec", "wav2vec 2.0 Base", "wav2vec 2.0 Large",
    "HuBERT Base", "HuBERT Large", "MR-HuBERT", "XLS-R",
    "UniSpeech-SAT", "Data2Vec", "WAVLABLM", "WavLM Large",
    "SSAST", "MAE-AST-FRAME",
]

# Horizontal separator positions (drawn after these row indices, 0-based)
SEPARATOR_ROWS = [1, 7, 17]

DEGRADATION_COLS = ["Codec", "Noise", "Resampling", "Reverb", "Channel"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_matrix(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col="Model")
    # Keep only the models in the defined order; drop any extras
    present = [m for m in ORDERED_MODELS if m in df.index]
    missing = [m for m in ORDERED_MODELS if m not in df.index]
    if missing:
        print(f"  [WARN] models missing from CSV: {missing}")
    return df.loc[present]


def draw_separators(axes, rows):
    for ax in axes:
        xlim = ax.get_xlim()
        for y in rows:
            ax.hlines(y, *xlim, colors="black", linewidth=1.5)


# ---------------------------------------------------------------------------
# Absolute EER heatmap
# ---------------------------------------------------------------------------

def plot_absolute(df: pd.DataFrame, out_path: str) -> None:
    df = df.copy()
    df["Mean"] = df[DEGRADATION_COLS].mean(axis=1)

    sns.set(style="white", font_scale=0.9)
    side_cmap = sns.light_palette("gray", as_cmap=True)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 5.4, 1.15], wspace=0.06)

    ax0 = fig.add_subplot(gs[0, 0])   # Baseline
    ax1 = fig.add_subplot(gs[0, 1])   # Degradation conditions
    ax2 = fig.add_subplot(gs[0, 2])   # Mean

    sns.heatmap(
        df[["Baseline"]],
        annot=True, fmt=".1f", cmap=side_cmap, cbar=False,
        linewidths=0.5, linecolor="white",
        ax=ax0, yticklabels=df.index,
    )
    sns.heatmap(
        df[DEGRADATION_COLS],
        annot=True, fmt=".1f", cmap="YlGnBu",
        cbar_kws={"label": "EER (%)"},
        linewidths=0.5, linecolor="white",
        ax=ax1, yticklabels=False,
    )
    sns.heatmap(
        df[["Mean"]],
        annot=True, fmt=".1f", cmap=side_cmap, cbar=False,
        linewidths=0.5, linecolor="white",
        ax=ax2, yticklabels=False,
    )

    ax0.set_title("Baseline", fontsize=11)
    ax1.set_title("Acoustic Degradations", fontsize=11)
    ax2.set_title("Mean", fontsize=11)

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=9, rotation=0)

    ax0.tick_params(axis="y", labelsize=10, rotation=0)

    draw_separators([ax0, ax1, ax2], SEPARATOR_ROWS)

    plt.suptitle("EER Across Acoustic Degradations", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.06, 0, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Relative EER heatmap
# ---------------------------------------------------------------------------

def plot_relative(df: pd.DataFrame, out_path: str) -> None:
    baseline = df["Baseline"]
    rel = pd.DataFrame(index=df.index)
    for col in DEGRADATION_COLS:
        rel[col] = (df[col] - baseline) / baseline * 100
    rel["Mean"] = rel[DEGRADATION_COLS].mean(axis=1)

    sns.set(style="white", font_scale=0.9)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.6, 1.2], wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sns.heatmap(
        rel[DEGRADATION_COLS],
        annot=True, fmt=".1f",
        cmap="RdBu_r", center=0, vmin=-100, vmax=150,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Relative Change (%)"},
        ax=ax1,
    )
    sns.heatmap(
        rel[["Mean"]],
        annot=True, fmt=".1f",
        cmap=sns.light_palette("gray", as_cmap=True),
        cbar=False,
        linewidths=0.5, linecolor="white",
        ax=ax2, yticklabels=False,
    )

    ax1.set_title("Relative EER Change (Degradation Robustness)", fontsize=11)
    ax2.set_title("Mean", fontsize=11)

    for ax in [ax1, ax2]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=9, rotation=0)

    ax1.tick_params(axis="y", labelsize=10, rotation=0)

    draw_separators([ax1, ax2], SEPARATOR_ROWS)

    plt.suptitle("Relative EER Change Across Acoustic Degradations", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate EER heatmaps from a pre-computed CSV."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to eer_matrix.csv produced by compute_eer_matrix.py.",
    )
    parser.add_argument(
        "--out_dir", default="outputs/figures",
        help="Directory for output PNG files (default: outputs/figures).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading EER matrix ...")
    df = load_matrix(args.csv)
    print(df.to_string())

    print("\nGenerating absolute EER heatmap ...")
    plot_absolute(
        df,
        os.path.join(args.out_dir, "acoustic_eer_heatmap_absolute_eer_categorized.png"),
    )

    print("Generating relative EER heatmap ...")
    plot_relative(
        df,
        os.path.join(args.out_dir, "acoustic_eer_heatmap_relative_eer_categorized.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
