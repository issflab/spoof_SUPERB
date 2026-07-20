"""
Generate TTS diversity FAR heatmaps from pre-computed CSVs.

Produces five figures:
  1. tts_far_heatmap_ar.png          — FAR (%) per SSL model × AR TTS system
  2. tts_far_heatmap_nar.png         — FAR (%) per SSL model × NAR TTS system
  3. tts_far_pooled.png              — Mean FAR: AR pool | NAR pool | Overall
  4. tts_far_by_architecture.png     — Mean FAR per architecture tag group
  5. tts_far_by_vocoder.png          — Mean FAR per vocoder group

Usage
-----
    python3 scripts/create_tts_heatmaps.py \\
        --far_dir /data/ssl_anti_spoofing/asd_superb_score_files/far_results \\
        --out_dir outputs/figures
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Row separator positions (after these 0-based indices) — same as EER heatmaps
SEPARATOR_ROWS = [1, 7, 17]

ORDERED_MODELS = [
    "FBANK",
    "APC", "NPC", "Mockingjay", "TERA", "DeCoAR 2.0",
    "wav2vec", "wav2vec 2.0 Base", "wav2vec 2.0 Large",
    "HuBERT Base", "HuBERT Large", "MR-HuBERT", "XLS-R",
    "UniSpeech-SAT", "Data2Vec", "WAVLABLM", "WavLM Large",
    "SSAST", "MAE-AST-FRAME",
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Model")
    present = [m for m in ORDERED_MODELS if m in df.index]
    missing = [m for m in ORDERED_MODELS if m not in df.index]
    if missing:
        print(f"  [WARN] models missing from {os.path.basename(path)}: {missing}")
    return df.loc[present]


def draw_separators(ax, rows):
    xlim = ax.get_xlim()
    for y in rows:
        ax.hlines(y, *xlim, colors="black", linewidth=1.2)


def plot_far_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: str,
    figwidth: float = 18,
    figheight: float = 7,
) -> None:
    """
    Full FAR matrix heatmap (SSL models × TTS systems).
    Columns are rotated 90° to fit wide matrices.
    """
    n_cols = len(df.columns)
    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    sns.set(style="white", font_scale=0.75)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        df,
        annot=True, fmt=".0f",
        cmap=cmap,
        vmin=0, vmax=100,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "FAR (%)"},
        ax=ax,
        annot_kws={"size": 5.5},
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.tick_params(axis="x", labelsize=7, rotation=90)

    draw_separators(ax, SEPARATOR_ROWS)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_grouped_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: str,
    figwidth: float = 10,
    figheight: float = 7,
    col_fontsize: int = 9,
) -> None:
    """
    Compact heatmap for pooled / architecture / vocoder breakdowns.
    """
    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    sns.set(style="white", font_scale=0.85)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        df,
        annot=True, fmt=".1f",
        cmap=cmap,
        vmin=0, vmax=100,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "FAR (%)"},
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    ax.tick_params(axis="x", labelsize=col_fontsize, rotation=30)

    draw_separators(ax, SEPARATOR_ROWS)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate TTS diversity FAR heatmaps."
    )
    parser.add_argument(
        "--far_dir", required=True,
        help="Directory containing FAR CSV files from compute_far_matrix.py.",
    )
    parser.add_argument(
        "--out_dir", default="outputs/figures",
        help="Directory for output PNG files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    far_dir = Path(args.far_dir)

    print("Loading FAR matrices ...")

    ar_df   = load_csv(far_dir / "far_matrix_ar.csv")
    nar_df  = load_csv(far_dir / "far_matrix_nar.csv")
    pool_df = load_csv(far_dir / "far_pooled.csv")
    arch_df = load_csv(far_dir / "far_by_architecture.csv")
    voc_df  = load_csv(far_dir / "far_by_vocoder.csv")

    print("\nGenerating AR TTS heatmap ...")
    plot_far_heatmap(
        ar_df,
        title="FAR (%) Across Autoregressive TTS Systems",
        out_path=os.path.join(args.out_dir, "tts_far_heatmap_ar.png"),
        figwidth=20, figheight=7,
    )

    print("Generating NAR TTS heatmap ...")
    plot_far_heatmap(
        nar_df,
        title="FAR (%) Across Non-Autoregressive TTS Systems",
        out_path=os.path.join(args.out_dir, "tts_far_heatmap_nar.png"),
        figwidth=17, figheight=7,
    )

    print("Generating pooled AR vs NAR heatmap ...")
    plot_grouped_heatmap(
        pool_df,
        title="Mean FAR (%) — AR vs. NAR Pooled",
        out_path=os.path.join(args.out_dir, "tts_far_pooled.png"),
        figwidth=7, figheight=7,
        col_fontsize=10,
    )

    print("Generating architecture breakdown heatmap ...")
    plot_grouped_heatmap(
        arch_df,
        title="Mean FAR (%) by Architecture Type",
        out_path=os.path.join(args.out_dir, "tts_far_by_architecture.png"),
        figwidth=10, figheight=7,
        col_fontsize=9,
    )

    print("Generating vocoder breakdown heatmap ...")
    n_voc = len(voc_df.columns)
    plot_grouped_heatmap(
        voc_df,
        title="Mean FAR (%) by Vocoder",
        out_path=os.path.join(args.out_dir, "tts_far_by_vocoder.png"),
        figwidth=max(10, n_voc * 0.8),
        figheight=7,
        col_fontsize=8,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
