"""
Generate TTS diversity EER heatmaps from pre-computed CSVs.

EER (Equal Error Rate) measures separability of each TTS system from the
pooled bonafide reference: lower EER = easier to detect (more separable),
higher EER = harder to detect (more bonafide-like).

Produces five figures:
  1. tts_eer_heatmap_ar.png          — EER (%) per SSL model × AR TTS system
  2. tts_eer_heatmap_nar.png         — EER (%) per SSL model × NAR TTS system
  3. tts_eer_pooled.png              — Mean EER: AR pool | NAR pool | Overall
  4. tts_eer_by_architecture.png     — Mean EER per architecture tag group
  5. tts_eer_by_vocoder.png          — Mean EER per vocoder group

Usage
-----
    python3 scripts/create_tts_eer_heatmaps.py \\
        --eer_dir /data/ssl_anti_spoofing/asd_superb_score_files/eer_results_pooled_bonafide \\
        --out_dir outputs/figures_eer_pooled
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SEPARATOR_ROWS = [1, 7, 17]

ORDERED_MODELS = [
    "FBANK",
    "APC", "NPC", "Mockingjay", "TERA", "DeCoAR 2.0",
    "wav2vec", "wav2vec 2.0 Base", "wav2vec 2.0 Large",
    "HuBERT Base", "HuBERT Large", "MR-HuBERT", "XLS-R",
    "UniSpeech-SAT", "Data2Vec", "WAVLABLM", "WavLM Large",
    "SSAST", "MAE-AST-FRAME",
]

# Representative subset: one per performance tier, spanning baseline → best
SUBSET_MODELS = [
    "FBANK",           # baseline / worst
    "APC",             # best generative
    "wav2vec 2.0 Large",  # mid-range discriminative
    "HuBERT Large",    # strong discriminative
    "XLS-R",           # best overall
    "WavLM Large",     # second best
    "SSAST",           # spectrogram-based
]
SUBSET_SEPARATOR_ROWS = [1, 2]  # after FBANK; after APC


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


def plot_eer_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: str,
    figwidth: float = 18,
    figheight: float = 7,
    _sep_rows: list = None,
    _annot_size: float = 5.5,
    _mean_source_df: pd.DataFrame = None,
) -> None:
    """EER matrix heatmap (SSL models × TTS systems) with a Mean row.

    EER is capped at 50% for display (EER > 50% = worse than random / detection
    failure; all such cells appear dark red). Raw values are preserved in CSVs.
    The bottom row shows the mean EER across all SSL models per TTS system.

    _sep_rows: override separator positions (defaults to SEPARATOR_ROWS + [len(df)])
    _annot_size: annotation font size (larger for subset heatmaps)
    _mean_source_df: if provided, compute the Mean row from this df instead of df
                     (used by subset heatmaps so Mean reflects all 19 SSL models)
    """
    # Mean row computed from full model set when a source override is given
    mean_source = _mean_source_df if _mean_source_df is not None else df
    mean_row = mean_source.mean(axis=0).rename("Mean")
    df_with_mean = pd.concat([df, mean_row.to_frame().T])

    # Clip for colour mapping and annotation (50% = chance level)
    display_df = df_with_mean.clip(upper=50.0)

    # Separator above the Mean row in addition to the model-group separators
    base_seps = _sep_rows if _sep_rows is not None else SEPARATOR_ROWS
    sep_rows = base_seps + [len(df)]

    fig, ax = plt.subplots(figsize=(figwidth, figheight + 0.4))

    sns.set(style="white", font_scale=0.75)
    # YlOrRd: yellow = low EER (easy to detect), red = high EER (hard to detect)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        display_df,
        annot=True, fmt=".1f",
        cmap=cmap,
        vmin=0, vmax=50,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "EER (%, capped at 50)"},
        ax=ax,
        annot_kws={"size": _annot_size},
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.tick_params(axis="x", labelsize=7, rotation=90)

    draw_separators(ax, sep_rows)

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
    """Compact heatmap for pooled / architecture / vocoder breakdowns."""
    display_df = df.clip(upper=50.0)

    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    sns.set(style="white", font_scale=0.85)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        display_df,
        annot=True, fmt=".1f",
        cmap=cmap,
        vmin=0, vmax=50,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "EER (%, capped at 50)"},
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
        description="Generate TTS diversity EER heatmaps (pooled bonafide)."
    )
    parser.add_argument(
        "--eer_dir", required=True,
        help="Directory containing EER CSV files from compute_eer_tts.py.",
    )
    parser.add_argument(
        "--out_dir", default="outputs/figures_eer_pooled",
        help="Directory for output PNG files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eer_dir = Path(args.eer_dir)

    print("Loading EER matrices ...")
    ar_df   = load_csv(eer_dir / "eer_matrix_ar.csv")
    nar_df  = load_csv(eer_dir / "eer_matrix_nar.csv")
    pool_df = load_csv(eer_dir / "eer_pooled.csv")
    arch_df = load_csv(eer_dir / "eer_by_architecture.csv")
    voc_df  = load_csv(eer_dir / "eer_by_vocoder.csv")

    print("\nGenerating AR TTS EER heatmap ...")
    plot_eer_heatmap(
        ar_df,
        title="EER (%) vs Pooled Bonafide — Autoregressive TTS Systems",
        out_path=os.path.join(args.out_dir, "tts_eer_heatmap_ar.png"),
        figwidth=20, figheight=7,
    )

    print("Generating NAR TTS EER heatmap ...")
    plot_eer_heatmap(
        nar_df,
        title="EER (%) vs Pooled Bonafide — Non-Autoregressive TTS Systems",
        out_path=os.path.join(args.out_dir, "tts_eer_heatmap_nar.png"),
        figwidth=17, figheight=7,
    )

    print("Generating pooled AR vs NAR heatmap ...")
    plot_grouped_heatmap(
        pool_df,
        title="Mean EER (%) vs Pooled Bonafide — AR vs. NAR",
        out_path=os.path.join(args.out_dir, "tts_eer_pooled.png"),
        figwidth=7, figheight=7,
        col_fontsize=10,
    )

    print("Generating architecture breakdown heatmap ...")
    plot_grouped_heatmap(
        arch_df,
        title="Mean EER (%) by Architecture Type",
        out_path=os.path.join(args.out_dir, "tts_eer_by_architecture.png"),
        figwidth=10, figheight=7,
        col_fontsize=9,
    )

    # --- Subset heatmaps (representative SSL models only) ---
    ar_sub  = ar_df.loc[[m for m in SUBSET_MODELS if m in ar_df.index]]
    nar_sub = nar_df.loc[[m for m in SUBSET_MODELS if m in nar_df.index]]

    print("\nGenerating AR TTS EER heatmap (subset) ...")
    plot_eer_heatmap(
        ar_sub,
        title="EER (%) vs Pooled Bonafide — Autoregressive TTS Systems (Representative SSL Models)",
        out_path=os.path.join(args.out_dir, "tts_eer_heatmap_ar_subset.png"),
        figwidth=20, figheight=4,
        _sep_rows=SUBSET_SEPARATOR_ROWS,
        _annot_size=8.0,
        _mean_source_df=ar_df,
    )

    print("Generating NAR TTS EER heatmap (subset) ...")
    plot_eer_heatmap(
        nar_sub,
        title="EER (%) vs Pooled Bonafide — Non-Autoregressive TTS Systems (Representative SSL Models)",
        out_path=os.path.join(args.out_dir, "tts_eer_heatmap_nar_subset.png"),
        figwidth=17, figheight=4,
        _sep_rows=SUBSET_SEPARATOR_ROWS,
        _annot_size=8.0,
        _mean_source_df=nar_df,
    )

    print("Generating vocoder breakdown heatmap ...")
    n_voc = len(voc_df.columns)
    plot_grouped_heatmap(
        voc_df,
        title="Mean EER (%) by Vocoder",
        out_path=os.path.join(args.out_dir, "tts_eer_by_vocoder.png"),
        figwidth=max(10, n_voc * 0.8),
        figheight=7,
        col_fontsize=8,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
