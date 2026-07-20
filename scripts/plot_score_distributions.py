"""
plot_score_distributions.py

Plots score distributions before and after z-score normalization.

Generates 4 PNG figures (2×2 subplot grid each, one panel per selected model):

  score_dist_before_by_dataset.png  — KDE per dataset (all scores), before normalization
  score_dist_before_by_class.png    — bonafide vs spoof KDE per TTS-relevant dataset, before
  score_dist_after_by_dataset.png   — same as above, after z-score normalization
  score_dist_after_by_class.png     — same as above, after z-score normalization

Selected models: XLS-R, UniSpeech-SAT, WavLM Large, wav2vec 2.0 Large

Usage
-----
    python3 scripts/plot_score_distributions.py \\
        --linear_head_dir      /data/ssl_anti_spoofing/asd_superb_score_files/linear_head \\
        --linear_head_norm_dir /data/ssl_anti_spoofing/asd_superb_score_files/linear_head_normalized_scores \\
        --out_dir              spoof_SUPERB/outputs/figures_dist
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SELECTED_STEMS = [
    "xls_r_300m",
    "unispeech_sat_large",
    "wavlm_large",
    "wav2vec2_large_ll60k",
]

STEM_DISPLAY = {
    "xls_r_300m":           "XLS-R",
    "unispeech_sat_large":  "UniSpeech-SAT",
    "wavlm_large":          "WavLM Large",
    "wav2vec2_large_ll60k": "wav2vec 2.0 Large",
}

ALL_DATASETS = [
    "eval_2019",
    "asvspoof2021_LA",
    "asvspoof2021_DF",
    "asvspoof5",
    "Famous_Figures",
    "Multilingual",
    "spoofceleb",
    "wild",
    "deepfake_eval_2024",
    "asvspoofLD",
]

DATASET_DISPLAY = {
    "eval_2019":          "ASV19",
    "asvspoof2021_LA":    "ASV21-LA",
    "asvspoof2021_DF":    "ASV21-DF",
    "asvspoof5":          "ASV5",
    "Famous_Figures":     "FamousFigures",
    "Multilingual":       "MLAAD",
    "spoofceleb":         "SpoofCeleb",
    "wild":               "ITW",
    "deepfake_eval_2024": "DFEval24",
    "asvspoofLD":         "ASVLD",
}

# 5 TTS-relevant datasets (for by-class figure)
TTS_DATASETS = ["eval_2019", "asvspoof5", "Famous_Figures", "Multilingual", "spoofceleb"]

DATASET_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

TTS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_scores_by_label(filepath: str):
    """
    Parse a linear_head score file (space-separated, 4 fields).
    Returns (bonafide_scores, spoof_scores) as numpy arrays.
    """
    bonafide, spoof = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 4:
                continue
            try:
                score = float(parts[3])
                label = parts[2]
            except (ValueError, IndexError):
                continue
            if label == "bonafide":
                bonafide.append(score)
            else:
                spoof.append(score)
    return np.array(bonafide, dtype=np.float32), np.array(spoof, dtype=np.float32)


def kde_curve(scores, n_points=500):
    """Return (x, y) for a KDE curve, clipped to [0.5th, 99.5th] percentile."""
    if len(scores) < 20:
        return None, None
    try:
        kde = gaussian_kde(scores, bw_method="scott")
        lo, hi = np.percentile(scores, 0.5), np.percentile(scores, 99.5)
        if lo >= hi:
            return None, None
        x = np.linspace(lo, hi, n_points)
        return x, kde(x)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------
def make_by_dataset_figure(linear_head_dir: str, title_suffix: str, out_path: str) -> None:
    """2×2 grid: KDE per dataset (all scores combined) for 4 selected models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax_idx, stem in enumerate(SELECTED_STEMS):
        ax = axes_flat[ax_idx]
        ax.set_title(STEM_DISPLAY[stem], fontsize=12, fontweight="bold")

        for ds_idx, dataset in enumerate(ALL_DATASETS):
            filepath = os.path.join(
                linear_head_dir, f"linear_head_{dataset}_{stem}.txt"
            )
            if not os.path.exists(filepath):
                continue

            bonafide, spoof = read_scores_by_label(filepath)
            all_scores = np.concatenate([bonafide, spoof])
            color = DATASET_COLORS[ds_idx % len(DATASET_COLORS)]

            x, y = kde_curve(all_scores)
            if x is not None:
                ax.plot(x, y, color=color, label=DATASET_DISPLAY[dataset], lw=1.8, alpha=0.85)

        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.7)

    fig.suptitle(
        f"Score Distribution by Dataset  {title_suffix}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_by_class_figure(linear_head_dir: str, title_suffix: str, out_path: str) -> None:
    """
    2×2 grid: bonafide (solid) vs spoof (dashed) KDE per TTS-relevant dataset.
    Shows where each dataset's EER crossing point lies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax_idx, stem in enumerate(SELECTED_STEMS):
        ax = axes_flat[ax_idx]
        ax.set_title(STEM_DISPLAY[stem], fontsize=12, fontweight="bold")

        for ds_idx, dataset in enumerate(TTS_DATASETS):
            filepath = os.path.join(
                linear_head_dir, f"linear_head_{dataset}_{stem}.txt"
            )
            if not os.path.exists(filepath):
                continue

            bonafide, spoof = read_scores_by_label(filepath)
            color = TTS_COLORS[ds_idx % len(TTS_COLORS)]
            label = DATASET_DISPLAY[dataset]

            xb, yb = kde_curve(bonafide)
            if xb is not None:
                ax.plot(xb, yb, color=color, linestyle="-",
                        label=f"{label} bonafide", lw=2.2, alpha=0.9)

            xs, ys = kde_curve(spoof)
            if xs is not None:
                ax.plot(xs, ys, color=color, linestyle="--",
                        label=f"{label} spoof", lw=1.6, alpha=0.65)

        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(labelsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=7.5, loc="best", ncol=1, framealpha=0.7)

    fig.suptitle(
        f"Score Distribution: Bonafide vs Spoof by Dataset  {title_suffix}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot score distributions before and after z-score normalization."
    )
    parser.add_argument(
        "--linear_head_dir", required=True,
        help="Raw linear_head score files (before normalization).",
    )
    parser.add_argument(
        "--linear_head_norm_dir", required=True,
        help="Z-score normalized linear_head score files (after normalization).",
    )
    parser.add_argument(
        "--out_dir", default="spoof_SUPERB/outputs/figures_dist",
        help="Output directory for PNG figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating before-normalization figures ...")
    make_by_dataset_figure(
        args.linear_head_dir,
        "(Before Z-Score)",
        os.path.join(args.out_dir, "score_dist_before_by_dataset.png"),
    )
    make_by_class_figure(
        args.linear_head_dir,
        "(Before Z-Score)",
        os.path.join(args.out_dir, "score_dist_before_by_class.png"),
    )

    print("Generating after-normalization figures ...")
    make_by_dataset_figure(
        args.linear_head_norm_dir,
        "(After Z-Score)",
        os.path.join(args.out_dir, "score_dist_after_by_dataset.png"),
    )
    make_by_class_figure(
        args.linear_head_norm_dir,
        "(After Z-Score)",
        os.path.join(args.out_dir, "score_dist_after_by_class.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
