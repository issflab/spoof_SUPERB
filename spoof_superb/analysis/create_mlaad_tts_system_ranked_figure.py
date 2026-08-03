"""
Paper-ready per-system MLAAD TTS heatmap: systems on rows, representative SSL
models on columns, systems ranked by difficulty.

Why this exists
---------------
`eer_by_tts_system.png` is 22 SSL models x 91 TTS systems. With 91 columns the
system names must rotate vertical and the figure is unreadable at column width.
For the paper we instead:

  - transpose: 91 systems become ROWS (names stay horizontal and scannable),
  - keep ALL 91 systems (TTS diversity is the contribution -- do not subset them),
  - show only the 6 representative SSL models used in the paper's other TTS
    figures (the SSL models are the instrument, not the object of study),
  - add a `Mean` column over EVERY model scored (not just the 6 shown), and
    SORT systems by it -- the Mean is a claim about the system, so it uses all
    the evidence rather than the display sample,
  - split the ranked list into two side-by-side panels (a full-width figure*),
    reading left-to-right easiest -> hardest, sharing one colour scale,
  - carry a generation-mode strip (AR / NAR / Closed) beside each panel.

Reads the raw, unclipped EER matrix already written by
create_mlaad_tts_eer_heatmaps.py (outputs/figures_mlaad_tts/eer_by_tts_system.csv),
so it does not recompute EERs.

Usage
-----
    python -m spoof_superb.analysis.create_mlaad_tts_system_ranked_figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from spoof_superb.analysis import metadata_csv
from spoof_superb.config import cfg

SCORE_ROOT = Path(cfg.scores_root)
#: Corpus metadata, resolved from the repo (see analysis.metadata_csv).
ARCH_NAME = "mlaad_v10_tts_architecture_groups.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures_mlaad_tts"
EER_CSV = OUT_DIR / "eer_by_tts_system.csv"

# Representative SSL models, one per performance tier, in a fixed weakest -> best
# reading order.  FBANK (the handcrafted baseline) is deliberately excluded: it
# sits at chance for 76 of 91 systems (mean 52.4, std 2.7), so it carries no
# per-system signal and, being pinned near 50, it compresses the Mean difficulty
# gradient the figure is built to show -- dropping it widens the spread from
# 40.7 to 47.4 pp while leaving the ranking essentially unchanged (Spearman
# 0.998 vs the FBANK-inclusive order).
REPRESENTATIVE = [
    "APC",
    "wav2vec 2.0 Large",
    "HuBERT Large",
    "XLS-R",
    "WavLM Large",
    "SSAST",
]

VMAX = 50.0  # colour saturates at chance, matching the sibling figures
Y_TICK_ROTATION = 20  # gentle tilt on the system (row) labels; 0 = horizontal
X_TICK_ROTATION = 20  # SSL model (column) labels; 90 = vertical, lower = flatter
LEGEND_Y = 0.035      # mode-legend height; raise to tuck it under the labels
LEGEND_X = 0.45       # mode-legend centre, as a figure fraction (0.5 = centred)

# Generation-mode strip: colour + fixed legend order.
MODE_LABEL = {"AR": "AR", "NAR": "NAR", "unknown": "Closed"}
MODE_ORDER = ["AR", "NAR", "Closed"]
MODE_COLOR = {
    "AR": "#4C72B0",
    "NAR": "#55A868",
    "Closed": "#B0B0B0",
}


def load_matrix() -> pd.DataFrame:
    """Return the representative-SSL EER matrix indexed by TTS system.

    Rows = TTS systems, columns = the 6 representative models, plus a Mean
    column, sorted ascending by Mean (easiest first).

    The Mean is over EVERY model in the CSV, not the six shown. The six are a
    display sample -- 91 systems x 19 models is unreadable -- but the Mean
    orders the figure and is the number a reader takes away about a system's
    detectability, so it uses every model scored.

    Over the six it was biased low by 8.5 pp, because three of them (XLS-R,
    WavLM Large, HuBERT Large) are among the strongest in the roster, and it
    ranked 84 of the 91 systems differently.
    """
    if not EER_CSV.is_file():
        sys.exit(
            f"FATAL: {EER_CSV} not found. Run create_mlaad_tts_eer_heatmaps.py first."
        )
    raw = pd.read_csv(EER_CSV, index_col="Model")  # 22 models x 91 systems

    missing = [m for m in REPRESENTATIVE if m not in raw.index]
    if missing:
        sys.exit(f"FATAL: representative models absent from {EER_CSV}: {missing}")

    # systems x representative-models
    mat = raw.loc[REPRESENTATIVE].T
    if mat.isna().any().any():
        bad = mat.index[mat.isna().any(axis=1)].tolist()
        sys.exit(f"FATAL: NaN EER for systems {bad}")

    mat[f"Mean (all {len(raw)})"] = raw.mean(axis=0)
    mat = mat.sort_values(f"Mean (all {len(raw)})", ascending=True)
    return mat


def mode_lookup() -> dict[str, str]:
    arch = pd.read_csv(metadata_csv(ARCH_NAME))
    return {
        s: MODE_LABEL.get(m, m)
        for s, m in zip(arch["tts_system"], arch["ar_nar"])
    }


def draw_panel(
    ax_strip,
    ax_heat,
    block: pd.DataFrame,
    modes: dict[str, str],
    cbar_ax,
    show_cbar: bool,
) -> None:
    """One ranked panel: a mode strip + the SSL/Mean heatmap for `block`."""
    # --- generation-mode strip -------------------------------------------
    codes = np.array([[MODE_ORDER.index(modes.get(s, "AR"))] for s in block.index])
    strip_cmap = ListedColormap([MODE_COLOR[m] for m in MODE_ORDER])
    ax_strip.imshow(codes, aspect="auto", cmap=strip_cmap, vmin=0, vmax=len(MODE_ORDER) - 1)
    ax_strip.set_xticks([0])
    ax_strip.set_xticklabels(
        ["Mode"], fontsize=7, rotation=X_TICK_ROTATION, ha="right")
    ax_strip.set_yticks(range(len(block)))
    ax_strip.set_yticklabels(
        block.index,
        fontsize=6.2,
        rotation=Y_TICK_ROTATION,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    ax_strip.tick_params(length=0)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    # --- EER heatmap: colour capped at VMAX, annotation shows true value --
    display_color = block.clip(upper=VMAX)
    annot = block.round(1).astype(str).values
    sns.heatmap(
        display_color,
        annot=annot,
        fmt="",
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        vmin=0,
        vmax=VMAX,
        linewidths=0.3,
        linecolor="white",
        cbar=show_cbar,
        cbar_ax=cbar_ax if show_cbar else None,
        cbar_kws={"label": "EER (%, colour capped at 50)"} if show_cbar else None,
        ax=ax_heat,
        annot_kws={"size": 5.4},
        yticklabels=False,
    )
    ax_heat.set_xticklabels(
        ax_heat.get_xticklabels(), fontsize=7.5,
        rotation=X_TICK_ROTATION, ha="right")
    ax_heat.set_ylabel("")
    ax_heat.set_xlabel("")
    # separate the Mean summary column from the model columns
    ax_heat.axvline(len(REPRESENTATIVE), color="black", linewidth=1.4)


def main(argv=None) -> None:
    import argparse
    global OUT_DIR, EER_CSV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default=str(OUT_DIR),
                    help="directory holding the EER CSVs, and where the ranked "
                         "figures are written")
    args = ap.parse_args(argv)
    OUT_DIR = Path(args.out_dir)
    EER_CSV = OUT_DIR / "eer_by_tts_system.csv"

    mat = load_matrix()
    modes = mode_lookup()
    n = len(mat)
    half = (n + 1) // 2
    left, right = mat.iloc[:half], mat.iloc[half:]
    mean_col = mat.columns[-1]
    print(f"{n} systems, ranked by {mean_col}: "
          f"{mat[mean_col].iloc[0]:.1f} (easiest: {mat.index[0]}) .. "
          f"{mat[mean_col].iloc[-1]:.1f} (hardest: {mat.index[-1]})")
    print(f"split: left panel ranks 1-{half}, right panel {half + 1}-{n}")

    fig = plt.figure(figsize=(14.5, 11))
    gs = fig.add_gridspec(
        1, 6,
        width_ratios=[0.22, 8, 1.3, 0.22, 8, 0.32],
        wspace=0.05,
    )
    ax_strip_l = fig.add_subplot(gs[0, 0])
    ax_heat_l = fig.add_subplot(gs[0, 1])
    ax_strip_r = fig.add_subplot(gs[0, 3])
    ax_heat_r = fig.add_subplot(gs[0, 4])
    cbar_ax = fig.add_subplot(gs[0, 5])

    draw_panel(ax_strip_l, ax_heat_l, left, modes, cbar_ax=None, show_cbar=False)
    draw_panel(ax_strip_r, ax_heat_r, right, modes, cbar_ax=cbar_ax, show_cbar=True)

    mode_legend = [Patch(facecolor=MODE_COLOR[m], label=m) for m in MODE_ORDER]
    fig.legend(
        handles=mode_legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(LEGEND_X, LEGEND_Y),
    )
    # No suptitle: the figure carries a LaTeX caption in the paper.

    out_png = OUT_DIR / "eer_by_tts_system_ranked.png"
    out_csv = OUT_DIR / "eer_by_tts_system_ranked.csv"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ranked = mat.copy()
    ranked.insert(0, "rank", range(1, n + 1))
    ranked.insert(1, "generation_mode", [modes.get(s, "AR") for s in ranked.index])
    ranked.to_csv(out_csv, float_format="%.4f", index_label="tts_system")

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    raise SystemExit(main())
