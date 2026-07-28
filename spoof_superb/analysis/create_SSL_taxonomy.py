from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_PATH = Path("outputs/figures/ssl_taxonomy_by_objective.png")


def add_card(ax, x, y, w, h, title, summary, objective, examples, facecolor, edgecolor):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.8,
            facecolor=facecolor,
            edgecolor=edgecolor,
            zorder=1,
        )
    )
    wrapped_summary = textwrap.fill(summary, width=34)
    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="top",
        fontsize=15.5,
        weight="bold",
        color="#182026",
        family="sans-serif",
        zorder=2,
    )
    ax.text(
        x + 0.22,
        y + h - 0.88,
        wrapped_summary,
        ha="left",
        va="top",
        fontsize=11.8,
        color="#24323d",
        family="sans-serif",
        linespacing=1.55,
        zorder=2,
    )

    ax.text(
        x + 0.22,
        y + 1.34,
        "Learning Objective",
        ha="left",
        va="bottom",
        fontsize=10.5,
        weight="bold",
        color="#4f6170",
        family="sans-serif",
        zorder=2,
    )
    add_badge(ax, x + 0.22, y + 0.9, objective, facecolor="#ffffff", edgecolor=edgecolor, width=1.8)

    ax.plot(
        [x + 0.22, x + w - 0.22],
        [y + 0.78, y + 0.78],
        color=edgecolor,
        lw=1.0,
        alpha=0.45,
        zorder=2,
    )
    ax.text(
        x + 0.22,
        y + 0.62,
        f"Examples\n{examples}",
        ha="left",
        va="top",
        fontsize=11.3,
        color="#24323d",
        family="sans-serif",
        linespacing=1.5,
        zorder=2,
    )


def add_badge(ax, x, y, text, facecolor, edgecolor, width=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0,
            facecolor=facecolor,
            edgecolor=edgecolor,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + 0.17,
        text,
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
        color="#22313d",
        family="sans-serif",
        zorder=3,
    )


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 7.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(
        7.5,
        7.0,
        "Self-Supervised Speech Models",
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color="#17212b",
        family="sans-serif",
    )
    ax.text(
        7.5,
        6.6,
        "Taxonomy by Learning Objective",
        ha="center",
        va="center",
        fontsize=15,
        color="#5d6b77",
        family="sans-serif",
    )

    add_card(
        ax,
        0.8,
        1.35,
        4.15,
        4.35,
        "Generative Models",
        "Learn representations by reconstructing the input signal or predicting masked and future content.",
        "Reconstruction",
        "APC, VQ-APC, Mockingjay, NPC",
        "#e8f0fb",
        "#7aa4df",
    )

    add_card(
        ax,
        5.42,
        1.35,
        4.15,
        4.35,
        "Discriminative Models",
        "Learn robust embeddings by distinguishing targets from distractors using contrastive or predictive objectives.",
        "Contrastive",
        "Wav2Vec, HuBERT, WavLM, XLS-R",
        "#e8f7ef",
        "#77bf93",
    )

    add_card(
        ax,
        10.04,
        1.35,
        4.15,
        4.35,
        "Hybrid Models",
        "Combine reconstruction and discriminative signals to capture both local detail and global semantic structure.",
        "Mixed Objective",
        "SSAST, MAE-AST",
        "#fff1e6",
        "#e59b57",
    )

    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
