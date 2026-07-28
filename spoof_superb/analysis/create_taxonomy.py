from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_PATH = Path("outputs/figures/audio_deepfake_taxonomy.png")
Y_OFFSET = 0.8


def add_round_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=14, weight="normal"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.3,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        family="sans-serif",
        color="#1a1a1a",
        zorder=3,
    )
    return box


def add_arrow(ax, start, end, color, lw=3.0, scale=28, connectionstyle="arc3"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="simple",
        mutation_scale=scale,
        linewidth=0.8,
        facecolor=color,
        edgecolor=color,
        connectionstyle=connectionstyle,
        zorder=2.5,
    )
    ax.add_patch(arrow)
    return arrow


def draw_signal(ax, x, y, width=0.55, height=0.55):
    bars = [0.18, 0.38, 0.62, 0.84, 1.0, 0.84, 0.62, 0.38, 0.18]
    step = width / (len(bars) - 1)
    for idx, scale in enumerate(bars):
        xpos = x + idx * step
        y0 = y + (height * (1 - scale)) / 2
        y1 = y + height - (height * (1 - scale)) / 2
        ax.plot(
            [xpos, xpos],
            [y0, y1],
            color="#111111",
            lw=2.5,
            solid_capstyle="round",
            zorder=3,
        )
    ax.plot([x - 0.06, x - 0.02], [y + height / 2, y + height / 2], color="#111111", lw=2.5, solid_capstyle="round")
    ax.plot([x + width + 0.02, x + width + 0.06], [y + height / 2, y + height / 2], color="#111111", lw=2.5, solid_capstyle="round")


def add_input_block(ax, center_y):
    draw_signal(ax, x=0.42, y=center_y - 0.28)
    ax.text(
        0.75,
        center_y - 0.55,
        "Input Speech\nSignal",
        ha="center",
        va="top",
        fontsize=13,
        family="sans-serif",
        color="#1a1a1a",
    )
    add_arrow(ax, (1.1, center_y), (1.72, center_y), "#2f66c6", lw=2.8, scale=24)


def add_section_title(ax, y, text):
    ax.text(
        4.8,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=16,
        weight="bold",
        family="sans-serif",
        color="#1a1a1a",
    )


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12.6)
    ax.set_ylim(-1.2, 10.1)
    ax.axis("off")

    # Row 1: Conventional machine learning
    y1 = 7.8 + Y_OFFSET
    add_section_title(ax, 8.4 + Y_OFFSET, "Conventional Machine Learning")
    add_input_block(ax, y1)
    add_round_box(ax, 2.0, 7.25 + Y_OFFSET, 3.0, 0.95, "Hand Crafted\nFeature Extraction", "#d9d9d9", "#9c9c9c")
    add_arrow(ax, (5.08, y1), (5.6, y1), "#d7d7d7", scale=22)
    add_round_box(ax, 5.65, 7.25 + Y_OFFSET, 2.35, 0.95, "Classification", "#f9b38f", "#ff7f2a")
    add_arrow(ax, (8.1, y1), (8.8, y1), "#ff9f43", scale=22)
    add_round_box(ax, 8.9, 7.25 + Y_OFFSET, 2.6, 0.95, "Prediction\nReal / Spoof", "#9fb6e9", "#3e6ccb")

    # Row 2: End-to-end
    y2 = 4.95 + Y_OFFSET
    add_section_title(ax, 5.5 + Y_OFFSET, "End-to-End Deep Learning Approaches")
    add_input_block(ax, y2)
    add_round_box(ax, 2.0, 4.4 + Y_OFFSET, 6.0, 0.95, "Deep Feature Extraction + Classification", "#f8dd95", "#ffbc00")
    add_arrow(ax, (8.1, y2), (8.85, y2), "#ffcc19", scale=22)
    add_round_box(ax, 8.9, 4.4 + Y_OFFSET, 2.6, 0.95, "Prediction\nReal / Spoof", "#9fb6e9", "#3e6ccb")

    # Row 3: Hybrid
    y3 = 2.35 + Y_OFFSET
    add_section_title(ax, 2.95 + Y_OFFSET, "Hybrid Deep Learning Approaches")
    add_input_block(ax, y3)
    add_round_box(ax, 2.15, 1.85 + Y_OFFSET, 4.3, 0.95, "Hand Crafted / Deep Feature\nExtraction", "#acd18d", "#6cab4e")
    add_arrow(ax, (6.55, y3), (7.1, y3), "#75bf58", scale=22)
    add_round_box(ax, 7.15, 1.85 + Y_OFFSET, 2.45, 0.95, "Deep Classification", "#f4b08d", "#f17b36")
    add_arrow(ax, (9.7, y3), (10.2, y3), "#f5a064", scale=22)
    add_round_box(ax, 10.25, 1.85 + Y_OFFSET, 2.1, 0.95, "Prediction\nReal / Spoof", "#9fb6e9", "#3e6ccb", fontsize=13)

    # Feedback loop between hybrid stages
    # add_arrow(
    #     ax,
    #     (9.45, 1.48),
    #     (6.15, 2.18),
    #     "#75bf58",
    #     scale=18,
    #     connectionstyle="arc3,rad=0.45",
    # )

    # Row 4: SSL-based hybrid
    y4 = -0.2 + Y_OFFSET
    ax.add_patch(
        FancyBboxPatch(
            (2.0, -0.9 + Y_OFFSET),
            7.85,
            1.35,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.4,
            facecolor="#edf9f5",
            edgecolor="#5bb8aa",
            linestyle="--",
            zorder=0.5,
        )
    )
    add_section_title(ax, 0.6 + Y_OFFSET, "SSL-Based Approaches")
    add_input_block(ax, y4)
    add_round_box(ax, 2.15, -0.7 + Y_OFFSET, 4.3, 0.95, "Self-Supervised Learning\n(SSL) Embeddings", "#9fd7cb", "#2f9b8f")
    add_arrow(ax, (6.55, y4), (7.1, y4), "#43b8a6", scale=22)
    add_round_box(ax, 7.15, -0.7 + Y_OFFSET, 2.45, 0.95, "Deep Classification", "#f4b08d", "#f17b36")
    add_arrow(ax, (9.7, y4), (10.2, y4), "#f5a064", scale=22)
    add_round_box(ax, 10.25, -0.7 + Y_OFFSET, 2.1, 0.95, "Prediction\nReal / Spoof", "#9fb6e9", "#3e6ccb", fontsize=13)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
