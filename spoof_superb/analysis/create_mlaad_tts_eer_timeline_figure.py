"""
Detection difficulty against system introduction date: 91 MLAAD v10 TTS systems
ordered oldest -> newest on the x axis, mean EER on the y axis.

Why this exists
---------------
The per-system figure (`eer_by_tts_system_ranked.png`) orders systems by
difficulty, which answers "which systems are hard" but not "are newer systems
harder".  This figure re-orders the same EERs by introduction date so the time
axis becomes readable.

Design decisions, and why:

  - x is CATEGORICAL, not a true time axis.  The request is one labelled tick
    per system; a real date axis would collapse the 2025 cluster into an
    unreadable pile and hide the systems entirely.  Ticks are ordered by date
    and carry both the name and the month, so the ordering is still visible.
  - y is the mean EER over ALL 19 SSL models, matching the `Mean` column of the
    paper's per-system figure.  Difficulty is a claim about the system, so it
    uses every model scored rather than a display subset.
  - the shaded band is the interquartile range across those 19 models.  The
    models disagree substantially (mean pairwise Spearman 0.60), so a bare mean
    would overstate how well-determined each system's difficulty is.
  - points are coloured AND shaped by `date_type`.  TTS_METHODOLOGY.md
    Limitation 7 records that only the `publication` dates are exact arXiv
    months; `release` and `announcement` dates are softer proxies on a
    not-strictly-comparable basis, so they must stay visually separable rather
    than being averaged into one series.

Colours are categorical slots 1-3 of the reference palette.  Verified for
all-pairs scatter use: worst normal-vision dE 24.0 (floor 15), worst CVD dE 9.2
under deuteranopia (target 8).  Marker shape is a secondary encoding so identity
never rests on colour alone.

Reads the EER matrix already written by create_mlaad_tts_eer_heatmaps.py and the
dated system list from TTS_METHODOLOGY.md section 7.5.  It does not recompute
EERs.

Usage
-----
    python spoof_superb/analysis/create_mlaad_tts_eer_timeline_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TIMELINE_CSV = REPO / "spoof_superb" / "analysis" / "mlaad_v10_tts_timeline_91.csv"
EER_CSV = REPO / "spoof_superb_outputs" / "tts" / "eer_by_tts_system.csv"
OUT_DIR = REPO / "spoof_superb_outputs" / "tts"

#: Categorical slots 1-3, validated for all-pairs scatter.  Marker shape repeats
#: the distinction so the encoding survives greyscale print and CVD.
DATE_TYPE_STYLE = {
    "publication": ("#2a78d6", "o", "arXiv month (exact)"),
    "release": ("#eb6834", "s", "repo / model-card creation"),
    "announcement": ("#1baf7a", "^", "vendor announcement"),
}
DATE_TYPE_ORDER = ["publication", "release", "announcement"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
BAND = "#c9c8c2"
SURFACE = "#fcfcfb"

CHANCE_EER = 50.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, implemented here to avoid a scipy dependency."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def load(timeline_csv: Path, eer_csv: Path) -> pd.DataFrame:
    """Join the dated system list to the 19-model EER matrix, oldest first."""
    tl = pd.read_csv(timeline_csv)
    eer = pd.read_csv(eer_csv).set_index("Model")

    missing = set(tl["tts_system"]) - set(eer.columns)
    if missing:
        raise SystemExit(f"systems dated but not scored: {sorted(missing)}")
    extra = set(eer.columns) - set(tl["tts_system"])
    if extra:
        raise SystemExit(f"systems scored but not dated: {sorted(extra)}")

    per_system = eer[tl["tts_system"]]
    tl = tl.assign(
        eer_mean=per_system.mean().to_numpy(),
        eer_q1=per_system.quantile(0.25).to_numpy(),
        eer_q3=per_system.quantile(0.75).to_numpy(),
        n_models=per_system.shape[0],
    )
    # pub_month already carries YYYY-MM; sort on it, then name for stability.
    return tl.sort_values(["pub_month", "tts_system"]).reset_index(drop=True)


def plot(df: pd.DataFrame, out_png: Path) -> None:
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(26, 9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Per-point whiskers, NOT a filled band: x is categorical, so an area between
    # neighbouring systems would imply an interpolation that does not exist.
    ax.vlines(
        x, df["eer_q1"], df["eer_q3"],
        color=BAND, linewidth=3.5, zorder=1,
        label=f"interquartile range across {int(df['n_models'].iloc[0])} SSL models",
    )
    ax.axhline(CHANCE_EER, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(
        len(df) - 0.5, CHANCE_EER + 0.9, "chance (50%)",
        ha="right", va="bottom", fontsize=9, color=INK_SECONDARY,
    )

    # Trend across date rank.  Reported alongside Spearman rho because the fit is
    # on rank position, not on elapsed time.
    coef = np.polyfit(x, df["eer_mean"], 1)
    ax.plot(
        x, np.polyval(coef, x),
        color=INK_SECONDARY, linewidth=2, linestyle="-", alpha=0.75, zorder=3,
        label=f"linear trend ({coef[0]:+.2f} pp per system, oldest to newest)",
    )

    for dtype in DATE_TYPE_ORDER:
        sub = df[df["date_type"] == dtype]
        if sub.empty:
            continue
        color, marker, desc = DATE_TYPE_STYLE[dtype]
        ax.scatter(
            sub.index, sub["eer_mean"],
            s=64, marker=marker, color=color,
            edgecolors=SURFACE, linewidths=1.6, zorder=4,
            label=f"{dtype} — {desc} (n={len(sub)})",
        )

    # Direct-label only the extremes; a number on every point would be noise.
    for idx in [df["eer_mean"].idxmin(), df["eer_mean"].idxmax()]:
        row = df.loc[idx]
        ax.annotate(
            f"{row['tts_system']} {row['eer_mean']:.1f}%",
            (idx, row["eer_mean"]),
            textcoords="offset points", xytext=(0, 13 if idx == df["eer_mean"].idxmin() else -20),
            ha="center", fontsize=9.5, color=INK_PRIMARY, fontweight="bold",
        )

    # Bottom axis: one label per system, vertical.  At 30 degrees these labels
    # need ~3x the horizontal room (text length x cos30 instead of cap height),
    # which forced a 40in canvas; vertical fits the same 91 names in 26in.
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["tts_system"], rotation=90, ha="center",
        fontsize=7, color=INK_SECONDARY,
    )
    ax.set_xlim(-1, len(df))

    # Dates sit INSIDE the axes, just above the bottom spine, horizontal.  The y
    # limit is dropped first to reserve a clear strip for them, so they never
    # collide with the lowest whisker.  Systems sharing a month get ONE label
    # centred over the group (47 distinct months for 91 systems), which is what
    # lets the dates stay horizontal without overlapping.
    y_lo = float(df["eer_q1"].min())
    y_hi = float(df["eer_q3"].max())
    ax.set_ylim(y_lo - 0.13 * (y_hi - y_lo), y_hi + 0.04 * (y_hi - y_lo))

    # Two staggered rows.  In the early years most months hold a single system, so
    # consecutive labels sit 1.0 data unit apart while a "YYYY-MM" label needs
    # about 1.2 at this font size.  A greedy two-row assignment keeps every date
    # visible at this width; one row would need a ~40in canvas.
    DATE_FS = 5.5
    ROW_Y = (0.018, 0.055)  # axes fraction, inside the plot area
    axes_pt = fig.get_size_inches()[0] * 72 * 0.96
    label_w = len("YYYY-MM") * DATE_FS * 0.60 / axes_pt * (len(df) + 1)  # data units

    trans = ax.get_xaxis_transform()  # x in data coords, y in axes coords
    row_free = [-np.inf, -np.inf]  # right edge of the last label placed per row
    for month, idx in df.groupby("pub_month", sort=False).indices.items():
        cx = float(np.mean(idx))
        row = 0 if cx - label_w / 2 >= row_free[0] else 1
        row_free[row] = cx + label_w / 2
        ax.text(
            cx, ROW_Y[row], month,
            transform=trans, rotation=0, ha="center", va="bottom",
            fontsize=DATE_FS, color=INK_SECONDARY, zorder=5,
        )
        ax.plot(
            [cx, cx], [0.0, ROW_Y[row] - 0.004],
            transform=trans, color=INK_MUTED, linewidth=0.7, zorder=5,
            clip_on=False, solid_capstyle="butt",
        )
    ax.set_ylabel("EER (%), mean over 19 SSL models", fontsize=11, color=INK_PRIMARY)
    ax.set_xlabel(
        "MLAAD v10 TTS system (below axis), ordered by introduction date in YYYY-MM (above axis), oldest to newest",
        fontsize=11, color=INK_PRIMARY, labelpad=10,
    )

    rho = spearman(x, df["eer_mean"].to_numpy())
    ax.set_title(
        "Detection difficulty against TTS introduction date\n"
        f"91 systems, {df['pub_month'].iloc[0]} to {df['pub_month'].iloc[-1]}   "
        f"Spearman ρ = {rho:+.2f} between date order and mean EER",
        fontsize=13, color=INK_PRIMARY, pad=14, loc="left",
    )

    ax.grid(axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY)

    ax.legend(
        loc="upper left", fontsize=9.5, frameon=True, facecolor=SURFACE,
        edgecolor=BAND, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeline_csv", default=str(TIMELINE_CSV))
    ap.add_argument("--eer_csv", default=str(EER_CSV))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.timeline_csv), Path(args.eer_csv))
    plot_paper(df, out_dir / "mlaad_tts_eer_timeline.png")
    out_png = out_dir / "eer_by_tts_timeline.png"
    out_csv = out_dir / "eer_by_tts_timeline.csv"
    plot(df, out_png)

    cols = ["tts_system", "pub_year", "pub_month", "date_basis", "date_type",
            "eer_mean", "eer_q1", "eer_q3"]
    df[cols].to_csv(out_csv, index=False)

    rho = spearman(np.arange(len(df)), df["eer_mean"].to_numpy())
    print(f"wrote {out_dir / 'mlaad_tts_eer_timeline.png'} (paper, two panels)")
    print(f"wrote {out_png}")
    print(f"wrote {out_csv}")
    print(f"{len(df)} systems, {df['pub_month'].iloc[0]} to {df['pub_month'].iloc[-1]}")
    print(f"Spearman rho (date order vs mean EER) = {rho:+.3f}")




# ---------------------------------------------------------------------------
# Paper layout
# ---------------------------------------------------------------------------
# IEEE Access \textwidth is 177.53mm = 6.99in.  The wide screen figure above is
# 26in across; dropping it into a figure* scales it by 0.27, which renders the
# 7pt system names at 1.9pt.  At full text width 91 systems get 0.077in each
# while a legible vertical name needs ~0.10in, so one panel cannot hold them.
# Two panels of 46 give 0.152in each, which fits -- the same split the per-system
# heatmap already uses.  Fonts here are stated at their FINAL printed size.
TEXTWIDTH_IN = 6.99


def _panel(ax, df, lo, hi, ylim, coef, name_fs, date_fs, fig):
    """Draw one contiguous slice of systems onto ``ax``."""
    sub = df.iloc[lo:hi]
    xs = np.arange(lo, hi)

    ax.set_facecolor(SURFACE)
    ax.vlines(xs, sub["eer_q1"], sub["eer_q3"], color=BAND, linewidth=2.0, zorder=1)
    ax.axhline(CHANCE_EER, color=INK_MUTED, linestyle="--", linewidth=0.8, zorder=2)
    # One global trend fitted on all 91 systems, drawn across both panels: two
    # per-panel fits would invent a discontinuity that is not in the data.
    ax.plot(xs, np.polyval(coef, xs), color=INK_SECONDARY, linewidth=1.3,
            alpha=0.75, zorder=3)

    for dtype in DATE_TYPE_ORDER:
        s = sub[sub["date_type"] == dtype]
        if s.empty:
            continue
        color, marker, _ = DATE_TYPE_STYLE[dtype]
        ax.scatter(s.index, s["eer_mean"], s=9, marker=marker, color=color,
                   edgecolors=SURFACE, linewidths=0.5, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(sub["tts_system"], rotation=90, ha="center",
                       fontsize=name_fs, color=INK_SECONDARY)
    ax.set_xlim(lo - 1, hi)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", colors=INK_SECONDARY, labelsize=name_fs, length=2, pad=1.5)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.20, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)

    # Dates inside the axes, VERTICAL like the names.  Rotated, a "YYYY-MM"
    # label occupies only its line height horizontally (~1.05x the font size)
    # rather than its full text length, so it needs about one system slot
    # instead of three and the multi-row stagger collapses to a single row.
    # The greedy assignment is kept so the layout still adapts if the figure is
    # narrowed, but at text width it resolves to one row.
    axes_pt = fig.get_size_inches()[0] * 72 * 0.90
    label_w = date_fs * 1.05 / axes_pt * (hi - lo + 1)  # data units, rotated
    trans = ax.get_xaxis_transform()
    row_free: list[float] = []
    placed = []
    for month, idx in sub.groupby("pub_month", sort=False).indices.items():
        cx = float(np.mean(idx)) + lo
        r = next((i for i, e in enumerate(row_free) if cx - label_w / 2 >= e), len(row_free))
        if r == len(row_free):
            row_free.append(-np.inf)
        row_free[r] = cx + label_w / 2
        placed.append((cx, r, month))

    row_h = 0.115  # a rotated YYYY-MM is tall, so rows are spaced accordingly
    base = 0.014
    for cx, r, month in placed:
        y = base + r * row_h
        ax.text(cx, y, month, transform=trans, rotation=90, ha="center",
                va="bottom", fontsize=date_fs, color=INK_SECONDARY, zorder=5)
        ax.plot([cx, cx], [0.0, y - 0.003], transform=trans,
                color=INK_MUTED, linewidth=0.4, zorder=5, clip_on=False)


def plot_paper(df: pd.DataFrame, out_png: Path) -> None:
    """Single panel at IEEE Access full text width.

    The font size here is not a free choice.  Text width is 6.99in; after the y
    axis furniture roughly 6.54in remains for 91 systems, i.e. 0.072in = 5.2pt
    each.  A vertical label occupies its line height horizontally, about 1.05x
    the font size, so 4.5pt is the largest that fits without collision and 5pt
    already overflows.  Two panels of 46 would allow ~9pt but cost a full page,
    which is the trade this layout deliberately makes the other way.
    """
    name_fs, date_fs = 4.5, 4.0
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.9))
    fig.patch.set_facecolor(SURFACE)

    coef = np.polyfit(np.arange(len(df)), df["eer_mean"], 1)
    pad = 0.22 * (df["eer_q3"].max() - df["eer_q1"].min())
    ylim = (df["eer_q1"].min() - pad, df["eer_q3"].max() + 0.05 * pad)

    _panel(ax, df, 0, len(df), ylim, coef, name_fs, date_fs, fig)
    ax.set_ylabel("EER (%), mean over 19 SSL models", fontsize=6, color=INK_PRIMARY)
    ax.tick_params(axis="y", labelsize=5.5)

    handles = [
        plt.Line2D([], [], color=BAND, linewidth=2.0,
                   label=f"IQR across {int(df['n_models'].iloc[0])} SSL models"),
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.3,
                   label=f"trend, {coef[0]:+.2f} pp per system"),
    ] + [
        plt.Line2D([], [], color=DATE_TYPE_STYLE[d][0], marker=DATE_TYPE_STYLE[d][1],
                   linestyle="none", markersize=3.0,
                   label=f"{d} (n={int((df['date_type'] == d).sum())})")
        for d in DATE_TYPE_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=4.4, ncol=5,
              frameon=False, labelcolor=INK_SECONDARY, handletextpad=0.4,
              columnspacing=0.9, borderpad=0.15)

    fig.tight_layout(pad=0.35)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
