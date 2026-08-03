#!/usr/bin/env python3
"""EER heatmaps for MLAAD v10 synthesis diversity (SSL models x TTS grouping).

Every MLAAD system synthesizes from the same bonafide source (M-AILABS), so a
per-system EER measured against the *full pooled* M-AILABS bonafide set
(584,006 utterances) isolates system detectability from source-dataset
difficulty, and architectures become directly comparable.  Every group in
every figure is therefore scored against the same shared bonafide reference —
never a per-system or per-language subset.

Bonafide is the target class (higher score = more bonafide), matching
``compute_eer`` in ``evaluation.py``.

Four figures, rows = SSL models, each with its raw EER matrix as CSV:

  1. eer_by_tts_system      — 91 canonical TTS systems
  2. eer_by_architecture    — ``architecture_group`` (mutually exclusive, §4.2)
  3. eer_by_generation_mode — AR | NAR | Closed / Undisclosed
  4. eer_by_vocoder_family  — ``vocoder_family`` (not the verbatim vocoder, §3.3)

Usage
-----
    python -m spoof_superb.analysis.create_mlaad_tts_eer_heatmaps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spoof_superb.core.metrics import compute_eer  # noqa: E402

from spoof_superb.analysis import metadata_csv
from spoof_superb.config import cfg
from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import mlaad_pool_paths

SCORE_ROOT = Path(cfg.scores_root)
ARCH_NAME = "mlaad_v10_tts_architecture_groups.csv"
DIR_MAP = Path(__file__).resolve().parent / "mlaad_v10_dir_to_system.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "figures_mlaad_tts"

EXPECTED_RETAINED = 431_000
EXPECTED_BONAFIDE = 584_006
LOW_SUPPORT = 100

# Main results row order, extended with the two models the MLAAD v10 run adds.
MODELS = [
    ("FBANK", "fbank"),
    ("APC", "apc"),
    ("VQ-APC", "vq_apc"),
    ("NPC", "npc"),
    ("Mockingjay-960h", "mockingjay_960hr"),
    ("AudioALBERT-960h", "audio_albert_960hr"),
    ("TERA", "tera"),
    ("DeCoAR 2.0", "decoar2"),
    ("Modified CPC", "modified_cpc"),
    ("wav2vec", "wav2vec"),
    ("wav2vec 2.0 Base", "wav2vec2_base_960"),
    ("wav2vec 2.0 Large", "wav2vec2_large_ll60k"),
    ("HuBERT Base", "hubert_base"),
    ("HuBERT Large", "hubert_large_ll60k"),
    ("MR-HuBERT", "multires_hubert_multilingual_large600k"),
    ("XLS-R", "xls_r_300m"),
    ("UniSpeech-SAT", "unispeech_sat_large"),
    ("Data2Vec", "data2vec_large_ll60k"),
    ("WAVLABLM", "wavlablm_ek_40k"),
    ("WavLM Large", "wavlm_large"),
    ("SSAST", "ssast_frame_base"),
    ("MAE-AST-FRAME", "mae_ast_frame"),
]

#: Pretraining family of each row, in MODELS order. The heatmap draws a rule
#: wherever this changes.
#:
#: This used to be `SEPARATOR_ROWS = [1, 8, 20]` -- the boundaries as ROW
#: INDICES into the 22-model list above. That is correct only while every one of
#: those 22 models is present, and the v3 tree scores the 19 paper models, so
#: the rules would have been drawn across the middle of a family without any
#: error. Naming each row's family instead makes the boundaries follow whatever
#: subset is actually plotted.
FAMILY = {
    "FBANK": "baseline",
    "APC": "generative", "VQ-APC": "generative", "NPC": "generative",
    "Mockingjay-960h": "generative", "AudioALBERT-960h": "generative",
    "TERA": "generative", "DeCoAR 2.0": "generative",
    "Modified CPC": "discriminative", "wav2vec": "discriminative",
    "wav2vec 2.0 Base": "discriminative", "wav2vec 2.0 Large": "discriminative",
    "HuBERT Base": "discriminative", "HuBERT Large": "discriminative",
    "MR-HuBERT": "discriminative", "XLS-R": "discriminative",
    "UniSpeech-SAT": "discriminative", "Data2Vec": "discriminative",
    "WAVLABLM": "discriminative", "WavLM Large": "discriminative",
    "SSAST": "spectrogram", "MAE-AST-FRAME": "spectrogram",
}


def separator_rows(displays):
    """Row indices where the pretraining family changes, for the rows plotted."""
    return [i for i in range(1, len(displays))
            if FAMILY.get(displays[i]) != FAMILY.get(displays[i - 1])]

MODE_LABEL = {"AR": "AR", "NAR": "NAR", "unknown": "Closed / Undisclosed"}
MODE_ORDER = ["AR", "NAR", "Closed / Undisclosed"]


def auc_bonafide_vs_spoof(bonafide: np.ndarray, spoof: np.ndarray) -> float:
    """P(bonafide score > spoof score), ties counted as 0.5.

    Reported alongside any EER > 50% so that inverted score polarity (AUC well
    below 0.5) is distinguishable from a genuinely hard, near-chance group.
    """
    combined = np.concatenate([bonafide, spoof])
    ranks = pd.Series(combined).rank().to_numpy()
    n_b, n_s = len(bonafide), len(spoof)
    return float((ranks[:n_b].sum() - n_b * (n_b + 1) / 2) / (n_b * n_s))


def load_model_scores(scores_root, layout, slug: str,
                      dir_map: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (pooled bonafide scores, retained spoof rows with canonical_system).

    The MLAAD pool is one file under legacy and two under v2/v3; either way the
    spoof half is read from its tab-separated copy, which is mandatory because
    ~8.6% of utt_ids contain literal spaces.
    """
    paths = mlaad_pool_paths(slug, scores_root=scores_root, layout=layout)
    utt, lab, sc = read_scored(paths)
    df = pd.DataFrame({"utt_id": utt, "label": lab, "score": sc})
    if df["score"].isna().any():
        sys.exit(f"FATAL: NaN scores in {paths}")

    bonafide = df.loc[df["label"] == "bonafide", "score"].to_numpy(dtype=float)
    spoof = df[df["label"] == "spoof"].copy()
    spoof["raw_dir"] = spoof["utt_id"].str.split("/", expand=True)[3]

    merged = spoof.merge(dir_map, on="raw_dir", how="left", validate="many_to_one")
    if merged["canonical_system"].isna().any():
        sys.exit(f"FATAL: unmapped directories in {paths}")
    retained = merged[merged["canonical_system"] != "EXCLUDED"]

    assert len(bonafide) == EXPECTED_BONAFIDE, f"{slug}: {len(bonafide)} bonafide"
    assert len(retained) == EXPECTED_RETAINED, f"{slug}: {len(retained)} spoof"
    return bonafide, retained[["canonical_system", "score"]]


def eer_by_group(
    bonafide: np.ndarray, spoof: pd.DataFrame, key: pd.Series, label: str, model: str
) -> dict[str, float]:
    """EER (%) of every group in ``key`` against the full pooled bonafide set."""
    out: dict[str, float] = {}
    for group, rows in spoof.groupby(key, sort=True):
        scores = rows["score"].to_numpy(dtype=float)
        eer = compute_eer(bonafide, scores)[0] * 100.0
        if not np.isfinite(eer) or not (0.0 <= eer <= 100.0):
            sys.exit(f"FATAL: EER {eer} out of range for {label}/{group}/{model}")
        if eer > 50.0:
            auc = auc_bonafide_vs_spoof(bonafide, scores)
            print(
                f"    [EER>50] {label} '{group}' {model}: EER={eer:.2f}% "
                f"AUC(bonafide>spoof)={auc:.4f} n={len(scores)}"
            )
        out[group] = eer
    return out


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    figwidth: float,
    figheight: float,
    annot_size: float,
    xtick_size: float,
    xtick_rotation: float,
) -> None:
    """SSL models x groups heatmap with a Mean row, capped at 50% for display.

    EER > 50% means worse than chance; those cells saturate dark red.  The CSV
    written beside the figure keeps the raw, unclipped values.
    """
    mean_row = df.mean(axis=0).rename("Mean")
    df_with_mean = pd.concat([df, mean_row.to_frame().T])
    display_df = df_with_mean.clip(upper=50.0)

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    sns.set(style="white", font_scale=0.75)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        display_df,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0,
        vmax=50,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "EER (%, capped at 50)"},
        ax=ax,
        annot_kws={"size": annot_size},
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.tick_params(axis="x", labelsize=xtick_size, rotation=xtick_rotation)

    xlim = ax.get_xlim()
    for y in separator_rows(list(df.index)) + [len(df)]:
        ax.hlines(y, *xlim, colors="black", linewidth=1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores_root", default=None,
                        help="score tree to read (default: the configured one)")
    parser.add_argument("--layout", default=None, choices=("legacy", "v2", "v3"),
                        help="layout of that tree (default: the configured one)")
    parser.add_argument("--dir-map", type=Path, default=DIR_MAP)
    parser.add_argument("--arch-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scores_root = args.scores_root or cfg.scores_root
    layout = args.layout or getattr(cfg, "score_layout", "legacy")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dir_map = pd.read_csv(args.dir_map)[["raw_dir", "canonical_system"]]
    arch = pd.read_csv(args.arch_csv or metadata_csv(ARCH_NAME))
    arch_group = dict(zip(arch["tts_system"], arch["architecture_group"]))
    mode_group = dict(zip(arch["tts_system"], arch["ar_nar"].map(MODE_LABEL)))
    voc_group = dict(zip(arch["tts_system"], arch["vocoder_family"]))

    results: dict[str, dict[str, dict[str, float]]] = {
        "system": {},
        "architecture": {},
        "mode": {},
        "vocoder": {},
    }
    support: dict[str, int] = {}
    skipped: list[str] = []

    for display, slug in MODELS:
        paths = [Path(p) for p in mlaad_pool_paths(slug, scores_root=scores_root,
                                                   layout=layout)]
        absent = [p for p in paths if not p.is_file()]
        if absent:
            # Not fatal: a tree legitimately holds a subset of the roster (the
            # v3 tree scores the paper's 19 models, not all 22). Exiting here
            # made one absent model suppress the whole figure.
            skipped.append(f"{display} ({absent[0].name})")
            continue
        print(f"[{display}] {slug}")
        bonafide, spoof = load_model_scores(scores_root, layout, slug, dir_map)
        system = spoof["canonical_system"]

        if not support:
            support = system.value_counts().to_dict()

        results["system"][display] = eer_by_group(bonafide, spoof, system, "system", display)
        results["architecture"][display] = eer_by_group(
            bonafide, spoof, system.map(arch_group), "architecture", display
        )
        results["mode"][display] = eer_by_group(
            bonafide, spoof, system.map(mode_group), "mode", display
        )
        results["vocoder"][display] = eer_by_group(
            bonafide, spoof, system.map(voc_group), "vocoder_family", display
        )

    low = {s: n for s, n in support.items() if n < LOW_SUPPORT}
    print(f"\nSupport: {len(support)} systems, "
          f"min={min(support.values())} max={max(support.values())}")
    print(f"Low-confidence systems (<{LOW_SUPPORT} spoof utts): {low or 'none'}")

    figures = [
        (
            "system",
            "eer_by_tts_system",
            "MLAAD v10: EER (%) per SSL model x canonical TTS system "
            "(pooled M-AILABS bonafide)",
            sorted(set(arch["tts_system"]) - {"Dual-AR", "RVC", "Voxtral"}),
            dict(figwidth=30, figheight=8.5, annot_size=4.0, xtick_size=6, xtick_rotation=90),
        ),
        (
            "architecture",
            "eer_by_architecture",
            "MLAAD v10: EER (%) per SSL model x architecture group",
            None,
            dict(figwidth=14, figheight=8, annot_size=7, xtick_size=9, xtick_rotation=30),
        ),
        (
            "mode",
            "eer_by_generation_mode",
            "MLAAD v10: EER (%) per SSL model x generation mode",
            MODE_ORDER,
            dict(figwidth=7, figheight=8, annot_size=9, xtick_size=10, xtick_rotation=0),
        ),
        (
            "vocoder",
            "eer_by_vocoder_family",
            "MLAAD v10: EER (%) per SSL model x vocoder family",
            None,
            dict(figwidth=18, figheight=8, annot_size=6.5, xtick_size=8, xtick_rotation=60),
        ),
    ]

    summary = []
    for key, stem, title, expected_cols, style in figures:
        df = pd.DataFrame(results[key]).T
        df.index.name = "Model"
        # Order by the roster, but keep only models this tree actually scored:
        # reindexing to the full roster would re-add the skipped models as
        # all-NaN rows, giving blank bands in the figure and NaN rows in the CSV.
        df = df.reindex([d for d, _ in MODELS if d in df.index])

        if expected_cols is not None:
            defined = list(expected_cols)
        elif key == "architecture":
            defined = sorted(set(arch["architecture_group"]))
        else:
            defined = sorted(set(arch["vocoder_family"]))

        plotted = list(df.columns)
        empty = sorted(set(defined) - set(plotted))
        extra = sorted(set(plotted) - set(defined))
        df = df[[c for c in defined if c in plotted] + extra]

        csv_path = args.out_dir / f"{stem}.csv"
        df.to_csv(csv_path, float_format="%.4f")
        print(f"\n{stem}: plotted {len(plotted)} groups / {len(defined)} defined in protocol")
        if empty:
            print(f"  protocol groups with zero utterances: {empty}")
        if extra:
            print(f"  [WARN] groups not in the protocol definition: {extra}")
        print(f"  Saved: {csv_path}")

        plot_heatmap(df, title, args.out_dir / f"{stem}.png", **style)

        # A cell exceeds 50% mainly for the weakest SSL front-ends, so count
        # cells and mean-EER groups separately rather than flagging any column
        # that a single weak model pushes over chance.
        over50_cells = int((df > 50.0).to_numpy().sum())
        over50_any = sorted(c for c in df.columns if (df[c] > 50.0).any())
        over50_mean = sorted(c for c in df.columns if df[c].mean() > 50.0)
        summary.append(
            {
                "figure": stem,
                "n_ssl_models": len(df),
                "groups_plotted": len(plotted),
                "groups_in_protocol": len(defined),
                "cells_eer_gt_50": over50_cells,
                "groups_any_model_gt_50": len(over50_any),
                "groups_mean_gt_50": ", ".join(over50_mean) or "none",
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir / "figure_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))
    if skipped:
        print(f"\nNot scored on this tree, omitted from the figures "
              f"({len(skipped)}): {', '.join(skipped)}")
    print(
        f"\n✅ STEP 4 — {args.out_dir} — 4 heatmaps + 4 EER CSVs, "
        f"{len(MODELS) - len(skipped)} SSL models, {len(support)} systems, "
        f"{EXPECTED_RETAINED} spoof vs {EXPECTED_BONAFIDE} pooled bonafide utts"
    )


if __name__ == "__main__":
    main()
