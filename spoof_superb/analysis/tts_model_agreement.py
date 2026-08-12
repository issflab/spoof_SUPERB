"""Do SSL models agree on which TTS systems are hard, and does the vocoder matter?

    python -m spoof_superb.analysis.tts_model_agreement

Two questions the per-system heatmap cannot answer, because it is read by its
Mean column and the Mean averages the models away.

1. Agreement. If detection difficulty were a property of the synthesis system,
   every model would order the 91 systems the same way. Reported three ways:
   pairwise Spearman rho; a two-way decomposition of the 19 x 91 EER matrix
   into system, model and interaction shares; and the overlap between the ten
   systems each model finds hardest, which is the same fact in the units a
   reader can act on.

   The decomposition has one observation per (system, model) cell, so the
   interaction share is a residual and cannot be separated from measurement
   noise on its own. It is reported alongside the correlations, which are
   computed independently and point the same way.

2. Vocoder. The taxonomy records a vocoder family per system in the
   expectation that waveform generation leaves its own artifacts. There are 27
   families for 91 systems and 17 of them hold two systems or fewer, so family
   means cannot be ranked against each other -- the same objection that limits
   the architecture figure to its larger groups. The question is therefore
   asked at the level of the distinction the taxonomy was recorded for,
   neural codec tokenizer against waveform vocoder, and asked again inside the
   LLM-backbone systems, where codec vocoders are concentrated, to see how much
   of the difference is the vocoder and how much is the backbone that feeds it.

Reads `{outputs_root}/tts/`: `eer_by_tts_system.csv` (all 19 models x 91
systems, unclipped), `eer_by_tts_system_ranked.csv` for the `Mean (all 19)`
column, `eer_by_vocoder_family_ranked.csv`, and the vocoder/architecture
assignment in `analysis/mlaad_v10_tts_architecture_groups.csv`. Recomputes no
EER.

Writes `{outputs_root}/tts/model_agreement.{csv,md}`.
"""

import argparse
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spoof_superb import REPO_ROOT
from spoof_superb.config import cfg

HERE = Path(__file__).resolve().parent
GROUPS_CSV = HERE / "mlaad_v10_tts_architecture_groups.csv"

MEAN_COL = "Mean (all 19)"

#: Models operating on time-frequency patches rather than the waveform. The
#: agreement question is about representation families, so this split is the
#: one that matters.
SPECTROGRAM_MODELS = ["SSAST", "MAE-AST-FRAME"]

#: Vocoder families that emit discrete acoustic tokens decoded by a neural
#: codec, as opposed to a vocoder that regresses a waveform from a spectrogram.
CODEC_FAMILIES = {
    "EnCodec", "SNAC", "Mimi", "DAC", "WavTokenizer", "BiCodec", "NeuCodec",
    "X-Codec2", "NanoCodec", "Code2Wav", "MOSS-Audio-Tokenizer",
    "Higgs-Audio tokenizer", "VibeVoice tokenizer",
}

LLM_GROUPS = {"LLM", "Flow + LLM", "Diffusion + LLM"}

#: Minimum members for a group mean to be quoted, matching the threshold the
#: paper applies to the architecture groups. Applying a looser bar to vocoders
#: than to architectures would be the very inconsistency this module exists to
#: avoid.
MIN_GROUP = 7

#: How many of each model's hardest systems to intersect.
TOP_K = 10


def _count(col):
    return int(re.search(r"\((\d+)\)", col).group(1))


def agreement(matrix):
    """matrix: models x systems. Returns (pairwise frame, variance shares)."""
    models = list(matrix.index)
    rows = [{"model_a": a, "model_b": b,
             "spearman_rho": spearmanr(matrix.loc[a], matrix.loc[b]).statistic}
            for a, b in itertools.combinations(models, 2)]
    pair = pd.DataFrame(rows).sort_values("spearman_rho")

    x = matrix.values.T                       # systems x models
    grand = x.mean()
    sys_eff = x.mean(axis=1) - grand
    mod_eff = x.mean(axis=0) - grand
    resid = x - grand - sys_eff[:, None] - mod_eff[None, :]
    total = ((x - grand) ** 2).sum()
    shares = {
        "system": 100.0 * x.shape[1] * (sys_eff ** 2).sum() / total,
        "model": 100.0 * x.shape[0] * (mod_eff ** 2).sum() / total,
        "interaction": 100.0 * (resid ** 2).sum() / total,
    }
    return pair, shares


def top_k_overlap(matrix, k=TOP_K):
    hardest = {m: set(matrix.loc[m].nlargest(k).index) for m in matrix.index}
    return pd.DataFrame(
        [{"model_a": a, "model_b": b, "shared": len(hardest[a] & hardest[b])}
         for a, b in itertools.combinations(matrix.index, 2)])


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.tts_model_agreement",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)

    root = getattr(cfg, "outputs_root", "") or str(REPO_ROOT / "outputs")
    in_dir = Path(args.in_dir or os.path.join(root, "tts"))
    out_dir = Path(args.out_dir or in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(in_dir / "eer_by_tts_system.csv", index_col="Model")
    ranked = pd.read_csv(in_dir / "eer_by_tts_system_ranked.csv")
    voc = pd.read_csv(in_dir / "eer_by_vocoder_family_ranked.csv", index_col="Model")
    groups = pd.read_csv(GROUPS_CSV)

    pair, shares = agreement(full)
    overlap = top_k_overlap(full)

    spec = [m for m in SPECTROGRAM_MODELS if m in full.index]
    wave = [m for m in full.index if m not in spec]

    def subset(frame, pred):
        return frame[frame.apply(
            lambda r: pred(r.model_a, r.model_b), axis=1)]

    cross = subset(pair, lambda a, b: (a in spec) != (b in spec))
    within = subset(pair, lambda a, b: a not in spec and b not in spec)

    pair.to_csv(out_dir / "model_agreement.csv", index=False)

    # --- vocoder -----------------------------------------------------------
    big_fams = [c for c in voc.columns
                if _count(c) >= MIN_GROUP and not c.startswith("unknown")]
    fam_tbl = pd.DataFrame(
        [{"family": c.rsplit(" (", 1)[0], "n": _count(c),
          "eer": voc.loc[MEAN_COL, c]} for c in big_fams]).sort_values("eer")

    merged = ranked.merge(groups, on="tts_system", how="left")
    if merged.vocoder_family.isna().any():
        raise SystemExit("systems missing a vocoder assignment")
    merged["voc_class"] = merged.vocoder_family.map(
        lambda v: "unknown" if v == "unknown"
        else ("neural codec" if v in CODEC_FAMILIES else "waveform vocoder"))

    def by_class(frame):
        return (frame[frame.voc_class != "unknown"]
                .groupby("voc_class")[MEAN_COL].agg(["mean", "count"]))

    overall = by_class(merged)
    in_llm = by_class(merged[merged.architecture_group.isin(LLM_GROUPS)])

    def gap(tbl):
        return tbl.loc["neural codec", "mean"] - tbl.loc["waveform vocoder", "mean"]

    # --- report ------------------------------------------------------------
    n_fams = len(voc.columns)
    tiny = sum(1 for c in voc.columns if _count(c) <= 2)
    xlsr = overlap[(overlap.model_a == "XLS-R") | (overlap.model_b == "XLS-R")]

    L = [
        "# TTS synthesis diversity: model agreement and vocoder effect",
        "",
        f"{full.shape[1]} MLAAD v10 systems, {full.shape[0]} SSL models. "
        "Every statistic below uses all of them.",
        "",
        "## 1. Do the models agree on which systems are hard?",
        "",
        f"Pairwise Spearman rho over the {full.shape[1]} systems: "
        f"mean {pair.spearman_rho.mean():.3f}, median "
        f"{pair.spearman_rho.median():.3f}, "
        f"range {pair.spearman_rho.min():.3f} to {pair.spearman_rho.max():.3f}.",
        "",
        f"  spectrogram-patch vs waveform models  mean rho "
        f"{cross.spearman_rho.mean():.3f} "
        f"({cross.spearman_rho.min():.2f} to {cross.spearman_rho.max():.2f})",
        f"  waveform vs waveform models           mean rho "
        f"{within.spearman_rho.mean():.3f}",
        "",
        "Least agreement:",
        pair.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "Variance shares (EER ~ grand + system + model + interaction):",
        "",
    ]
    L += [f"  {k:12s} {v:5.1f}%" for k, v in shares.items()]
    L += [
        "",
        f"## 2. Overlap of the {TOP_K} hardest systems, XLS-R against each model",
        "",
        xlsr.assign(other=lambda f: np.where(f.model_a == "XLS-R",
                                             f.model_b, f.model_a))
            [["other", "shared"]]
            .sort_values("shared", ascending=False)
            .to_string(index=False),
        "",
        "## 3. Vocoder",
        "",
        f"{n_fams} vocoder families for {full.shape[1]} systems; "
        f"{tiny} of them hold two systems or fewer. "
        f"Families reaching n >= {MIN_GROUP} (excluding `unknown`):",
        "",
        fam_tbl.to_string(index=False, float_format=lambda v: f"{v:.1f}"),
        "",
        "By vocoder class (mean of the per-system Mean (all 19) EER):",
        "",
        overall.to_string(float_format=lambda v: f"{v:.2f}"),
        f"  gap = {gap(overall):.1f} points",
        "",
        "Restricted to systems with an LLM backbone, where codec vocoders "
        "are concentrated:",
        "",
        in_llm.to_string(float_format=lambda v: f"{v:.2f}"),
        f"  gap = {gap(in_llm):.1f} points",
        "",
    ]
    (out_dir / "model_agreement.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"  Saved: {out_dir / 'model_agreement.csv'}")
    print(f"  Saved: {out_dir / 'model_agreement.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
