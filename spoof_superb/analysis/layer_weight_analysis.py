"""
Analysis A -- learned layer-weight profiles of the frozen SSL front-ends.

Every linear_head model uses the s3prl Featurizer (SUPERB weighted-sum): a
trainable per-layer weight vector, softmax-normalised, that forms a weighted sum
over all upstream hidden states. Those weights are learned during downstream
training and stored in each checkpoint as `ssl_model.featurizer.weights`. This
script reads them -- no GPU, no retraining -- and reports, per model:

  - the softmax-normalised weight of every layer,
  - the centre of gravity (weight-averaged layer index), in relative depth 0..1,
  - the peak layer, and
  - the spread (max - min), which quantifies how far the profile departs from
    uniform. For the large transformer models the spread is tiny (~0.005-0.02),
    i.e. the trained detector weights all layers almost equally; the profiles are
    therefore shown WITHIN-MODEL normalised to expose the weak tilt, and the raw
    spread is reported alongside so the reader knows how weak it is.

Models have very different depths (2 to 27 layers), so the heatmap places every
model on a common relative-depth axis (0 = first layer, 1 = last) by linear
interpolation. FBANK (not SSL) and BYOL-A (no weighted-sum featurizer) have no
layer weights and are skipped.

Usage
-----
    python -m spoof_superb.analysis.layer_weight_analysis
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from spoof_superb.config import cfg

CKPT_ROOT = Path(cfg.models_root)
PREFIX = "model_weighted_CCE_50_64_linear_head_ASV19_"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "layer_weights"

# Display name + branch/family grouping for the heatmap row order.
# Branch labels follow the tab:top_ssl_lineage branches: contrastive (wav2vec 2.0 -> XLS-R)
# vs predictive (HuBERT and descendants). Generative and spectrogram models are
# grouped separately.
MODELS = [
    # slug, display, group
    ("apc", "APC", "Generative"),
    ("vq_apc", "VQ-APC", "Generative"),
    ("npc", "NPC", "Generative"),
    ("mockingjay_960hr", "Mockingjay-960h", "Generative"),
    ("audio_albert_960hr", "AudioALBERT-960h", "Generative"),
    ("tera", "TERA", "Generative"),
    ("decoar2", "DeCoAR 2.0", "Generative"),
    ("modified_cpc", "Modified CPC", "Contrastive"),
    ("wav2vec", "wav2vec", "Contrastive"),
    ("wav2vec2_base_960", "wav2vec 2.0 Base", "Contrastive"),
    ("wav2vec2_large_ll60k", "wav2vec 2.0 Large", "Contrastive"),
    ("xls_r_300m", "XLS-R", "Contrastive"),
    ("hubert_base", "HuBERT Base", "Predictive"),
    ("hubert_large_ll60k", "HuBERT Large", "Predictive"),
    ("multires_hubert_multilingual_large600k", "MR-HuBERT", "Predictive"),
    ("unispeech_sat_large", "UniSpeech-SAT", "Predictive"),
    ("wavlm_large", "WavLM Large", "Predictive"),
    ("wavlablm_ek_40k", "WAVLABLM", "Predictive"),
    ("data2vec_large_ll60k", "Data2Vec", "Predictive"),
    ("ssast_frame_base", "SSAST", "Spectrogram"),
    ("mae_ast_frame", "MAE-AST-FRAME", "Spectrogram"),
]

def load_weights(slug: str) -> np.ndarray | None:
    ck = CKPT_ROOT / f"{PREFIX}{slug}" / "swa.pth"
    if not ck.exists():
        return None
    sd = torch.load(ck, map_location="cpu")
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    key = next((k for k in sd if "featurizer.weights" in k), None)
    if key is None:
        return None
    return F.softmax(sd[key].float(), dim=-1).numpy()


def summarize(w: np.ndarray) -> dict:
    L = len(w)
    idx = np.arange(L)
    cog = float((idx * w).sum())              # absolute centre of gravity
    return {
        "n_layers": L,
        "cog_abs": round(cog, 3),
        "cog_rel": round(cog / (L - 1), 4) if L > 1 else 0.0,
        "peak_layer": int(w.argmax()),
        "peak_rel": round(float(w.argmax()) / (L - 1), 4) if L > 1 else 0.0,
        "spread": round(float(w.max() - w.min()), 4),
        "weights": [round(float(x), 5) for x in w],
    }


# Depth panels: models are grouped by layer count so every panel can use a
# genuine integer layer-number x-axis. Mixing e.g. a 13-layer and a 25-layer
# model on one integer axis would misplace the shorter model's last layer in the
# middle of the axis, which is exactly what the relative-depth axis was hiding.
PANELS = [("Deep models (24--27 layers)", 20, 99),
          ("Base / spectrogram models (12--13 layers)", 12, 19)]
ROW_ORDER = ["Contrastive", "Predictive", "Spectrogram", "Generative"]


def norm_within(w: np.ndarray) -> np.ndarray:
    rng = w.max() - w.min()
    return (w - w.min()) / rng if rng > 0 else np.full(len(w), 0.5)


def draw_panel(ax, models, summary, weights, title):
    """Heatmap: rows = models, x = actual layer index, colour = within-model
    normalised weight, white dot = peak layer. Shorter models are NaN-padded."""
    models = sorted(models, key=lambda d: (ROW_ORDER.index(summary[d]["group"]), d))
    maxL = max(summary[d]["n_layers"] for d in models)
    mat = np.full((len(models), maxL), np.nan)
    labels, groups = [], []
    for r, d in enumerate(models):
        w = norm_within(weights[d])
        mat[r, :len(w)] = w
        labels.append(f"{d}  (L={summary[d]['n_layers']}, sp={summary[d]['spread']:.3f})")
        groups.append(summary[d]["group"])
    sns.heatmap(
        mat, cmap="viridis", ax=ax, yticklabels=labels,
        xticklabels=list(range(maxL)),
        cbar_kws={"label": "within-model normalised weight"},
    )
    for r, d in enumerate(models):
        ax.plot(summary[d]["peak_layer"] + 0.5, r + 0.5, "o",
                color="white", markersize=5, markeredgecolor="black")
    for r in range(1, len(groups)):
        if groups[r] != groups[r - 1]:
            ax.axhline(r, color="black", linewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("SSL layer index (0 = CNN output; white dot = peak-weight layer)")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=8, rotation=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, weights, missing = {}, {}, []

    for slug, disp, group in MODELS:
        w = load_weights(slug)
        if w is None:
            missing.append(disp)
            continue
        s = summarize(w)
        s["group"] = group
        summary[disp] = s
        weights[disp] = w

    (OUT_DIR / "layer_weights.json").write_text(json.dumps(summary, indent=2))
    print(f"Extracted weights for {len(summary)} models; no weights for: {missing}")
    print(f"\n{'model':20s} {'L':>3s} {'peak':>4s} {'spread':>7s}  group")
    for disp, s in summary.items():
        print(f"{disp:20s} {s['n_layers']:3d} {s['peak_layer']:4d} "
              f"{s['spread']:7.3f}  {s['group']}")

    panel_models = [[d for d in summary if lo <= summary[d]["n_layers"] <= hi]
                    for _, lo, hi in PANELS]
    used = {d for pm in panel_models for d in pm}
    print(f"\nShallow models in CSV only (<12 layers): "
          f"{[d for d in summary if d not in used]}")

    heights = [0.45 * len(pm) + 1.0 for pm in panel_models]
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, sum(heights)),
                             gridspec_kw={"height_ratios": heights})
    for ax, (title, _, _), pm in zip(axes, PANELS, panel_models):
        draw_panel(ax, pm, summary, weights, title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer_weight_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'layer_weight_profiles.png'}")
    print(f"Saved: {OUT_DIR / 'layer_weights.json'}")


if __name__ == "__main__":
    main()
