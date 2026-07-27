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
    python3 scripts/layer_weight_analysis.py
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

CKPT_ROOT = Path("/data/ssl_anti_spoofing/asd_superb_models/linear_head_models")
PREFIX = "model_weighted_CCE_50_64_linear_head_ASV19_"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "layer_weights"

# Display name + branch/family grouping for the heatmap row order.
# Branch labels follow the Table 7 lineage: contrastive (wav2vec 2.0 -> XLS-R)
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

GRID = 25  # relative-depth bins for the common heatmap axis
MIN_LAYERS_FOR_HEATMAP = 10  # models shallower than this go to the CSV only


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


def to_grid(w: np.ndarray) -> np.ndarray:
    """Interpolate a length-L weight vector onto GRID points in relative depth,
    then min-max normalise within the model so the weak tilt is visible."""
    L = len(w)
    xs = np.linspace(0, 1, L)
    g = np.interp(np.linspace(0, 1, GRID), xs, w)
    rng = g.max() - g.min()
    return (g - g.min()) / rng if rng > 0 else np.full(GRID, 0.5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, heat_rows, heat_labels, heat_groups = {}, [], [], []
    missing = []

    for slug, disp, group in MODELS:
        w = load_weights(slug)
        if w is None:
            missing.append(disp)
            continue
        s = summarize(w)
        s["group"] = group
        summary[disp] = s
        if s["n_layers"] >= MIN_LAYERS_FOR_HEATMAP:
            heat_rows.append(to_grid(w))
            heat_labels.append(f"{disp}  (L={s['n_layers']}, sp={s['spread']:.3f})")
            heat_groups.append(group)

    (OUT_DIR / "layer_weights.json").write_text(json.dumps(summary, indent=2))
    print(f"Extracted weights for {len(summary)} models; no weights for: {missing}")
    print(f"\n{'model':20s} {'L':>3s} {'CoG(rel)':>8s} {'peak(rel)':>9s} {'spread':>7s}  group")
    for disp, s in summary.items():
        print(f"{disp:20s} {s['n_layers']:3d} {s['cog_rel']:8.3f} {s['peak_rel']:9.3f} "
              f"{s['spread']:7.3f}  {s['group']}")

    # --- heatmap: deep models, relative depth, within-model normalised --------
    order = ["Contrastive", "Predictive", "Spectrogram", "Generative"]
    zipped = sorted(zip(heat_labels, heat_rows, heat_groups),
                    key=lambda t: (order.index(t[2]), t[0]))
    labels = [z[0] for z in zipped]
    mat = np.array([z[1] for z in zipped])
    groups = [z[2] for z in zipped]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(labels) + 1.2))
    sns.heatmap(
        mat, cmap="viridis", ax=ax, yticklabels=labels,
        xticklabels=[f"{i/(GRID-1):.1f}" for i in range(GRID)],
        cbar_kws={"label": "within-model normalised layer weight"},
    )
    # centre-of-gravity marker per row
    for r, lab in enumerate(labels):
        disp = lab.split("  (")[0]
        ax.plot((summary[disp]["cog_rel"]) * (GRID - 1) + 0.5, r + 0.5,
                "o", color="white", markersize=4, markeredgecolor="black")
    # group separators
    for r in range(1, len(groups)):
        if groups[r] != groups[r - 1]:
            ax.axhline(r, color="black", linewidth=1.5)
    ax.set_xlabel("relative layer depth (0 = first, 1 = last); white dot = centre of gravity")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=0)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer_weight_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'layer_weight_profiles.png'}")
    print(f"Saved: {OUT_DIR / 'layer_weights.json'}")


if __name__ == "__main__":
    main()
