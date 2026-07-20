"""
Compute FAR (False Acceptance Rate) for all SSL models across all TTS systems.

Each model's decision threshold is the EER threshold derived from its pooled
combined score file (normalized_scores_by_ssl_model/combined_<stem>.txt).
Z-score normalization applied upstream ensures cross-dataset score ranges are
equalized, so a single pooled threshold is calibration-valid.

Outputs five CSVs:
  1. far_matrix_ar.csv       — 19 SSL models × 30 AR TTS systems
  2. far_matrix_nar.csv      — 19 SSL models × 26 NAR TTS systems
  3. far_pooled.csv          — mean FAR per SSL model (AR mean | NAR mean | Overall mean)
  4. far_by_architecture.csv — mean FAR per SSL model per arch tag
                               (No NN | RNN | Transformer | Diffusion | Flow | LLM | Masked)
  5. far_by_vocoder.csv      — mean FAR per SSL model per vocoder family

FAR is computed on spoof-labelled rows only.
Systems with multiple arch tags / vocoders count in each relevant group.

Usage
-----
    python3 scripts/compute_far_matrix.py \\
        --tts_dir      /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_TTS_norm \\
        --combined_dir /data/ssl_anti_spoofing/asd_superb_score_files/normalized_scores_by_ssl_model \\
        --out_dir      /data/ssl_anti_spoofing/asd_superb_score_files/far_results_zscore
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import compute_eer  # type: ignore

# ---------------------------------------------------------------------------
# SSL model mapping: file stem → display name  (19 models, benchmark order)
# ---------------------------------------------------------------------------
SSL_STEMS: Dict[str, str] = {
    "fbank":                                  "FBANK",
    "apc":                                    "APC",
    "npc":                                    "NPC",
    "mockingjay":                             "Mockingjay",
    "tera":                                   "TERA",
    "decoar2":                                "DeCoAR 2.0",
    "wav2vec":                                "wav2vec",
    "wav2vec2_base_960":                      "wav2vec 2.0 Base",
    "wav2vec2_large_ll60k":                   "wav2vec 2.0 Large",
    "hubert_base":                            "HuBERT Base",
    "hubert_large_ll60k":                     "HuBERT Large",
    "multires_hubert_multilingual_large600k": "MR-HuBERT",
    "xls_r_300m":                             "XLS-R",
    "unispeech_sat_large":                    "UniSpeech-SAT",
    "data2vec_large_ll60k":                   "Data2Vec",
    "wavlablm_ek_40k":                        "WAVLABLM",
    "wavlm_large":                            "WavLM Large",
    "ssast_frame_base":                       "SSAST",
    "mae_ast_frame":                          "MAE-AST-FRAME",
}

ORDERED_MODELS: List[str] = [
    "FBANK",
    "APC", "NPC", "Mockingjay", "TERA", "DeCoAR 2.0",
    "wav2vec", "wav2vec 2.0 Base", "wav2vec 2.0 Large",
    "HuBERT Base", "HuBERT Large", "MR-HuBERT", "XLS-R",
    "UniSpeech-SAT", "Data2Vec", "WAVLABLM", "WavLM Large",
    "SSAST", "MAE-AST-FRAME",
]

# ---------------------------------------------------------------------------
# TTS display names (directory stem → short label for heatmap columns)
# ---------------------------------------------------------------------------
AR_DISPLAY: Dict[str, str] = {
    "ASR_TTS_AR":                       "ASR-TTS",
    "Bark":                             "Bark",
    "Capacitron":                       "Capacitron",
    "CosyVoice2":                       "CosyVoice2",
    "FishSpeech":                       "FishSpeech",
    "FishTTS":                          "FishTTS",
    "Indri-TTS-0.1":                    "Indri",
    "In-house_ASR-based":               "In-house ASR",
    "LLASA":                            "LLASA",
    "MQTTS":                            "MQTTS",
    "Multi-scale_Transformer":          "MST",
    "Multi-scale_Transformer_pre-trained": "MST-pt",
    "NeuralHMM":                        "NeuralHMM",
    "Neural_TTS":                       "Neural TTS",
    "NN-SPSS_TTS":                      "NN-SPSS",
    "NN-TTS":                           "NN-TTS",
    "Openaudio-S1-Mini":                "Openaudio",
    "OpenVoiceV2":                      "OpenVoice2",
    "Overflow":                         "Overflow",
    "Parler_TTS":                       "Parler",
    "Sesame_CSM_1B":                    "Sesame CSM",
    "Spark_TTS":                        "Spark TTS",
    "SSRSpeech":                        "SSRSpeech",
    "Tacotron_2":                       "Tacotron 2",
    "Tortoise":                         "Tortoise",
    "TransformerTTS":                   "TransformerTTS",
    "VALL-E":                           "VALL-E",
    "Veena":                            "Veena",
    "VITS":                             "VITS",
    "WhisperSpeech":                    "WhisperSpeech",
}

NAR_DISPLAY: Dict[str, str] = {
    "ASR_TTS_NAR":          "ASR-TTS",
    "BVAE-TTS":             "BVAE-TTS",
    "E2TTS":                "E2TTS",
    "F5TTS":                "F5TTS",
    "FastPitch":            "FastPitch",
    "GlowTTS":              "GlowTTS",
    "GradTTS":              "GradTTS",
    "Kokoro":               "Kokoro",
    "MaryTTS":              "MaryTTS",
    "MaskGCT":              "MaskGCT",
    "MatchaTTS":            "MatchaTTS",
    "MetaVoice-1B":         "MetaVoice",
    "NN-VC":                "NN-VC",
    "Non-parallel_VC":      "Non-par VC",
    "SpeechT5":             "SpeechT5",
    "SpeedySpeech":         "SpeedySpeech",
    "StyleTTS2":            "StyleTTS2",
    "ToucanTTS":            "ToucanTTS",
    "Transfer-function_VC": "TF-VC",
    "VixTTS":               "VixTTS",
    "VoiceTextWebAPI":      "VoiceText",
    "XTTS":                 "XTTS",
    "YourTTS":              "YourTTS",
    "Zero-Shot_VC":         "ZS-VC",
    "ZipVoice":             "ZipVoice",
    "ZMM-TTS":              "ZMM-TTS",
}

# ---------------------------------------------------------------------------
# Architecture tags
# ---------------------------------------------------------------------------
ARCH_TAGS: Dict[str, List[str]] = {
    "ASR_TTS_NAR":          ["No NN"],
    "BVAE-TTS":             ["RNN"],
    "E2TTS":                ["Transformer", "Flow", "Masked"],
    "F5TTS":                ["Diffusion", "Flow", "Masked"],
    "FastPitch":            ["Transformer"],
    "GlowTTS":              ["Transformer", "Flow"],
    "GradTTS":              ["Diffusion"],
    "Kokoro":               ["Diffusion"],
    "MaryTTS":              ["No NN"],
    "MaskGCT":              ["Transformer", "LLM", "Masked"],
    "MatchaTTS":            ["Transformer", "Flow"],
    "MetaVoice-1B":         ["Transformer", "LLM"],
    "NN-VC":                ["RNN"],
    "Non-parallel_VC":      ["No NN"],
    "SpeechT5":             ["Transformer"],
    "SpeedySpeech":         ["RNN"],
    "StyleTTS2":            ["Diffusion"],
    "ToucanTTS":            ["Transformer", "Masked"],
    "Transfer-function_VC": ["No NN"],
    "VixTTS":               ["LLM"],
    "VoiceTextWebAPI":      ["No NN"],
    "XTTS":                 ["LLM"],
    "YourTTS":              ["Transformer"],
    "Zero-Shot_VC":         ["Transformer"],
    "ZipVoice":             ["Transformer", "Flow", "Masked"],
    "ZMM-TTS":              ["Transformer", "LLM"],
    "ASR_TTS_AR":                       ["RNN"],
    "Bark":                             ["Transformer", "LLM"],
    "Capacitron":                       ["RNN"],
    "CosyVoice2":                       ["Flow", "LLM"],
    "FishSpeech":                       ["Transformer", "LLM"],
    "FishTTS":                          ["Transformer", "LLM"],
    "Indri-TTS-0.1":                    ["Transformer", "LLM"],
    "In-house_ASR-based":               ["No NN"],
    "LLASA":                            ["Transformer", "LLM"],
    "MQTTS":                            ["Transformer"],
    "Multi-scale_Transformer":          ["Transformer", "LLM"],
    "Multi-scale_Transformer_pre-trained": ["Transformer", "LLM"],
    "NeuralHMM":                        ["RNN"],
    "Neural_TTS":                       ["RNN"],
    "NN-SPSS_TTS":                      ["RNN"],
    "NN-TTS":                           ["RNN"],
    "Openaudio-S1-Mini":                ["Transformer", "LLM"],
    "OpenVoiceV2":                      ["RNN"],
    "Overflow":                         ["RNN"],
    "Parler_TTS":                       ["Transformer", "LLM"],
    "Sesame_CSM_1B":                    ["Transformer", "LLM"],
    "Spark_TTS":                        ["Transformer", "LLM"],
    "SSRSpeech":                        ["Transformer", "Masked"],
    "Tacotron_2":                       ["RNN"],
    "Tortoise":                         ["Transformer", "Diffusion"],
    "TransformerTTS":                   ["Transformer"],
    "VALL-E":                           ["LLM"],
    "Veena":                            ["Transformer", "LLM"],
    "VITS":                             ["RNN"],
    "WhisperSpeech":                    ["Transformer", "LLM"],
}

# ---------------------------------------------------------------------------
# Vocoder tags
# ---------------------------------------------------------------------------
VOCODER_TAGS: Dict[str, List[str]] = {
    "ASR_TTS_NAR":          ["STRAIGHT"],
    "BVAE-TTS":             ["WaveGlow"],
    "E2TTS":                ["BigVGAN"],
    "F5TTS":                ["Vocos"],
    "FastPitch":            ["WaveGlow"],
    "GlowTTS":              ["WaveGlow"],
    "GradTTS":              ["HiFi-GAN"],
    "Kokoro":               ["iSTFTNet"],
    "MaryTTS":              ["Built-in"],
    "MaskGCT":              ["Vocos"],
    "MatchaTTS":            ["HiFi-GAN"],
    "MetaVoice-1B":         ["Multi-band Diffusion"],
    "NN-VC":                ["Waveform Filtering"],
    "Non-parallel_VC":      ["PLDA"],
    "SpeechT5":             ["HiFi-GAN"],
    "SpeedySpeech":         ["MelGAN"],
    "StyleTTS2":            ["HiFi-GAN", "iSTFTNet"],
    "ToucanTTS":            ["BigVGAN"],
    "Transfer-function_VC": ["Spectral Filtering"],
    "VixTTS":               ["HiFi-GAN", "iSTFTNet"],
    "VoiceTextWebAPI":      ["World"],
    "XTTS":                 ["HiFi-GAN", "iSTFTNet"],
    "YourTTS":              ["HiFi-GAN"],
    "Zero-Shot_VC":         ["HiFi-GAN"],
    "ZipVoice":             ["Vocos"],
    "ZMM-TTS":              ["HiFi-GAN"],
    "ASR_TTS_AR":                       ["WaveNet"],
    "Bark":                             ["SoundStream"],
    "Capacitron":                       ["WaveNet"],
    "CosyVoice2":                       ["HiFi-GAN"],
    "FishSpeech":                       ["Firefly-GAN", "FF-GAN"],
    "FishTTS":                          ["Firefly-GAN"],
    "Indri-TTS-0.1":                    ["Mimi"],
    "In-house_ASR-based":               ["BigVGAN"],
    "LLASA":                            ["Vocos"],
    "MQTTS":                            ["HiFi-GAN"],
    "Multi-scale_Transformer":          ["Encodec"],
    "Multi-scale_Transformer_pre-trained": ["Encodec"],
    "NeuralHMM":                        ["WaveGlow"],
    "Neural_TTS":                       ["WaveNet"],
    "NN-SPSS_TTS":                      ["Vocaine"],
    "NN-TTS":                           ["World", "Neural-Source-Filter"],
    "Openaudio-S1-Mini":                ["FF-GAN"],
    "OpenVoiceV2":                      ["HiFi-GAN"],
    "Overflow":                         ["HiFi-GAN"],
    "Parler_TTS":                       ["RVQ"],
    "Sesame_CSM_1B":                    ["Mimi"],
    "Spark_TTS":                        ["BiCodec"],
    "SSRSpeech":                        ["Transformer"],
    "Tacotron_2":                       ["WaveNet", "WaveRNN", "Griffin-Lim"],
    "Tortoise":                         ["UnivNet"],
    "TransformerTTS":                   ["WaveNet"],
    "VALL-E":                           ["Encodec"],
    "Veena":                            ["SNAC"],
    "VITS":                             ["HiFi-GAN"],
    "WhisperSpeech":                    ["Vocos"],
}

ARCH_ORDER = ["No NN", "RNN", "Transformer", "Diffusion", "Flow", "LLM", "Masked"]


# ---------------------------------------------------------------------------
# EER threshold extraction
# ---------------------------------------------------------------------------
def get_eer_thresholds(combined_dir: Path) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for stem, model_name in SSL_STEMS.items():
        path = combined_dir / f"combined_{stem}.txt"
        if not path.exists():
            print(f"  [WARN] combined file not found: {path}")
            continue

        bonafide, spoof = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) == 4:
                    label, score_str = parts[2], parts[3]
                elif len(parts) == 3:
                    label, score_str = parts[1], parts[2]
                else:
                    continue
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                if not math.isfinite(score):
                    continue
                if label == "bonafide":
                    bonafide.append(score)
                elif label == "spoof":
                    spoof.append(score)

        if not bonafide or not spoof:
            print(f"  [WARN] no bonafide/spoof scores in {path}")
            continue

        _, threshold = compute_eer(np.array(bonafide), np.array(spoof))
        thresholds[model_name] = float(threshold)
        print(f"  {model_name:<25s} EER threshold = {threshold:.4f}")

    return thresholds


def save_thresholds(thresholds: Dict[str, float], out_path: Path) -> None:
    rows = [{"Model": m, "EER_Threshold": thresholds[m]}
            for m in ORDERED_MODELS if m in thresholds]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Thresholds saved: {out_path}")


# ---------------------------------------------------------------------------
# Core FAR computation
# ---------------------------------------------------------------------------
def compute_far(score_file: Path, threshold: float) -> Optional[float]:
    """
    Read a TTS score file (TSV with header) and return FAR at the given threshold.
    FAR = # spoof rows with score > threshold / # total spoof rows.
    """
    if not score_file.exists():
        return None
    try:
        df = pd.read_csv(score_file, sep="\t", dtype=str)
    except Exception as e:
        print(f"  [ERROR] cannot read {score_file}: {e}")
        return None

    if "key" not in df.columns or "score" not in df.columns:
        print(f"  [WARN] unexpected columns in {score_file}: {list(df.columns)}")
        return None

    spoof = df[df["key"] == "spoof"].copy()
    if spoof.empty:
        return None

    spoof["score"] = pd.to_numeric(spoof["score"], errors="coerce")
    spoof = spoof.dropna(subset=["score"])
    if spoof.empty:
        return None

    far = (spoof["score"] > threshold).sum() / len(spoof)
    return float(far) * 100


def build_far_matrix(
    tts_dir: Path,
    category: str,
    display_map: Dict[str, str],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    cat_dir = tts_dir / category
    tts_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    records: Dict[str, Dict[str, Optional[float]]] = {m: {} for m in ORDERED_MODELS}

    for tts_path in tts_dirs:
        dir_name = tts_path.name
        col_label = display_map.get(dir_name, dir_name)

        for stem, model_name in SSL_STEMS.items():
            threshold = thresholds.get(model_name)
            if threshold is None:
                records[model_name][col_label] = None
                continue
            score_file = tts_path / f"{stem}.txt"
            records[model_name][col_label] = compute_far(score_file, threshold)

    df = pd.DataFrame.from_dict(records, orient="index")
    df = df.loc[[m for m in ORDERED_MODELS if m in df.index]]
    col_order = [display_map.get(d.name, d.name) for d in tts_dirs]
    df = df[[c for c in col_order if c in df.columns]]
    df.index.name = "Model"
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def pooled_far(ar_df: pd.DataFrame, nar_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=ORDERED_MODELS)
    result["AR Mean"]      = ar_df.mean(axis=1)
    result["NAR Mean"]     = nar_df.mean(axis=1)
    result["Overall Mean"] = pd.concat([ar_df, nar_df], axis=1).mean(axis=1)
    result.index.name = "Model"
    return result


def far_by_tag(
    tts_dir: Path,
    tag_map: Dict[str, List[str]],
    tag_order: List[str],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    tag_rows: Dict[str, Dict[str, List[float]]] = {
        t: {m: [] for m in ORDERED_MODELS} for t in tag_order
    }

    for category_dir in [tts_dir / "AR", tts_dir / "NAR"]:
        if not category_dir.exists():
            continue
        for tts_path in sorted(category_dir.iterdir()):
            if not tts_path.is_dir():
                continue
            dir_name = tts_path.name
            tags = tag_map.get(dir_name)
            if tags is None:
                print(f"  [WARN] no tag metadata for {dir_name}; skipping")
                continue

            for stem, model_name in SSL_STEMS.items():
                threshold = thresholds.get(model_name)
                if threshold is None:
                    continue
                score_file = tts_path / f"{stem}.txt"
                far = compute_far(score_file, threshold)
                if far is None:
                    continue
                for tag in tags:
                    if tag in tag_rows:
                        tag_rows[tag][model_name].append(far)

    records: Dict[str, Dict[str, Optional[float]]] = {}
    for tag in tag_order:
        records[tag] = {}
        for model in ORDERED_MODELS:
            vals = tag_rows[tag][model]
            records[tag][model] = float(np.mean(vals)) if vals else None

    df = pd.DataFrame.from_dict(records, orient="columns")
    df = df.loc[[m for m in ORDERED_MODELS if m in df.index]]
    df.index.name = "Model"
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FAR matrices for TTS diversity analysis."
    )
    parser.add_argument(
        "--tts_dir", required=True,
        help="Root directory with AR/ and NAR/ subdirs of TTS score files.",
    )
    parser.add_argument(
        "--combined_dir", required=True,
        help="Directory with combined_<stem>.txt pooled score files.",
    )
    parser.add_argument(
        "--out_dir", default="outputs/far_results",
        help="Directory for output CSVs.",
    )
    args = parser.parse_args()

    tts_dir      = Path(args.tts_dir)
    combined_dir = Path(args.combined_dir)
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting per-model EER thresholds from pooled combined score files ...")
    thresholds = get_eer_thresholds(combined_dir)
    save_thresholds(thresholds, out_dir / "eer_thresholds.csv")
    print()

    print("Building AR FAR matrix ...")
    ar_df = build_far_matrix(tts_dir, "AR", AR_DISPLAY, thresholds)
    ar_df.to_csv(out_dir / "far_matrix_ar.csv")
    print(ar_df.to_string())

    print("\nBuilding NAR FAR matrix ...")
    nar_df = build_far_matrix(tts_dir, "NAR", NAR_DISPLAY, thresholds)
    nar_df.to_csv(out_dir / "far_matrix_nar.csv")
    print(nar_df.to_string())

    print("\nComputing pooled AR vs NAR FAR ...")
    pooled = pooled_far(ar_df, nar_df)
    pooled.to_csv(out_dir / "far_pooled.csv")
    print(pooled.to_string())

    print("\nComputing FAR by architecture ...")
    arch_df = far_by_tag(tts_dir, ARCH_TAGS, ARCH_ORDER, thresholds)
    arch_df.to_csv(out_dir / "far_by_architecture.csv")
    print(arch_df.to_string())

    print("\nComputing FAR by vocoder ...")
    all_vocoders = sorted({v for vlist in VOCODER_TAGS.values() for v in vlist})
    vocoder_df = far_by_tag(tts_dir, VOCODER_TAGS, all_vocoders, thresholds)
    vocoder_df.to_csv(out_dir / "far_by_vocoder.csv")
    print(vocoder_df.to_string())

    print(f"\nAll CSVs saved to {out_dir}")


if __name__ == "__main__":
    main()
