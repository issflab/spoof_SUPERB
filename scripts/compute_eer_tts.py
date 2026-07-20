"""
compute_eer_tts.py

Computes per-TTS-system EER against a pooled bonafide reference collection.

Approach
--------
  1. For each SSL model, pool ALL bonafide z-scores from all 10 datasets'
     normalized linear_head files into a single reference collection.
     These are already on a common z-score scale (mean=0, std=1 per dataset),
     providing a diverse, consistent bonafide reference for all TTS systems.

  2. For each (TTS system, SSL model), compute EER between the system's spoof
     z-scores and the pooled bonafide reference.

  EER directly measures separability: how well the model discriminates each
  TTS system from genuine speech, independent of any fixed threshold.

Outputs (parallel structure to FAR outputs)
-------------------------------------------
  eer_matrix_ar.csv       — 19 SSL models × 30 AR TTS systems
  eer_matrix_nar.csv      — 19 SSL models × 26 NAR TTS systems
  eer_pooled.csv          — mean EER per SSL model (AR | NAR | Overall)
  eer_by_architecture.csv — mean EER per SSL model per architecture tag
  eer_by_vocoder.csv      — mean EER per SSL model per vocoder family
  pooled_bonafide_counts.csv — bonafide sample count per (dataset, SSL model)

Usage
-----
    python3 scripts/compute_eer_tts.py \\
        --norm_dir  /data/ssl_anti_spoofing/asd_superb_score_files/linear_head_normalized_scores \\
        --tts_dir   /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_TTS_norm \\
        --out_dir   /data/ssl_anti_spoofing/asd_superb_score_files/eer_results_pooled_bonafide
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
# Constants
# ---------------------------------------------------------------------------
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

AR_DISPLAY: Dict[str, str] = {
    "ASR_TTS_AR":                          "ASR-TTS",
    "Bark":                                "Bark",
    "Capacitron":                          "Capacitron",
    "CosyVoice2":                          "CosyVoice2",
    "FishSpeech":                          "FishSpeech",
    "FishTTS":                             "FishTTS",
    "Indri-TTS-0.1":                       "Indri",
    "In-house_ASR-based":                  "In-house ASR",
    "LLASA":                               "LLASA",
    "MQTTS":                               "MQTTS",
    "Multi-scale_Transformer":             "MST",
    "Multi-scale_Transformer_pre-trained": "MST-pt",
    "NeuralHMM":                           "NeuralHMM",
    "Neural_TTS":                          "Neural TTS",
    "NN-SPSS_TTS":                         "NN-SPSS",
    "NN-TTS":                              "NN-TTS",
    "Openaudio-S1-Mini":                   "Openaudio",
    "OpenVoiceV2":                         "OpenVoice2",
    "Overflow":                            "Overflow",
    "Parler_TTS":                          "Parler",
    "Sesame_CSM_1B":                       "Sesame CSM",
    "Spark_TTS":                           "Spark TTS",
    "SSRSpeech":                           "SSRSpeech",
    "Tacotron_2":                          "Tacotron 2",
    "Tortoise":                            "Tortoise",
    "TransformerTTS":                      "TransformerTTS",
    "VALL-E":                              "VALL-E",
    "Veena":                               "Veena",
    "VITS":                                "VITS",
    "WhisperSpeech":                       "WhisperSpeech",
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

ARCH_TAGS: Dict[str, List[str]] = {
    "ASR_TTS_NAR":                         ["No NN"],
    "BVAE-TTS":                            ["RNN"],
    "E2TTS":                               ["Transformer", "Flow", "Masked"],
    "F5TTS":                               ["Diffusion", "Flow", "Masked"],
    "FastPitch":                           ["Transformer"],
    "GlowTTS":                             ["Transformer", "Flow"],
    "GradTTS":                             ["Diffusion"],
    "Kokoro":                              ["Diffusion"],
    "MaryTTS":                             ["No NN"],
    "MaskGCT":                             ["Transformer", "LLM", "Masked"],
    "MatchaTTS":                           ["Transformer", "Flow"],
    "MetaVoice-1B":                        ["Transformer", "LLM"],
    "NN-VC":                               ["RNN"],
    "Non-parallel_VC":                     ["No NN"],
    "SpeechT5":                            ["Transformer"],
    "SpeedySpeech":                        ["RNN"],
    "StyleTTS2":                           ["Diffusion"],
    "ToucanTTS":                           ["Transformer", "Masked"],
    "Transfer-function_VC":                ["No NN"],
    "VixTTS":                              ["LLM"],
    "VoiceTextWebAPI":                     ["No NN"],
    "XTTS":                                ["LLM"],
    "YourTTS":                             ["Transformer"],
    "Zero-Shot_VC":                        ["Transformer"],
    "ZipVoice":                            ["Transformer", "Flow", "Masked"],
    "ZMM-TTS":                             ["Transformer", "LLM"],
    "ASR_TTS_AR":                          ["RNN"],
    "Bark":                                ["Transformer", "LLM"],
    "Capacitron":                          ["RNN"],
    "CosyVoice2":                          ["Flow", "LLM"],
    "FishSpeech":                          ["Transformer", "LLM"],
    "FishTTS":                             ["Transformer", "LLM"],
    "Indri-TTS-0.1":                       ["Transformer", "LLM"],
    "In-house_ASR-based":                  ["No NN"],
    "LLASA":                               ["Transformer", "LLM"],
    "MQTTS":                               ["Transformer"],
    "Multi-scale_Transformer":             ["Transformer", "LLM"],
    "Multi-scale_Transformer_pre-trained": ["Transformer", "LLM"],
    "NeuralHMM":                           ["RNN"],
    "Neural_TTS":                          ["RNN"],
    "NN-SPSS_TTS":                         ["RNN"],
    "NN-TTS":                              ["RNN"],
    "Openaudio-S1-Mini":                   ["Transformer", "LLM"],
    "OpenVoiceV2":                         ["RNN"],
    "Overflow":                            ["RNN"],
    "Parler_TTS":                          ["Transformer", "LLM"],
    "Sesame_CSM_1B":                       ["Transformer", "LLM"],
    "Spark_TTS":                           ["Transformer", "LLM"],
    "SSRSpeech":                           ["Transformer", "Masked"],
    "Tacotron_2":                          ["RNN"],
    "Tortoise":                            ["Transformer", "Diffusion"],
    "TransformerTTS":                      ["Transformer"],
    "VALL-E":                              ["LLM"],
    "Veena":                               ["Transformer", "LLM"],
    "VITS":                                ["RNN"],
    "WhisperSpeech":                       ["Transformer", "LLM"],
}

VOCODER_TAGS: Dict[str, List[str]] = {
    "ASR_TTS_NAR":                         ["STRAIGHT"],
    "BVAE-TTS":                            ["WaveGlow"],
    "E2TTS":                               ["BigVGAN"],
    "F5TTS":                               ["Vocos"],
    "FastPitch":                           ["WaveGlow"],
    "GlowTTS":                             ["WaveGlow"],
    "GradTTS":                             ["HiFi-GAN"],
    "Kokoro":                              ["iSTFTNet"],
    "MaryTTS":                             ["Built-in"],
    "MaskGCT":                             ["Vocos"],
    "MatchaTTS":                           ["HiFi-GAN"],
    "MetaVoice-1B":                        ["Multi-band Diffusion"],
    "NN-VC":                               ["Waveform Filtering"],
    "Non-parallel_VC":                     ["PLDA"],
    "SpeechT5":                            ["HiFi-GAN"],
    "SpeedySpeech":                        ["MelGAN"],
    "StyleTTS2":                           ["HiFi-GAN", "iSTFTNet"],
    "ToucanTTS":                           ["BigVGAN"],
    "Transfer-function_VC":                ["Spectral Filtering"],
    "VixTTS":                              ["HiFi-GAN", "iSTFTNet"],
    "VoiceTextWebAPI":                     ["World"],
    "XTTS":                                ["HiFi-GAN", "iSTFTNet"],
    "YourTTS":                             ["HiFi-GAN"],
    "Zero-Shot_VC":                        ["HiFi-GAN"],
    "ZipVoice":                            ["Vocos"],
    "ZMM-TTS":                             ["HiFi-GAN"],
    "ASR_TTS_AR":                          ["WaveNet"],
    "Bark":                                ["SoundStream"],
    "Capacitron":                          ["WaveNet"],
    "CosyVoice2":                          ["HiFi-GAN"],
    "FishSpeech":                          ["Firefly-GAN", "FF-GAN"],
    "FishTTS":                             ["Firefly-GAN"],
    "Indri-TTS-0.1":                       ["Mimi"],
    "In-house_ASR-based":                  ["BigVGAN"],
    "LLASA":                               ["Vocos"],
    "MQTTS":                               ["HiFi-GAN"],
    "Multi-scale_Transformer":             ["Encodec"],
    "Multi-scale_Transformer_pre-trained": ["Encodec"],
    "NeuralHMM":                           ["WaveGlow"],
    "Neural_TTS":                          ["WaveNet"],
    "NN-SPSS_TTS":                         ["Vocaine"],
    "NN-TTS":                              ["World", "Neural-Source-Filter"],
    "Openaudio-S1-Mini":                   ["FF-GAN"],
    "OpenVoiceV2":                         ["HiFi-GAN"],
    "Overflow":                            ["HiFi-GAN"],
    "Parler_TTS":                          ["RVQ"],
    "Sesame_CSM_1B":                       ["Mimi"],
    "Spark_TTS":                           ["BiCodec"],
    "SSRSpeech":                           ["Transformer"],
    "Tacotron_2":                          ["WaveNet", "WaveRNN", "Griffin-Lim"],
    "Tortoise":                            ["UnivNet"],
    "TransformerTTS":                      ["WaveNet"],
    "VALL-E":                              ["Encodec"],
    "Veena":                               ["SNAC"],
    "VITS":                                ["HiFi-GAN"],
    "WhisperSpeech":                       ["Vocos"],
}

ARCH_ORDER = ["No NN", "RNN", "Transformer", "Diffusion", "Flow", "LLM", "Masked"]


# ---------------------------------------------------------------------------
# Step 1: Build pooled bonafide collection per SSL model
# ---------------------------------------------------------------------------
def build_bonafide_pool(
    norm_dir: str,
) -> tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Read all 10 normalized linear_head files per SSL model and pool bonafide scores.

    Returns:
        pool: {stem: np.ndarray of float32 bonafide z-scores}
        counts_df: DataFrame with per-(dataset, model) bonafide counts
    """
    pool: Dict[str, np.ndarray] = {}
    count_rows = []

    for stem, model_name in SSL_STEMS.items():
        scores: List[float] = []
        print(f"  {model_name:<30}", end="", flush=True)

        for dataset in ALL_DATASETS:
            path = os.path.join(norm_dir, f"linear_head_{dataset}_{stem}.txt")
            if not os.path.exists(path):
                count_rows.append({"Model": model_name, "Dataset": dataset, "Bonafide": 0})
                continue

            n_bon = 0
            with open(path) as f:
                for line in f:
                    parts = line.rstrip("\n").split(" ")
                    if len(parts) < 4 or parts[2] != "bonafide":
                        continue
                    try:
                        val = float(parts[3])
                        if math.isfinite(val):
                            scores.append(val)
                            n_bon += 1
                    except (ValueError, IndexError):
                        continue
            count_rows.append({"Model": model_name, "Dataset": dataset, "Bonafide": n_bon})

        pool[stem] = np.array(scores, dtype=np.float32)
        print(f"  {len(pool[stem]):>10,} bonafide scores pooled")

    counts_df = pd.DataFrame(count_rows)
    return pool, counts_df


# ---------------------------------------------------------------------------
# Step 2: Compute EER for one TTS score file vs. pooled bonafide
# ---------------------------------------------------------------------------
def compute_eer_for_file(
    score_file: Path,
    bonafide_pool: np.ndarray,
) -> Optional[float]:
    """
    Read a cleaned TTS norm score file (TSV, all data rows are spoof).
    Compute EER between spoof z-scores and pooled bonafide z-scores.
    Returns EER as a percentage (0–50), or None if data is insufficient.
    """
    if not score_file.exists():
        return None
    if len(bonafide_pool) < 10:
        return None

    spoof_scores: List[float] = []
    with open(score_file) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                val = float(parts[3])
                if math.isfinite(val):
                    spoof_scores.append(val)
            except (ValueError, IndexError):
                continue

    if len(spoof_scores) < 5:
        return None

    spoof_arr = np.array(spoof_scores, dtype=np.float32)
    eer, _ = compute_eer(bonafide_pool, spoof_arr)
    return float(eer) * 100


# ---------------------------------------------------------------------------
# Step 3: Build full EER matrices
# ---------------------------------------------------------------------------
def build_eer_matrix(
    tts_dir: Path,
    category: str,
    display_map: Dict[str, str],
    pool: Dict[str, np.ndarray],
) -> pd.DataFrame:
    cat_dir = tts_dir / category
    tts_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    records: Dict[str, Dict[str, Optional[float]]] = {m: {} for m in ORDERED_MODELS}

    for tts_path in tts_dirs:
        dir_name = tts_path.name
        col_label = display_map.get(dir_name, dir_name)
        print(f"    {dir_name}", end="", flush=True)

        for stem, model_name in SSL_STEMS.items():
            bonafide_pool = pool.get(stem, np.array([]))
            score_file = tts_path / f"{stem}.txt"
            records[model_name][col_label] = compute_eer_for_file(score_file, bonafide_pool)

        print()

    df = pd.DataFrame.from_dict(records, orient="index")
    df = df.loc[[m for m in ORDERED_MODELS if m in df.index]]
    col_order = [display_map.get(d.name, d.name) for d in tts_dirs]
    df = df[[c for c in col_order if c in df.columns]]
    df.index.name = "Model"
    return df


# ---------------------------------------------------------------------------
# Step 4: Aggregation helpers
# ---------------------------------------------------------------------------
def pooled_eer(ar_df: pd.DataFrame, nar_df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=ORDERED_MODELS)
    result["AR Mean"]      = ar_df.mean(axis=1)
    result["NAR Mean"]     = nar_df.mean(axis=1)
    result["Overall Mean"] = pd.concat([ar_df, nar_df], axis=1).mean(axis=1)
    result.index.name = "Model"
    return result


def eer_by_tag(
    tts_dir: Path,
    tag_map: Dict[str, List[str]],
    tag_order: List[str],
    pool: Dict[str, np.ndarray],
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
                bonafide_pool = pool.get(stem, np.array([]))
                score_file = tts_path / f"{stem}.txt"
                eer = compute_eer_for_file(score_file, bonafide_pool)
                if eer is None:
                    continue
                for tag in tags:
                    if tag in tag_rows:
                        tag_rows[tag][model_name].append(eer)

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
        description="Compute per-TTS-system EER against pooled bonafide reference."
    )
    parser.add_argument(
        "--norm_dir", required=True,
        help="Z-scored linear_head normalized score files (linear_head_normalized_scores/).",
    )
    parser.add_argument(
        "--tts_dir", required=True,
        help="Root of scores_by_TTS_norm directory (AR/ and NAR/ subdirs).",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for EER CSV files.",
    )
    args = parser.parse_args()

    tts_dir = Path(args.tts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1 — Building pooled bonafide collection per model")
    print("=" * 60)
    pool, counts_df = build_bonafide_pool(args.norm_dir)
    counts_df.to_csv(out_dir / "pooled_bonafide_counts.csv", index=False)
    print(f"\n  Bonafide counts saved: {out_dir / 'pooled_bonafide_counts.csv'}")
    print(f"  Smallest pool: {min(len(v) for v in pool.values()):,} scores")
    print(f"  Largest pool:  {max(len(v) for v in pool.values()):,} scores")

    print()
    print("=" * 60)
    print("Step 2 — Computing EER matrices")
    print("=" * 60)

    print("\nAR systems:")
    ar_df = build_eer_matrix(tts_dir, "AR", AR_DISPLAY, pool)
    ar_df.to_csv(out_dir / "eer_matrix_ar.csv")
    print(ar_df.to_string())

    print("\nNAR systems:")
    nar_df = build_eer_matrix(tts_dir, "NAR", NAR_DISPLAY, pool)
    nar_df.to_csv(out_dir / "eer_matrix_nar.csv")
    print(nar_df.to_string())

    print()
    print("=" * 60)
    print("Step 3 — Aggregations")
    print("=" * 60)

    print("\nPooled AR vs NAR mean EER:")
    pool_df = pooled_eer(ar_df, nar_df)
    pool_df.to_csv(out_dir / "eer_pooled.csv")
    print(pool_df.to_string())

    print("\nEER by architecture tag:")
    arch_df = eer_by_tag(tts_dir, ARCH_TAGS, ARCH_ORDER, pool)
    arch_df.to_csv(out_dir / "eer_by_architecture.csv")
    print(arch_df.to_string())

    print("\nEER by vocoder:")
    all_vocoders = sorted({v for vlist in VOCODER_TAGS.values() for v in vlist})
    voc_df = eer_by_tag(tts_dir, VOCODER_TAGS, all_vocoders, pool)
    voc_df.to_csv(out_dir / "eer_by_vocoder.csv")
    print(voc_df.to_string())

    print(f"\nAll CSVs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
