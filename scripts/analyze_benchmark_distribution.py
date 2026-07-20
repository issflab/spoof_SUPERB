"""
analyze_benchmark_distribution.py

Analyzes distributions relevant to the Spoof-SUPERB benchmark:

1. Dataset distribution in the pooled combined score file
   (used for per-model EER threshold calibration).

2. Architecture distribution across the five TTS-diversity datasets
   (ASV19, ASV5, FamousFigures, MLAAD-En, Spoof-Celeb), broken down by
   architecture type: No NN, RNN, Transformer, Diffusion, Flow Matching, LLM.
   Multi-label: a system that belongs to multiple architectures is counted
   in each.

3. Per-dataset EER thresholds for all 19 SSL models across each of the five
   TTS-relevant datasets, plus a TTS-pooled threshold and the existing
   all-data-pooled threshold — enabling direct comparison of how threshold
   calibration varies with dataset choice.

Outputs (saved to --out_dir):
  combined_dataset_distribution.csv  — per-dataset bonafide/spoof counts
  tts_architecture_by_dataset.csv    — utterances per (Architecture, Dataset)
  per_dataset_eer_thresholds.csv     — EER threshold per (Model, Dataset)

Usage
-----
    python3 scripts/analyze_benchmark_distribution.py \\
        --combined_dir /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_ssl_model \\
        --asv19_protocol /data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \\
        --tts_dir /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_TTS \\
        --out_dir /data/ssl_anti_spoofing/asd_superb_score_files/far_results
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

# Import compute_eer from the parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import compute_eer

# ---------------------------------------------------------------------------
# Architecture tags — one entry per TTS system directory name.
# Multi-label: systems may appear in more than one architecture group.
# "Flow Matching" used instead of "Flow" per paper convention.
# ---------------------------------------------------------------------------
ARCH_TAGS: Dict[str, list] = {
    # NAR systems
    "ASR_TTS_NAR":                      ["No NN"],
    "BVAE-TTS":                         ["RNN"],
    "E2TTS":                            ["Transformer", "Flow Matching", "Masked"],
    "F5TTS":                            ["Diffusion", "Flow Matching", "Masked"],
    "FastPitch":                        ["Transformer"],
    "GlowTTS":                          ["Transformer", "Flow Matching"],
    "GradTTS":                          ["Diffusion"],
    "Kokoro":                           ["Diffusion"],
    "MaryTTS":                          ["No NN"],
    "MaskGCT":                          ["Transformer", "LLM", "Masked"],
    "MatchaTTS":                        ["Transformer", "Flow Matching"],
    "MetaVoice-1B":                     ["Transformer", "LLM"],
    "NN-VC":                            ["RNN"],
    "Non-parallel_VC":                  ["No NN"],
    "SpeechT5":                         ["Transformer"],
    "SpeedySpeech":                     ["RNN"],
    "StyleTTS2":                        ["Diffusion"],
    "ToucanTTS":                        ["Transformer", "Masked"],
    "Transfer-function_VC":             ["No NN"],
    "VixTTS":                           ["LLM"],
    "VoiceTextWebAPI":                  ["No NN"],
    "XTTS":                             ["LLM"],
    "YourTTS":                          ["Transformer"],
    "Zero-Shot_VC":                     ["Transformer"],
    "ZipVoice":                         ["Transformer", "Flow Matching", "Masked"],
    "ZMM-TTS":                          ["Transformer", "LLM"],
    # AR systems
    "ASR_TTS_AR":                       ["RNN"],
    "Bark":                             ["Transformer", "LLM"],
    "Capacitron":                       ["RNN"],
    "CosyVoice2":                       ["Flow Matching", "LLM"],
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

ARCH_ORDER = ["No NN", "RNN", "Transformer", "Diffusion", "Flow Matching", "LLM", "Masked"]

# Five TTS-relevant benchmark datasets (in display order)
DATASET_ORDER = ["ASV19", "ASV5", "FamousFigures", "MLAAD-En", "Spoof-Celeb"]

# ---------------------------------------------------------------------------
# SSL model stems and display names (19 models used in the main benchmark)
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

# Regex matching any acoustic-degradation suffix added to ASV19/21 LA filenames
_DEGRADE_RE = re.compile(
    r"_(babble|cafe|white|volvo|RT_\d|resample|c\d{2}_|codec)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Helper: load ASVspoof 2019 LA eval utterance IDs
# ---------------------------------------------------------------------------

def load_asv19_eval_ids(protocol_path: str) -> Set[str]:
    """Return the set of bare utterance IDs from the ASV19 LA eval CM protocol."""
    ids: Set[str] = set()
    with open(protocol_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])   # e.g. LA_E_2834763
    return ids


# ---------------------------------------------------------------------------
# Dataset classifier for combined score file entries
# ---------------------------------------------------------------------------

def classify_filepath(filepath: str, asv19_ids: Set[str]) -> str:
    """Map a filepath from the combined score file to its dataset label."""

    # FamousFigures: stored under NFS path with /famousfigures/
    if "/famousfigures/" in filepath:
        return "FamousFigures"

    # MLAAD — English subset only
    if re.match(r"^MLAAD/(?:fake|real)/en/", filepath):
        return "MLAAD-En"
    if re.match(r"^MAILabs/en_", filepath):
        return "MLAAD-En"
    if filepath.startswith("MLAAD/") or filepath.startswith("MAILabs/"):
        return "MLAAD-Other"

    # Strip to bare filename for short-form IDs
    bare = filepath.split("/")[-1]
    bare = bare.replace(".wav", "").replace(".flac", "")

    # ASV5: E_ followed by exactly 10 digits
    if re.match(r"^E_\d{10}$", bare):
        return "ASV5"

    # LA_E_ prefix — could be ASV19 or ASV21 LA
    if bare.startswith("LA_E_"):
        if _DEGRADE_RE.search(bare):
            return "ASV19/21 LA (degraded)"
        # Base ID = strip any known degradation tokens
        base_id = re.split(r"_(?:babble|cafe|white|volvo|RT|resample)", bare)[0]
        return "ASV19" if base_id in asv19_ids else "ASV21 LA"

    # ASV21 DF
    if bare.startswith("DF_") or bare.startswith("df_"):
        return "ASV21 DF"

    # SpoofCeleb: aXX/idXXXXX/... prefix (VoxCeleb-style paths)
    if re.match(r"^a\d{2}/id\d+/", filepath):
        return "Spoof-Celeb"
    if filepath.startswith("id"):
        return "Spoof-Celeb"
    # YouTube IDs: alphanumeric + dash + underscore, no directory separator
    if "/" not in filepath and re.match(r"^[-_A-Za-z0-9]{20,35}\.(wav|flac)$", filepath):
        return "Spoof-Celeb"

    # Malformed entries (bare integers, etc.) — skip
    if re.match(r"^\d+$", filepath):
        return "Malformed"

    return "Unknown"


# ---------------------------------------------------------------------------
# Analysis 1: dataset distribution in combined score file
# ---------------------------------------------------------------------------

def compute_dataset_distribution(
    combined_file: str, asv19_ids: Set[str]
) -> pd.DataFrame:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"Bonafide": 0, "Spoof": 0})

    with open(combined_file) as fh:
        for line in fh:
            parts = line.strip().split(" ")
            if len(parts) < 4:
                continue
            filepath, label = parts[0], parts[2]
            if label == "bonafide":
                counts[classify_filepath(filepath, asv19_ids)]["Bonafide"] += 1
            elif label == "spoof":
                counts[classify_filepath(filepath, asv19_ids)]["Spoof"] += 1

    # Ordered output: TTS-relevant first, then other benchmark data, then rest
    preferred_order = [
        "ASV19", "ASV5", "FamousFigures", "MLAAD-En", "Spoof-Celeb",
        "ASV21 LA", "ASV19/21 LA (degraded)", "ASV21 DF", "MLAAD-Other", "Unknown",
    ]
    all_keys = list(counts.keys())
    ordered = [k for k in preferred_order if k in all_keys] + \
              [k for k in all_keys if k not in preferred_order]

    rows = []
    for ds in ordered:
        bf = counts[ds]["Bonafide"]
        sp = counts[ds]["Spoof"]
        rows.append({"Dataset": ds, "Bonafide": bf, "Spoof": sp, "Total": bf + sp})

    df = pd.DataFrame(rows)
    total = pd.DataFrame([{
        "Dataset": "TOTAL",
        "Bonafide": df["Bonafide"].sum(),
        "Spoof": df["Spoof"].sum(),
        "Total": df["Total"].sum(),
    }])
    return pd.concat([df, total], ignore_index=True)


# ---------------------------------------------------------------------------
# Analysis 2: architecture distribution across TTS diversity datasets
# ---------------------------------------------------------------------------

def compute_architecture_distribution(tts_dir: str) -> pd.DataFrame:
    """
    Count utterances per (Dataset, Architecture) using TTS diversity score files.

    Uses one SSL model (fbank) as the representative since every model scores
    the same set of utterances.  Multi-label: a system in N architecture groups
    contributes its utterance count to each group.
    """
    # (dataset, arch) -> utterance count
    arch_counts: Dict[tuple, int] = defaultdict(int)
    unknown_systems = set()

    for cat in ("AR", "NAR"):
        cat_path = os.path.join(tts_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for system in sorted(os.listdir(cat_path)):
            sys_path = os.path.join(cat_path, system)
            if not os.path.isdir(sys_path):
                continue

            files = os.listdir(sys_path)
            # Prefer fbank score file as representative
            score_file = next(
                (f for f in files if "fbank" in f.lower()),
                files[0] if files else None,
            )
            if not score_file:
                continue

            tags = ARCH_TAGS.get(system)
            if tags is None:
                unknown_systems.add(system)
                continue

            # Read (dataset, count) from the TSV
            dataset_counts: Dict[str, int] = defaultdict(int)
            with open(os.path.join(sys_path, score_file)) as fh:
                next(fh)  # skip header
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 1:
                        dataset_counts[parts[0]] += 1

            for ds, cnt in dataset_counts.items():
                for tag in tags:
                    arch_counts[(ds, tag)] += cnt

    if unknown_systems:
        print(f"  [WARN] Systems missing from ARCH_TAGS: {sorted(unknown_systems)}")

    rows = []
    for ds in DATASET_ORDER:
        for arch in ARCH_ORDER:
            rows.append({
                "Architecture": arch,
                "Dataset": ds,
                "Utterances": arch_counts.get((ds, arch), 0),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3: per-dataset EER thresholds for all SSL models
# ---------------------------------------------------------------------------

def compute_per_dataset_thresholds(
    combined_dir: str,
    asv19_ids: Set[str],
) -> pd.DataFrame:
    """
    For each of the 19 SSL models, compute the EER threshold separately on:
      - each of the 5 TTS-relevant datasets (ASV19, ASV5, FamousFigures,
        MLAAD-En, Spoof-Celeb)
      - Pooled TTS  : the 5 datasets combined (no degraded/ASV21 data)
      - Pooled All  : every sample in the combined file (current approach)

    Returns a DataFrame with rows = models, columns = datasets + pooled columns.
    """
    tts_datasets = DATASET_ORDER   # the 5 TTS-relevant datasets
    col_order = tts_datasets + ["Pooled TTS", "Pooled All"]

    results: Dict[str, Dict] = {}

    for stem, model_name in SSL_STEMS.items():
        combined_file = os.path.join(combined_dir, f"combined_{stem}.txt")
        if not os.path.exists(combined_file):
            print(f"  [WARN] Missing: combined_{stem}.txt — skipping")
            continue

        print(f"  {model_name:<30}", end="", flush=True)

        # Accumulate (bonafide_scores, spoof_scores) per dataset bucket
        bf_scores: Dict[str, list] = {ds: [] for ds in tts_datasets}
        sp_scores: Dict[str, list] = {ds: [] for ds in tts_datasets}
        all_bf: list = []
        all_sp: list = []

        with open(combined_file) as fh:
            for line in fh:
                parts = line.strip().split(" ")
                if len(parts) < 4:
                    continue
                filepath, label, score_str = parts[0], parts[2], parts[3]
                if label not in ("bonafide", "spoof"):
                    continue
                try:
                    score = float(score_str)
                except ValueError:
                    continue

                ds = classify_filepath(filepath, asv19_ids)

                # Always accumulate for Pooled All
                if label == "bonafide":
                    all_bf.append(score)
                else:
                    all_sp.append(score)

                # Only accumulate for per-dataset / Pooled TTS if TTS-relevant
                if ds in tts_datasets:
                    if label == "bonafide":
                        bf_scores[ds].append(score)
                    else:
                        sp_scores[ds].append(score)

        row: Dict[str, float] = {}

        # Per-dataset thresholds
        pooled_tts_bf: list = []
        pooled_tts_sp: list = []
        for ds in tts_datasets:
            bf, sp = bf_scores[ds], sp_scores[ds]
            pooled_tts_bf.extend(bf)
            pooled_tts_sp.extend(sp)
            if bf and sp:
                _, thresh = compute_eer(np.array(bf), np.array(sp))
                row[ds] = float(thresh)
            else:
                row[ds] = float("nan")

        # Pooled TTS threshold (5 datasets, no degraded/ASV21 data)
        if pooled_tts_bf and pooled_tts_sp:
            _, thresh = compute_eer(
                np.array(pooled_tts_bf), np.array(pooled_tts_sp)
            )
            row["Pooled TTS"] = float(thresh)
        else:
            row["Pooled TTS"] = float("nan")

        # Pooled All threshold (current approach: all data in combined file)
        if all_bf and all_sp:
            _, thresh = compute_eer(np.array(all_bf), np.array(all_sp))
            row["Pooled All"] = float(thresh)
        else:
            row["Pooled All"] = float("nan")

        results[model_name] = row
        print("done")

    # Build ordered DataFrame (rows = models, columns = datasets)
    df = pd.DataFrame(results).T
    df = df.loc[[m for m in ORDERED_MODELS if m in df.index], col_order]
    df.index.name = "Model"
    return df.round(6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Spoof-SUPERB benchmark dataset and architecture distributions."
    )
    parser.add_argument(
        "--combined_dir", required=True,
        help="Directory containing combined_<stem>.txt files (scores_by_ssl_model).",
    )
    parser.add_argument(
        "--asv19_protocol", required=True,
        help="ASVspoof 2019 LA eval CM protocol file path.",
    )
    parser.add_argument(
        "--tts_dir", required=True,
        help="Root of scores_by_TTS directory (contains AR/ and NAR/).",
    )
    parser.add_argument(
        "--out_dir", default="outputs/distributions",
        help="Output directory for CSV files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: load ASV19 eval IDs ────────────────────────────────────────
    print("Loading ASVspoof 2019 LA eval utterance IDs ...")
    asv19_ids = load_asv19_eval_ids(args.asv19_protocol)
    print(f"  {len(asv19_ids):,} IDs loaded")

    # ── Step 2: dataset distribution ──────────────────────────────────────
    combined_file = os.path.join(args.combined_dir, "combined_fbank.txt")
    print(f"\nAnalyzing dataset distribution in combined_fbank.txt ...")
    dist_df = compute_dataset_distribution(combined_file, asv19_ids)

    out1 = os.path.join(args.out_dir, "combined_dataset_distribution.csv")
    dist_df.to_csv(out1, index=False)
    print(f"  Saved: {out1}")
    print()
    print(dist_df.to_string(index=False))

    # ── Step 3: architecture distribution ────────────────────────────────
    print(f"\nAnalyzing architecture distribution from TTS score files ...")
    arch_df = compute_architecture_distribution(args.tts_dir)

    out2 = os.path.join(args.out_dir, "tts_architecture_by_dataset.csv")
    arch_df.to_csv(out2, index=False)
    print(f"  Saved: {out2}")

    # Print pivot table for quick inspection
    pivot = arch_df.pivot(index="Architecture", columns="Dataset", values="Utterances")
    pivot = pivot.reindex(ARCH_ORDER, axis=0)[DATASET_ORDER]
    pivot["Total"] = pivot.sum(axis=1)
    # Add % of total column (unique utterances, no multi-label inflation)
    total_utts = arch_df.groupby("Dataset")["Utterances"].sum()
    print()
    print("  Architecture × Dataset (utterance count, multi-label):")
    print(pivot.to_string())
    print()
    print("  Unique utterances per dataset (multi-label sums differ):")
    for ds in DATASET_ORDER:
        # Get unique count by reading one file per system
        total = arch_df[arch_df["Dataset"] == ds]["Utterances"].max()
    print(total_utts.to_string())

    # ── Step 4: per-dataset EER thresholds ───────────────────────────────
    print(f"\nComputing per-dataset EER thresholds for all {len(SSL_STEMS)} models ...")
    print("  (reads all 19 combined score files — this takes a few minutes)\n")
    thresh_df = compute_per_dataset_thresholds(args.combined_dir, asv19_ids)

    out3 = os.path.join(args.out_dir, "per_dataset_eer_thresholds.csv")
    thresh_df.to_csv(out3)
    print(f"\n  Saved: {out3}")
    print()
    print(thresh_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
