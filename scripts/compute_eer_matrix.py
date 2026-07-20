"""
Compute EER for all SSL models across all acoustic degradation conditions.

Handles two score-file formats:
  - 3-col (baseline): "utt_id  label  score"  (with optional header line)
  - 4-col (conditions): "utt_id  -  label  score"

Outputs a CSV file with SSL models as rows and conditions as columns.

Usage
-----
    python3 scripts/compute_eer_matrix.py \\
        --baseline_dir  /data/ssl_anti_spoofing/asd_superb_score_files/Baseline_by_Hashim/ \\
        --augmented_dir /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_category_augmented/ \\
        --output_csv    /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_category_augmented/eer_matrix.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import compute_eer   # type: ignore


# ---------------------------------------------------------------------------
# Model display names — matches ordered_models in create_heatmap.py exactly
# ---------------------------------------------------------------------------
ORDERED_MODELS = [
    "FBANK",
    "APC", "NPC", "Mockingjay-960h", "TERA", "DeCoAR 2.0",
    "wav2vec", "wav2vec 2.0 Base", "wav2vec 2.0 Large",
    "HuBERT Base", "HuBERT Large", "MR-HuBERT", "XLS-R",
    "UniSpeech-SAT", "Data2Vec", "WAVLABLM", "WavLM Large",
    "SSAST", "MAE-AST-FRAME",
]

# Condition file stems for standard conditions (file = stem + ".txt")
STANDARD_STEMS = {
    "FBANK":            "FBANK",
    "APC":              "APC",
    "NPC":              "NPC",
    "Mockingjay-960h":  "Mockingjay_960hr",
    "TERA":             "TERA",
    "DeCoAR 2.0":       "DeCoAR2",
    "wav2vec":          "wav2vec",
    "wav2vec 2.0 Base": "wav2vec2_Base",
    "wav2vec 2.0 Large":"wav2vec2_Large",
    "HuBERT Base":      "HuBERT_Base",
    "HuBERT Large":     "HuBERT_Large",
    "MR-HuBERT":        "MR_HuBERT",
    "XLS-R":            "XLS-R",
    "UniSpeech-SAT":    "Unispeech-SAT",
    "Data2Vec":         "Data2Vec",
    "WAVLABLM":         "WAVLABLM",
    "WavLM Large":      "WavLM_Large",
    "SSAST":            "SSAST",
    "MAE-AST-FRAME":    "MAE_AST_FRAME",
}

# Resampling: file = "linear_head_resamp_" + stem + ".txt"
# None means the file does not exist for that model.
RESAMP_STEMS = {
    "FBANK":            "fbank",
    "APC":              "apc",
    "NPC":              "npc",
    "Mockingjay-960h":  "mockingjay_960hr",
    "TERA":             "tera",
    "DeCoAR 2.0":       "decoar2",
    "wav2vec":          "wav2vec",
    "wav2vec 2.0 Base": "wav2vec2_base_960",
    "wav2vec 2.0 Large":"wav2vec2_large_ll60k",
    "HuBERT Base":      "hubert_base",
    "HuBERT Large":     "hubert_large_ll60k",
    "MR-HuBERT":        "multires_hubert_multilingual_large600k",
    "XLS-R":            "xls_r_300m",
    "UniSpeech-SAT":    "unispeech_sat_large",
    "Data2Vec":         "data2vec_large_ll60k",
    "WAVLABLM":         "wavlablm_ek_40k",
    "WavLM Large":      "wavlm_large",
    "SSAST":            "ssast_frame_base",
    "MAE-AST-FRAME":    "mae_ast_frame",
}

# Baseline: file = "linear_head_asvspoof5_" + stem + ".txt"
BASELINE_STEMS = {
    "FBANK":            "fbank",
    "APC":              "apc",
    "NPC":              "npc",
    "Mockingjay-960h":  "mockingjay_960hr",
    "TERA":             "tera",
    "DeCoAR 2.0":       "decoar2",
    "wav2vec":          "wav2vec",
    "wav2vec 2.0 Base": "wav2vec2_base_960",
    "wav2vec 2.0 Large":"wav2vec2_large_ll60k",
    "HuBERT Base":      "hubert_base",
    "HuBERT Large":     "hubert_large_ll60k",
    "MR-HuBERT":        "mr_hubert",
    "XLS-R":            "xls_r_300m",
    "UniSpeech-SAT":    "unispeech_sat_large",
    "Data2Vec":         "data2vec_large_ll60k",
    "WAVLABLM":         "wavlablm_ek_40k",
    "WavLM Large":      "wavlm_large",
    "SSAST":            "ssast_frame_base",
    "MAE-AST-FRAME":    "mae_ast_frame",
}

# Column order in the output CSV (and heatmap)
COLUMNS = ["Baseline", "Codec", "Noise", "Resampling", "Reverb", "Channel"]

# Maps column label → subdirectory name under --augmented_dir
CONDITION_DIR = {
    "Codec":      "Codec_Compression",
    "Noise":      "Additive_Noise",
    "Resampling": "Resampling",
    "Reverb":     "Reverberation",
    "Channel":    "Channel_Distortions",
}

HEADER_TOKENS = {"filename", "utt_id", "uttid", "key", "score", "label"}


def compute_eer_from_file(path: str) -> Optional[float]:
    """
    Parse a score file in either format and return EER (%).
      3-col: utt_id label score          (baseline files, may have a header)
      4-col: utt_id - label score        (condition files)
    """
    bonafide, spoof = [], []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0].lower() in HEADER_TOKENS:
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
                if label == "bonafide":
                    bonafide.append(score)
                elif label == "spoof":
                    spoof.append(score)
    except OSError as e:
        print(f"  [ERROR] cannot open {path}: {e}")
        return None

    if not bonafide or not spoof:
        print(f"  [WARN]  no bonafide/spoof scores in {path}")
        return None

    eer, _ = compute_eer(np.array(bonafide), np.array(spoof))
    return eer * 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute EER matrix for all SSL models × all conditions."
    )
    parser.add_argument("--baseline_dir",  required=True,
                        help="Directory with Hashim baseline .txt files.")
    parser.add_argument("--augmented_dir", required=True,
                        help="Root of augmented condition score directories.")
    parser.add_argument("--output_csv",    required=True,
                        help="Output CSV path.")
    args = parser.parse_args()

    import csv
    results = []   # list of dicts, one per model

    for model in ORDERED_MODELS:
        row: dict = {"Model": model}

        # ---- Baseline ----
        stem = BASELINE_STEMS[model]
        path = os.path.join(args.baseline_dir, f"linear_head_asvspoof5_{stem}.txt")
        if os.path.isfile(path):
            eer = compute_eer_from_file(path)
            row["Baseline"] = f"{eer:.2f}" if eer is not None else ""
        else:
            print(f"  [MISS] Baseline  {model} → {path}")
            row["Baseline"] = ""

        # ---- Degradation conditions ----
        for col in COLUMNS[1:]:   # skip Baseline
            cond_dir = os.path.join(args.augmented_dir, CONDITION_DIR[col])

            if col == "Resampling":
                stem = RESAMP_STEMS[model]
                if stem is None:
                    print(f"  [MISS] Resampling  {model} → no file")
                    row[col] = ""
                    continue
                path = os.path.join(cond_dir, f"linear_head_resamp_{stem}.txt")
            else:
                stem = STANDARD_STEMS[model]
                path = os.path.join(cond_dir, f"{stem}.txt")

            if os.path.isfile(path):
                eer = compute_eer_from_file(path)
                row[col] = f"{eer:.2f}" if eer is not None else ""
            else:
                print(f"  [MISS] {col:12s}  {model} → {path}")
                row[col] = ""

        # Print progress line
        vals = "  ".join(f"{c}={row[c] or 'N/A':>6}" for c in COLUMNS)
        print(f"  {model:<20} {vals}")

        results.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model"] + COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
