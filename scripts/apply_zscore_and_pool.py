"""
apply_zscore_and_pool.py

Applies z-score normalization to raw linear_head score files (per dataset, per
SSL model) and pools them by SSL model.  Also z-scores TTS diversity score files.

Z-score parameters (mean, std) are computed from ALL scores (bonafide + spoof)
within each (dataset, stem) pair.  This equalizes cross-dataset score ranges
while preserving within-dataset rank ordering.

Steps
-----
  1  Compute (mean, std) per (dataset, stem) → linear_head_normalized_scores/zscore_stats.json
  2  Apply z-score to linear_head files → linear_head_normalized_scores/
  3  Pool z-scored files by SSL model   → normalized_scores_by_ssl_model/
  4  Apply same z-score to TTS files    → scores_by_TTS_norm/

TTS dataset name mapping (TTS score files → linear_head dataset stem):
  ASV19        → eval_2019
  ASV5         → asvspoof5
  FamousFigures → Famous_Figures
  MLAAD-En     → Multilingual  (MLAAD-En is the English subset; stats from full Multilingual)
  Spoof-Celeb  → spoofceleb

Usage
-----
    python3 scripts/apply_zscore_and_pool.py \\
        --linear_head_dir /data/ssl_anti_spoofing/asd_superb_score_files/linear_head \\
        --tts_dir         /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_TTS \\
        --out_base        /data/ssl_anti_spoofing/asd_superb_score_files
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

DATASETS = [
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

SSL_STEMS = [
    "fbank",
    "apc",
    "npc",
    "mockingjay",
    "tera",
    "decoar2",
    "wav2vec",
    "wav2vec2_base_960",
    "wav2vec2_large_ll60k",
    "hubert_base",
    "hubert_large_ll60k",
    "multires_hubert_multilingual_large600k",
    "xls_r_300m",
    "unispeech_sat_large",
    "data2vec_large_ll60k",
    "wavlablm_ek_40k",
    "wavlm_large",
    "ssast_frame_base",
    "mae_ast_frame",
]

# (dataset, stem) pairs with raw std below this threshold are degenerate:
# the model assigns near-identical scores to all utterances on that dataset,
# so z-scoring would amplify noise. These pairs are excluded from pooling.
MIN_STD = 0.05

# TTS score file dataset column → linear_head dataset directory name
TTS_DATASET_MAP = {
    "ASV19":         "eval_2019",
    "ASV5":          "asvspoof5",
    "FamousFigures": "Famous_Figures",
    "MLAAD-En":      "Multilingual",
    "Spoof-Celeb":   "spoofceleb",
}


# ---------------------------------------------------------------------------
# Step 1: compute z-score statistics
# ---------------------------------------------------------------------------
def compute_stats(linear_head_dir: str) -> dict:
    """
    Compute mean and std for each (stem, dataset) from raw linear_head files.
    Returns {stem: {dataset: {'mean': float, 'std': float | None, 'degenerate': bool}}}.
    Pairs with std < MIN_STD are marked degenerate and excluded from pooling.
    """
    stats: dict = {}
    total = len(SSL_STEMS) * len(DATASETS)
    done = 0

    for stem in SSL_STEMS:
        stats[stem] = {}
        for dataset in DATASETS:
            in_path = os.path.join(linear_head_dir, f"linear_head_{dataset}_{stem}.txt")
            if not os.path.exists(in_path):
                done += 1
                continue

            scores = []
            with open(in_path) as f:
                for line in f:
                    parts = line.rstrip("\n").split(" ")
                    if len(parts) < 4:
                        continue
                    try:
                        val = float(parts[3])
                        if math.isfinite(val):
                            scores.append(val)
                    except (ValueError, IndexError):
                        continue

            done += 1
            if len(scores) < 2:
                continue

            arr = np.array(scores, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr))

            if std < MIN_STD:
                stats[stem][dataset] = {"mean": mean, "std": None, "degenerate": True}
                print(f"  [DEGENERATE] ({stem}, {dataset})  μ={mean:+.4f}  σ={std:.4f} < MIN_STD={MIN_STD}")
            else:
                stats[stem][dataset] = {"mean": mean, "std": std}

            if done % 19 == 0:
                print(f"  [{done:>3}/{total}] {stem:<50} {dataset}")

    return stats


# ---------------------------------------------------------------------------
# Step 2: apply z-score to linear_head files
# ---------------------------------------------------------------------------
def apply_zscore_linear_head(
    linear_head_dir: str,
    norm_dir: str,
    stats: dict,
) -> None:
    os.makedirs(norm_dir, exist_ok=True)

    for stem in SSL_STEMS:
        stem_total = 0
        print(f"\n  {stem}")

        for dataset in DATASETS:
            in_path = os.path.join(linear_head_dir, f"linear_head_{dataset}_{stem}.txt")
            out_path = os.path.join(norm_dir, f"linear_head_{dataset}_{stem}.txt")

            if not os.path.exists(in_path):
                print(f"    [WARN] missing: linear_head_{dataset}_{stem}.txt")
                continue

            s = stats.get(stem, {}).get(dataset)
            if s is None:
                print(f"    [WARN] no stats for ({stem}, {dataset})")
                continue
            if s.get("degenerate"):
                print(f"    [SKIP DEGENERATE] ({stem}, {dataset})  excluded from pooling")
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            mean, std = s["mean"], s["std"]
            n = 0

            with open(in_path) as fin, open(out_path, "w") as fout:
                for line in fin:
                    parts = line.rstrip("\n").split(" ")
                    if len(parts) < 4:
                        continue
                    try:
                        z = (float(parts[3]) - mean) / std
                    except (ValueError, IndexError):
                        continue
                    parts[3] = repr(z)
                    fout.write(" ".join(parts) + "\n")
                    n += 1

            stem_total += n
            print(f"    {dataset:<35} {n:>10,}  (μ={mean:+.4f}, σ={std:.4f})")

        print(f"    {'TOTAL':<35} {stem_total:>10,}")


# ---------------------------------------------------------------------------
# Step 3: pool z-scored files by SSL model
# ---------------------------------------------------------------------------
def pool_by_model(norm_dir: str, combined_dir: str) -> None:
    os.makedirs(combined_dir, exist_ok=True)

    for stem in SSL_STEMS:
        combined_path = os.path.join(combined_dir, f"combined_{stem}.txt")
        stem_total = 0

        with open(combined_path, "w") as combined_out:
            for dataset in DATASETS:
                in_path = os.path.join(norm_dir, f"linear_head_{dataset}_{stem}.txt")
                if not os.path.exists(in_path):
                    continue
                n = 0
                with open(in_path) as fin:
                    for line in fin:
                        combined_out.write(line)
                        n += 1
                stem_total += n

        print(f"  {stem:<50} {stem_total:>12,}  → combined_{stem}.txt")


# ---------------------------------------------------------------------------
# Step 4: apply z-score to TTS score files
# ---------------------------------------------------------------------------
def apply_zscore_tts(tts_dir: str, tts_norm_dir: str, stats: dict) -> None:
    total_files = 0
    total_lines = 0

    for cat in ("AR", "NAR"):
        cat_in = os.path.join(tts_dir, cat)
        cat_out = os.path.join(tts_norm_dir, cat)
        if not os.path.isdir(cat_in):
            continue

        for system in sorted(os.listdir(cat_in)):
            sys_in = os.path.join(cat_in, system)
            if not os.path.isdir(sys_in):
                continue
            sys_out = os.path.join(cat_out, system)
            os.makedirs(sys_out, exist_ok=True)

            for fname in sorted(os.listdir(sys_in)):
                if not fname.endswith(".txt"):
                    continue

                fin_path = os.path.join(sys_in, fname)
                fout_path = os.path.join(sys_out, fname)
                stem = os.path.splitext(fname)[0]
                n = 0

                with open(fin_path) as fin, open(fout_path, "w") as fout:
                    fout.write(next(fin))          # preserve header
                    for line in fin:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 4:
                            tts_dataset = parts[0]
                            linear_dataset = TTS_DATASET_MAP.get(tts_dataset)
                            if linear_dataset is not None:
                                s = stats.get(stem, {}).get(linear_dataset)
                                if s is not None and not s.get("degenerate"):
                                    try:
                                        z = (float(parts[3]) - s["mean"]) / s["std"]
                                        parts[3] = repr(z)
                                    except ValueError:
                                        pass
                        fout.write("\t".join(parts) + "\n")
                        n += 1

                total_lines += n
                total_files += 1

    print(f"  {total_files} files processed, {total_lines:,} score lines written")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply z-score normalization to score files and pool by SSL model."
    )
    parser.add_argument("--linear_head_dir", required=True,
                        help="Raw linear_head_{dataset}_{stem}.txt files.")
    parser.add_argument("--tts_dir", required=True,
                        help="Root of scores_by_TTS directory (AR/ and NAR/).")
    parser.add_argument("--out_base", required=True,
                        help="Base output directory.")
    args = parser.parse_args()

    norm_dir     = os.path.join(args.out_base, "linear_head_normalized_scores")
    combined_dir = os.path.join(args.out_base, "normalized_scores_by_ssl_model")
    tts_norm_dir = os.path.join(args.out_base, "scores_by_TTS_norm")
    stats_path   = os.path.join(norm_dir, "zscore_stats.json")

    print("=" * 60)
    print("Step 1 — Computing z-score statistics")
    print("=" * 60)
    stats = compute_stats(args.linear_head_dir)
    os.makedirs(norm_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    print()
    print("=" * 60)
    print("Step 2 — Applying z-score to linear_head files")
    print("=" * 60)
    apply_zscore_linear_head(args.linear_head_dir, norm_dir, stats)

    print()
    print("=" * 60)
    print("Step 3 — Pooling z-scored files by SSL model")
    print("=" * 60)
    pool_by_model(norm_dir, combined_dir)

    print()
    print("=" * 60)
    print("Step 4 — Applying z-score to TTS score files")
    print("=" * 60)
    apply_zscore_tts(args.tts_dir, tts_norm_dir, stats)

    print()
    print("Output directories:")
    print(f"  {norm_dir}")
    print(f"  {combined_dir}")
    print(f"  {tts_norm_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
