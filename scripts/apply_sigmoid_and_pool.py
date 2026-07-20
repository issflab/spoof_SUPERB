"""
apply_sigmoid_and_pool.py

Applies sigmoid normalization to raw linear_head score files and pools them
by SSL model. Also normalizes TTS diversity score files.

Steps
-----
  1+2  sigmoid(score) applied to each raw score in linear_head files for the
       10 main benchmark datasets × 19 SSL models, then pooled by model.
       Outputs:
         linear_head_normalized_scores/linear_head_{dataset}_{stem}.txt
         normalized_scores_by_ssl_model/combined_{stem}.txt

  3    sigmoid applied to TTS diversity score files (score column only,
       preserving all other TSV columns).
       Output:
         scores_by_TTS_norm/{AR,NAR}/{system}/{file}.txt

Usage
-----
    python3 scripts/apply_sigmoid_and_pool.py \\
        --linear_head_dir /data/ssl_anti_spoofing/asd_superb_score_files/linear_head \\
        --tts_dir         /data/ssl_anti_spoofing/asd_superb_score_files/scores_by_TTS \\
        --out_base        /data/ssl_anti_spoofing/asd_superb_score_files
"""

import argparse
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 10 main benchmark datasets (linear_head filename prefix)
# ---------------------------------------------------------------------------
DATASETS = [
    "eval_2019",        # ASVspoof 2019 LA eval
    "asvspoof2021_LA",  # ASVspoof 2021 LA
    "asvspoof2021_DF",  # ASVspoof 2021 DF
    "asvspoof5",        # ASVspoof 5
    "Famous_Figures",   # Famous Figures
    "Multilingual",     # MLAAD + M-AILABS (all languages)
    "spoofceleb",       # SpoofCeleb
    "wild",             # In-the-Wild (ITW)
    "deepfake_eval_2024",  # DeepfakeEval 2024
    "asvspoofLD",       # ASVLD (noise + reverb + resampling combined)
]

# ---------------------------------------------------------------------------
# 19 benchmark SSL model stems
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Numerically stable sigmoid
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ---------------------------------------------------------------------------
# Step 1+2: sigmoid + pool linear_head files
# ---------------------------------------------------------------------------
def apply_sigmoid_and_pool(
    linear_head_dir: str,
    norm_dir: str,
    combined_dir: str,
) -> None:
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    for stem in SSL_STEMS:
        combined_path = os.path.join(combined_dir, f"combined_{stem}.txt")
        stem_total = 0
        print(f"\n  {stem}")

        with open(combined_path, "w") as combined_out:
            for dataset in DATASETS:
                in_path = os.path.join(
                    linear_head_dir, f"linear_head_{dataset}_{stem}.txt"
                )
                if not os.path.exists(in_path):
                    print(f"    [WARN] missing: linear_head_{dataset}_{stem}.txt")
                    continue

                out_path = os.path.join(
                    norm_dir, f"linear_head_{dataset}_{stem}.txt"
                )
                n = 0
                with open(in_path) as fin, open(out_path, "w") as fout:
                    for line in fin:
                        parts = line.rstrip("\n").split(" ")
                        if len(parts) < 4:
                            continue
                        try:
                            norm_score = sigmoid(float(parts[3]))
                        except (ValueError, IndexError):
                            continue
                        parts[3] = repr(norm_score)
                        out_line = " ".join(parts) + "\n"
                        fout.write(out_line)
                        combined_out.write(out_line)
                        n += 1

                stem_total += n
                print(f"    {dataset:<35} {n:>10,}")

        print(f"    {'TOTAL':<35} {stem_total:>10,}  → combined_{stem}.txt")


# ---------------------------------------------------------------------------
# Step 3: sigmoid on TTS diversity score files (TSV, score = column 3)
# ---------------------------------------------------------------------------
def apply_sigmoid_tts(tts_dir: str, tts_norm_dir: str) -> None:
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
                fin_path = os.path.join(sys_in, fname)
                fout_path = os.path.join(sys_out, fname)
                n = 0

                with open(fin_path) as fin, open(fout_path, "w") as fout:
                    fout.write(next(fin))          # preserve header unchanged
                    for line in fin:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 4:
                            try:
                                parts[3] = repr(sigmoid(float(parts[3])))
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
        description="Apply sigmoid normalization to score files and pool by SSL model."
    )
    parser.add_argument(
        "--linear_head_dir", required=True,
        help="Directory containing raw linear_head_<dataset>_<stem>.txt files.",
    )
    parser.add_argument(
        "--tts_dir", required=True,
        help="Root of scores_by_TTS directory (contains AR/ and NAR/).",
    )
    parser.add_argument(
        "--out_base", required=True,
        help="Base output directory. Three subdirs will be created here.",
    )
    args = parser.parse_args()

    norm_dir     = os.path.join(args.out_base, "linear_head_normalized_scores")
    combined_dir = os.path.join(args.out_base, "normalized_scores_by_ssl_model")
    tts_norm_dir = os.path.join(args.out_base, "scores_by_TTS_norm")

    print("=" * 60)
    print("Step 1+2 — sigmoid + pool linear_head files")
    print("=" * 60)
    apply_sigmoid_and_pool(args.linear_head_dir, norm_dir, combined_dir)

    print()
    print("=" * 60)
    print("Step 3 — sigmoid on TTS diversity score files")
    print("=" * 60)
    apply_sigmoid_tts(args.tts_dir, tts_norm_dir)

    print()
    print("Output directories:")
    print(f"  {norm_dir}")
    print(f"  {combined_dir}")
    print(f"  {tts_norm_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
