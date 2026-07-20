"""
Append ASV21 LA:C1 (no-codec) scores into each Hashim baseline file.

The Hashim baseline files currently contain ASV19 eval + ASV21 DF:C1 + ASV5
but are missing ASV21 LA:C1.  This script:

  1. Loads the ASV21 LA:C1 utterance ID set (codec == "none") from the
     trial_metadata.txt protocol.
  2. For each Hashim baseline file, finds the matching ASV21 LA full-eval
     score file, filters it to C1-only rows, and appends them.

Usage
-----
    python3 scripts/merge_asv21la_into_hashim_baseline.py \
        --hashim_dir   /data/ssl_anti_spoofing/asd_superb_score_files/Baseline_by_Hashim/ \
        --la_dir       /data/ssl_anti_spoofing/asd_superb_score_files/linear_head/ \
        --asv21_la_protocol /data/Data/ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/trial_metadata.txt \
        [--dry_run]
"""

import argparse
import os
from typing import Dict, List, Set, Tuple

Row = Tuple[str, str, float]

# Stem aliases: Hashim ASV5 stem -> ASV21 LA stem (only where they differ)
STEM_ALIASES: Dict[str, str] = {
    "byol_audio": "byol_a_2048",
    "mr_hubert":  "multires_hubert_multilingual_large600k",
}


def load_asv21_la_c1_ids(protocol_path: str) -> Set[str]:
    ids: Set[str] = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "none":
                ids.add(parts[1])
    return ids


def parse_score_file(path: str) -> List[Row]:
    rows: List[Row] = []
    header_tokens = {"filename", "utt_id", "uttid", "key", "score", "label"}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].lower() in header_tokens:
                continue
            if len(parts) == 4:
                utt_id, label, score_str = parts[0], parts[2], parts[3]
            elif len(parts) == 3:
                utt_id, label, score_str = parts[0], parts[1], parts[2]
            else:
                continue
            for ext in (".flac", ".wav"):
                if utt_id.endswith(ext):
                    utt_id = utt_id[: -len(ext)]
            try:
                rows.append((utt_id, label, float(score_str)))
            except ValueError:
                continue
    return rows


def append_rows(path: str, rows: List[Row]) -> None:
    with open(path, "a") as f:
        for utt_id, label, score in rows:
            f.write(f"{utt_id} - {label} {score}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append ASV21 LA:C1 rows to Hashim baseline files."
    )
    parser.add_argument("--hashim_dir", required=True,
                        help="Directory with Hashim ASV5 baseline .txt files.")
    parser.add_argument("--la_dir", required=True,
                        help="Directory with linear_head_asvspoof2021_LA_*.txt files.")
    parser.add_argument("--asv21_la_protocol", required=True,
                        help="ASVspoof2021 LA eval trial_metadata.txt")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without modifying any files.")
    args = parser.parse_args()

    print("Loading ASV21 LA:C1 IDs ...")
    c1_ids = load_asv21_la_c1_ids(args.asv21_la_protocol)
    print(f"  {len(c1_ids)} utterances with codec=none")

    hashim_prefix = "linear_head_asvspoof5_"
    la_prefix     = "linear_head_asvspoof2021_LA_"

    for fname in sorted(os.listdir(args.hashim_dir)):
        if not fname.endswith(".txt"):
            continue
        if not fname.startswith(hashim_prefix):
            print(f"  [SKIP] unexpected filename pattern: {fname}")
            continue

        stem = fname[len(hashim_prefix):-len(".txt")]
        la_stem = STEM_ALIASES.get(stem, stem)
        la_fname = f"{la_prefix}{la_stem}.txt"
        la_path = os.path.join(args.la_dir, la_fname)

        if not os.path.isfile(la_path):
            print(f"  [SKIP] no ASV21 LA file for '{stem}' "
                  f"(looked for {la_fname})")
            continue

        # Filter ASV21 LA file to C1-only rows
        la_rows = parse_score_file(la_path)
        c1_rows = [r for r in la_rows if r[0] in c1_ids]

        hashim_path = os.path.join(args.hashim_dir, fname)

        # Sanity check: make sure these IDs are not already in the baseline
        existing_rows = parse_score_file(hashim_path)
        existing_ids  = {r[0] for r in existing_rows}
        already_present = sum(1 for r in c1_rows if r[0] in existing_ids)

        if already_present > 0:
            print(f"  [WARN] {fname}: {already_present}/{len(c1_rows)} "
                  f"ASV21 LA:C1 IDs already present — skipping to avoid duplicates")
            continue

        if args.dry_run:
            print(f"  [DRY]  {fname}  +  {len(c1_rows)} ASV21 LA:C1 rows "
                  f"(from {la_fname})")
        else:
            append_rows(hashim_path, c1_rows)
            print(f"  [OK]   {fname}  +  {len(c1_rows)} rows appended "
                  f"(from {la_fname})")


if __name__ == "__main__":
    main()
