"""
Verify the composition of augmented degraded condition score files.

Each augmented condition file must contain two kinds of rows:
  1. "Native" rows  — the original degraded-condition utterances
                      (ASVLD-suffixed IDs, Codec C2-C9 IDs, Channel C2-C7 IDs, etc.)
  2. "Protocol" rows — clean utterances appended from the baseline
                      (IDs that appear in the 4 standard protocol ID sets)

CONDITION_COMPOSITION defines the exact expected counts for both kinds.
Run this script on the OUTPUT of verify_and_split_condition_scores.py --output_dir.

Usage
-----
    python3 scripts/check_condition_composition.py \\
        --asv19_protocol   /data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \\
        --asv21_la_protocol /data/Data/ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/trial_metadata.txt \\
        --asv21_df_protocol /data/Data/ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/trial_metadata.txt \\
        --asv5_protocol    /data/Data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv \\
        --condition_dirs   /path/to/augmented/Additive_Noise \\
                           /path/to/augmented/Reverberation \\
                           /path/to/augmented/Resampling \\
                           /path/to/augmented/Codec_Compression \\
                           /path/to/augmented/Channel_Distortions
"""

import argparse
import os
from typing import Dict, List, Set, Tuple

Row = Tuple[str, str, float]


# ---------------------------------------------------------------------------
# Expected composition for each condition (AUGMENTED files)
#
# protocol_sources : {source_name: expected_row_count}
#     Rows whose utt_id is a member of that source's protocol ID set.
#     Any source NOT listed is expected to have 0 matching rows.
#
# other_la_e / other_df_e / other_e : int
#     Rows whose utt_id starts with that prefix but is NOT in any protocol
#     source ID set.  These are the native degraded-condition utterances:
#       Additive_Noise / Reverberation / Resampling : ASVLD-suffixed IDs
#                                                      e.g. LA_E_xxx_babble_10
#       Codec_Compression : ASV21 DF:C2-C9 + ASV5:C01-C10 bare IDs
#       Channel_Distortions : ASV21 LA:C2-C7 bare IDs + ASV5:C11 bare IDs
# ---------------------------------------------------------------------------
CONDITION_COMPOSITION: Dict[str, Dict] = {
    "Additive_Noise": {
        "protocol_sources": {
            "asv21_la_c1": 25938,
            "asv21_df_c1": 17131,
            "asv5_noc":    171602,
        },
        "other_la_e": 712370,   # 10 noise types × 71 237 ASVLD utterances
        "other_df_e": 0,
        "other_e":    0,
    },
    "Reverberation": {
        "protocol_sources": {
            "asv21_la_c1": 25938,
            "asv21_df_c1": 17131,
            "asv5_noc":    171602,
        },
        "other_la_e": 210191,   # ASVLD reverb utterances
        "other_df_e": 0,
        "other_e":    0,
    },
    "Resampling": {
        "protocol_sources": {
            "asv21_la_c1": 25938,
            "asv21_df_c1": 17131,
            "asv5_noc":    171602,
        },
        "other_la_e": 284948,   # 4 resample rates × 71 237 ASVLD utterances
        "other_df_e": 0,
        "other_e":    0,
    },
    "Codec_Compression": {
        "protocol_sources": {
            "asv21_la_c1": 25938,   # added by augmentation
            # asv21_df_c1 = 0: C1 (nocodec) rows are excluded before augmentation
            # because the raw file was scored on the full DF eval set and contains
            # clean C1 rows that don't belong in a codec-degradation condition.
        },
        "other_la_e": 0,
        "other_df_e": 135824,   # DF:C2-C9 only (152955 raw minus 17131 C1 excluded)
        "other_e":    462562,   # ASV5:C01-C10 (codec-compressed substitutes)
    },
    "Channel_Distortions": {
        "protocol_sources": {
            "asv19_eval":  71237,   # ASVLD clean, appended from baseline
            "asv21_df_c1": 17131,   # clean DF:C1, appended from baseline
        },
        # asv21_la_c1 should be 0: C1 is excluded; C2-C7 are the channel substitutes
        # asv5_noc should be 0: C00 excluded; C11 is the channel substitute
        "other_la_e": 155628,   # ASV21 LA:C2-C7  (= 181 566 total − 25 938 C1 excluded)
        "other_df_e": 0,
        "other_e":    46610,    # ASV5:C11
    },
}


# ---------------------------------------------------------------------------
# Protocol loaders
# ---------------------------------------------------------------------------

def _load_col(path: str, col: int) -> Set[str]:
    ids: Set[str] = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > col:
                ids.add(parts[col])
    return ids


def load_protocols(
    asv19_protocol: str,
    asv21_la_protocol: str,
    asv21_df_protocol: str,
    asv5_protocol: str,
) -> Dict[str, Set[str]]:
    sources: Dict[str, Set[str]] = {}

    ids = set()
    with open(asv19_protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])
    sources["asv19_eval"] = ids
    print(f"  asv19_eval   : {len(ids):>7}")

    ids = set()
    with open(asv21_la_protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "none":
                ids.add(parts[1])
    sources["asv21_la_c1"] = ids
    print(f"  asv21_la_c1  : {len(ids):>7}")

    ids = set()
    with open(asv21_df_protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "nocodec":
                ids.add(parts[1])
    sources["asv21_df_c1"] = ids
    print(f"  asv21_df_c1  : {len(ids):>7} (full nocodec set)")

    ids = set()
    with open(asv5_protocol) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[3] == "-":
                ids.add(parts[1])
    sources["asv5_noc"] = ids
    print(f"  asv5_noc     : {len(ids):>7}")

    return sources


# ---------------------------------------------------------------------------
# Score file parser
# ---------------------------------------------------------------------------

def parse_score_file(path: str) -> List[Row]:
    header_tokens = {"filename", "utt_id", "uttid", "key", "score", "label"}
    rows: List[Row] = []
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


# ---------------------------------------------------------------------------
# Composition verifier
# ---------------------------------------------------------------------------

def verify_composition(
    score_path: str,
    expected: Dict,
    source_ids: Dict[str, Set[str]],
) -> bool:
    rows = parse_score_file(score_path)
    utt_ids = [r[0] for r in rows]
    id_set = set(utt_ids)

    # Count membership in each protocol source
    src_counts: Dict[str, int] = {}
    protocol_matched: Set[str] = set()
    for src, ids in source_ids.items():
        matched = ids & id_set
        src_counts[src] = len(matched)
        protocol_matched |= matched

    # Count "other" rows by prefix (not in any protocol set)
    other_la_e = sum(1 for uid in utt_ids if uid.startswith("LA_E_") and uid not in protocol_matched)
    other_df_e = sum(1 for uid in utt_ids if uid.startswith("DF_E_") and uid not in protocol_matched)
    other_e    = sum(1 for uid in utt_ids if uid.startswith("E_")    and uid not in protocol_matched)

    fname = os.path.basename(score_path)
    exp_proto  = expected.get("protocol_sources", {})
    exp_la_e   = expected.get("other_la_e", 0)
    exp_df_e   = expected.get("other_df_e", 0)
    exp_e      = expected.get("other_e",    0)

    errors: List[str] = []

    # Check protocol sources
    all_protocol_sources = set(source_ids.keys())
    for src in all_protocol_sources:
        exp_count = exp_proto.get(src, 0)
        got_count = src_counts.get(src, 0)
        if got_count != exp_count:
            errors.append(f"{src}: expected {exp_count}, got {got_count}")

    # Check native rows
    if other_la_e != exp_la_e:
        errors.append(f"other_la_e: expected {exp_la_e}, got {other_la_e}")
    if other_df_e != exp_df_e:
        errors.append(f"other_df_e: expected {exp_df_e}, got {other_df_e}")
    if other_e != exp_e:
        errors.append(f"other_e: expected {exp_e}, got {other_e}")

    total_exp = sum(exp_proto.values()) + exp_la_e + exp_df_e + exp_e
    total_got = len(rows)

    if errors:
        print(f"  [FAIL] {fname}: {total_got} rows (expected {total_exp})")
        for e in errors:
            print(f"         {e}")
        return False
    else:
        proto_summary = ", ".join(f"{k}={v}" for k, v in exp_proto.items())
        print(f"  [OK]   {fname}: {total_got} rows  "
              f"native(la_e={other_la_e}, df_e={other_df_e}, e={other_e})  "
              f"protocol({proto_summary})")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify composition of augmented degraded condition score files."
    )
    parser.add_argument("--asv19_protocol",    required=True)
    parser.add_argument("--asv21_la_protocol", required=True)
    parser.add_argument("--asv21_df_protocol", required=True)
    parser.add_argument("--asv5_protocol",     required=True)
    parser.add_argument(
        "--condition_dirs", nargs="+", required=True,
        help="Directories of augmented condition score files to verify.",
    )
    args = parser.parse_args()

    print("Loading protocol ID sets ...")
    source_ids = load_protocols(
        args.asv19_protocol,
        args.asv21_la_protocol,
        args.asv21_df_protocol,
        args.asv5_protocol,
    )

    total_files = 0
    total_ok    = 0

    for cond_dir in args.condition_dirs:
        cond_name = os.path.basename(cond_dir.rstrip("/"))
        expected  = CONDITION_COMPOSITION.get(cond_name)
        print(f"\n[{cond_name}]  {cond_dir}")
        if expected is None:
            print(f"  [WARN] no expected composition defined for '{cond_name}' — skipping")
            continue

        fnames = sorted(f for f in os.listdir(cond_dir) if f.endswith(".txt"))
        for fname in fnames:
            fpath = os.path.join(cond_dir, fname)
            ok = verify_composition(fpath, expected, source_ids)
            total_files += 1
            total_ok    += int(ok)

    print(f"\n{'='*60}")
    print(f"Result: {total_ok}/{total_files} files passed composition check.")
    if total_ok < total_files:
        print("  Run verify_and_split_condition_scores.py --output_dir to fix failing files.")


if __name__ == "__main__":
    main()
