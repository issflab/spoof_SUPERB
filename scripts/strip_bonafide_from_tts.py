"""
strip_bonafide_from_tts.py

Removes bonafide rows from TTS score files (both raw and normalized).
These files should contain only spoof utterances from each TTS system.
Bonafide rows appeared because the source datasets included all utterances.

Edits files in-place. Reports lines removed per system.

Usage
-----
    python3 scripts/strip_bonafide_from_tts.py --base_dir /data/ssl_anti_spoofing/asd_superb_score_files
"""

import argparse
import os
import tempfile


def strip_file(path: str) -> tuple[int, int]:
    """
    Rewrite path keeping only the header and spoof rows.
    Returns (kept, removed) line counts (header not counted).
    """
    kept = removed = 0
    dir_ = os.path.dirname(path)

    with open(path, "r") as fin:
        header = fin.readline()
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tmp:
            tmp.write(header)
            for line in fin:
                parts = line.split("\t")
                if len(parts) >= 3 and parts[2] == "bonafide":
                    removed += 1
                else:
                    tmp.write(line)
                    kept += 1
            tmp_path = tmp.name

    os.replace(tmp_path, path)
    return kept, removed


def process_dir(tts_dir: str, label: str) -> None:
    total_kept = total_removed = 0
    systems_affected = []

    for cat in ("AR", "NAR"):
        cat_dir = os.path.join(tts_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        for system in sorted(os.listdir(cat_dir)):
            sys_dir = os.path.join(cat_dir, system)
            if not os.path.isdir(sys_dir):
                continue

            sys_kept = sys_removed = 0
            for fname in sorted(os.listdir(sys_dir)):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(sys_dir, fname)
                k, r = strip_file(fpath)
                sys_kept += k
                sys_removed += r

            if sys_removed > 0:
                systems_affected.append(f"  [{cat}] {system}: removed {sys_removed:,} bonafide, kept {sys_kept:,} spoof")
                total_removed += sys_removed
                total_kept += sys_kept

    print(f"\n{label}")
    print(f"  Path: {tts_dir}")
    for s in systems_affected:
        print(s)
    print(f"  TOTAL: removed {total_removed:,} bonafide rows, kept {total_kept:,} spoof rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    args = parser.parse_args()

    raw_dir  = os.path.join(args.base_dir, "scores_by_TTS")
    norm_dir = os.path.join(args.base_dir, "scores_by_TTS_norm")

    print("Stripping bonafide rows from TTS score files...")

    process_dir(raw_dir,  "=== scores_by_TTS (raw) ===")
    process_dir(norm_dir, "=== scores_by_TTS_norm (z-scored) ===")

    print("\nDone.")


if __name__ == "__main__":
    main()
