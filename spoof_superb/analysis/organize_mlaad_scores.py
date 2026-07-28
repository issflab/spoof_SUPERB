#!/usr/bin/env python3
"""Split MLAAD v10 linear-head scores into per-system and per-language trees.

Reads the 22 per-SSL-model tsvs under ``linear_head_MLAAD_v10/tsv/`` and writes
two score trees:

    scores_by_TTS_MLAAD/<AR|NAR|closed_undisclosed>/<canonical_system>/<ssl>.txt
    scores_by_MLAAD_language/<language>/<ssl>.txt

Spoof scores only.  The M-AILABS bonafide set is shared by every group and is
written once at the root of each tree (``bonafide/<ssl>.txt``), never per
system — a single pooled bonafide reference is the whole point of restricting
the analysis to MLAAD.

The ``ar_nar`` bucket comes from ``mlaad_v10_tts_architecture_groups.csv``;
``unknown`` is written to ``closed_undisclosed`` so the directory tree matches
the figure labelling (these systems are undisclosed, not unmeasured).

Usage
-----
    python -m spoof_superb.analysis.organize_mlaad_scores
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from spoof_superb.config import cfg

SCORE_ROOT = Path(cfg.scores_root)
TSV_DIR = SCORE_ROOT / "linear_head_MLAAD_v10" / "tsv"
ARCH_CSV = SCORE_ROOT / "mlaad_v10_tts_architecture_groups.csv"
DIR_MAP = Path(__file__).resolve().parent / "mlaad_v10_dir_to_system.csv"

TTS_TREE = SCORE_ROOT / "scores_by_TTS_MLAAD"
LANG_TREE = SCORE_ROOT / "scores_by_MLAAD_language"

DATASET = "MLAAD-v10"
COLUMNS = ["dataset", "utt_id", "key", "score", "tts_system", "language", "ssl_model"]

AR_NAR_BUCKET = {"AR": "AR", "NAR": "NAR", "unknown": "closed_undisclosed"}

EXPECTED_RETAINED = 431_000
EXPECTED_BONAFIDE = 584_006


def ssl_name(tsv_path: Path) -> str:
    return tsv_path.stem.replace("linear_head_MLAAD_v10_", "")


def load_scores(tsv_path: Path) -> pd.DataFrame:
    """Read one tsv.  Tab separator is mandatory: utt_ids contain spaces."""
    df = pd.read_csv(tsv_path, sep="\t")
    if df["score"].isna().any():
        sys.exit(f"FATAL: NaN scores in {tsv_path}")
    return df


def annotate(df: pd.DataFrame, dir_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (spoof rows annotated with system/language/bucket, bonafide)."""
    bonafide = df[df["label"] == "bonafide"].copy()
    spoof = df[df["label"] == "spoof"].copy()

    parts = spoof["utt_id"].str.split("/", expand=True)
    spoof["language"] = parts[2]
    spoof["raw_dir"] = parts[3]

    merged = spoof.merge(dir_map, on="raw_dir", how="left", validate="many_to_one")
    if merged["canonical_system"].isna().any():
        missing = sorted(merged.loc[merged["canonical_system"].isna(), "raw_dir"].unique())
        sys.exit(f"FATAL: unmapped directories {missing}")
    return merged, bonafide


def write_frame(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, sep="\t", index=False, columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv-dir", type=Path, default=TSV_DIR)
    parser.add_argument("--dir-map", type=Path, default=DIR_MAP)
    parser.add_argument("--arch-csv", type=Path, default=ARCH_CSV)
    parser.add_argument("--tts-tree", type=Path, default=TTS_TREE)
    parser.add_argument("--lang-tree", type=Path, default=LANG_TREE)
    args = parser.parse_args()

    dir_map = pd.read_csv(args.dir_map)[["raw_dir", "canonical_system"]]
    arch = pd.read_csv(args.arch_csv)[["tts_system", "ar_nar"]]
    bucket_of = {
        r.tts_system: AR_NAR_BUCKET[r.ar_nar] for r in arch.itertuples(index=False)
    }

    tsvs = sorted(args.tsv_dir.glob("*.tsv"))
    print(f"SSL models: {len(tsvs)}")

    system_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    for tsv_path in tsvs:
        ssl = ssl_name(tsv_path)
        df = load_scores(tsv_path)
        spoof, bonafide = annotate(df, dir_map)

        excluded = spoof[spoof["canonical_system"] == "EXCLUDED"]
        retained = spoof[spoof["canonical_system"] != "EXCLUDED"].copy()
        retained["dataset"] = DATASET
        retained["key"] = "spoof"
        retained["tts_system"] = retained["canonical_system"]
        retained["ssl_model"] = ssl

        assert len(retained) == EXPECTED_RETAINED, (
            f"{ssl}: {len(retained)} retained spoof rows, expected {EXPECTED_RETAINED}"
        )
        assert len(bonafide) == EXPECTED_BONAFIDE, (
            f"{ssl}: {len(bonafide)} bonafide rows, expected {EXPECTED_BONAFIDE}"
        )

        bona_out = pd.DataFrame(
            {
                "dataset": DATASET,
                "utt_id": bonafide["utt_id"].to_numpy(),
                "key": "bonafide",
                "score": bonafide["score"].to_numpy(),
                "tts_system": "-",
                "language": "-",
                "ssl_model": ssl,
            }
        )
        write_frame(bona_out, args.tts_tree / "bonafide" / f"{ssl}.txt")
        write_frame(bona_out, args.lang_tree / "bonafide" / f"{ssl}.txt")

        written_sys = 0
        for system, group in retained.groupby("tts_system", sort=True):
            write_frame(group, args.tts_tree / bucket_of[system] / system / f"{ssl}.txt")
            written_sys += len(group)
            if ssl == ssl_name(tsvs[0]):
                system_counts[system] = len(group)

        written_lang = 0
        for language, group in retained.groupby("language", sort=True):
            write_frame(group, args.lang_tree / language / f"{ssl}.txt")
            written_lang += len(group)
            if ssl == ssl_name(tsvs[0]):
                language_counts[language] = len(group)

        assert written_sys == EXPECTED_RETAINED, f"{ssl}: system tree total {written_sys}"
        assert written_lang == EXPECTED_RETAINED, f"{ssl}: language tree total {written_lang}"
        print(
            f"  {ssl:<34} spoof={len(retained)} excluded={len(excluded)} "
            f"bonafide={len(bonafide)}"
        )

    assert sum(system_counts.values()) == EXPECTED_RETAINED
    assert sum(language_counts.values()) == EXPECTED_RETAINED

    low = {s: n for s, n in system_counts.items() if n < 100}
    print(f"\nSystems: {len(system_counts)}   Languages: {len(language_counts)}")
    print(f"Low-confidence systems (<100 spoof utts): {low or 'none'}")
    print("Support range: "
          f"min={min(system_counts.values())} ({min(system_counts, key=system_counts.get)}) "
          f"max={max(system_counts.values())} ({max(system_counts, key=system_counts.get)})")

    print(
        f"\n✅ STEP 2 — {args.tts_tree} + {args.lang_tree} — "
        f"{len(tsvs)} SSL models, {len(system_counts)} systems, "
        f"{len(language_counts)} languages, {EXPECTED_RETAINED} spoof + "
        f"{EXPECTED_BONAFIDE} bonafide utts per model"
    )


if __name__ == "__main__":
    main()
