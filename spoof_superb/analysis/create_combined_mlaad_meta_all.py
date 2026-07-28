#!/usr/bin/env python3
"""Combine every MLAAD fake/<language>/<system>/meta.csv into one text file.

Modelled on ``create_combined_mlaad_meta.py``, which covers ``fake/en`` only.
This variant walks all 54 language directories and adds a ``language`` column.

The language is taken from the meta.csv ``language`` column, not from the
directory path; any row where the two disagree is reported.

Output: ``/data/Data/MLAAD/combined_meta_all.txt``, pipe-delimited, header
``filename|absolute_path|model_name|language``.

Usage
-----
    python -m spoof_superb.analysis.create_combined_mlaad_meta_all
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from spoof_superb.config import cfg

# Some MLAAD transcripts exceed the 128 KiB default field limit.
csv.field_size_limit(sys.maxsize)

# MLAAD meta.csv is unquoted pipe-delimited, but transcripts contain bare `"`
# characters.  With csv's default quotechar those quotes swallow the following
# newlines and several physical lines collapse into one record — ja/kokoro
# alone loses 53 of its 1000 rows.  QUOTE_NONE is required for a faithful read.
READER_OPTS = {"delimiter": "|", "quoting": csv.QUOTE_NONE}

# Spoof utterances in linear_head_MLAAD_v10 — one meta row is expected per file.
EXPECTED_ROWS = 456_000


def resolve_audio_path(raw_path: str, dataset_root: Path, meta_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    cleaned = raw_path[2:] if raw_path.startswith("./") else raw_path
    candidate = dataset_root / cleaned
    if candidate.exists():
        return candidate.resolve()

    return (meta_dir / raw_path).resolve()


def normalize_model_name(raw_model_name: str, fallback_name: str) -> str:
    model_name = raw_model_name.strip() or fallback_name
    return model_name.rsplit("/", 1)[-1]


def collect_rows(
    fake_root: Path, dataset_root: Path
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str]]]:
    rows: list[tuple[str, str, str, str]] = []
    mismatches: list[tuple[str, str, str]] = []

    for meta_path in sorted(fake_root.glob("*/*/meta.csv")):
        path_language = meta_path.parent.parent.name

        with meta_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, **READER_OPTS)
            for row in reader:
                raw_path = (row.get("path") or "").strip()
                if not raw_path:
                    continue

                meta_language = (row.get("language") or "").strip() or path_language
                if meta_language != path_language:
                    mismatches.append((raw_path, path_language, meta_language))

                model_name = normalize_model_name(
                    raw_model_name=row.get("model_name") or "",
                    fallback_name=meta_path.parent.name,
                )
                absolute_path = resolve_audio_path(
                    raw_path=raw_path,
                    dataset_root=dataset_root,
                    meta_dir=meta_path.parent,
                )
                rows.append(
                    (absolute_path.name, str(absolute_path), model_name, meta_language)
                )

    return rows, mismatches


def write_output(rows: list[tuple[str, str, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["filename", "absolute_path", "model_name", "language"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-root", type=Path, default=Path(f"{cfg.data_root}/MLAAD/fake"))
    parser.add_argument("--dataset-root", type=Path, default=Path(f"{cfg.data_root}/MLAAD"))
    parser.add_argument(
        "--output", type=Path, default=Path(f"{cfg.data_root}/MLAAD/combined_meta_all.txt")
    )
    args = parser.parse_args()

    rows, mismatches = collect_rows(
        fake_root=args.fake_root.resolve(), dataset_root=args.dataset_root.resolve()
    )
    write_output(rows=rows, output_path=args.output.resolve())

    languages = Counter(r[3] for r in rows)
    print(f"Rows: {len(rows)}   languages: {len(languages)}")

    # One meta row per synthesized utterance; the scores tsvs carry 456,000
    # spoof rows, so any shortfall means meta.csv records were lost on read.
    if len(rows) != EXPECTED_ROWS:
        print(
            f"[WARN] {len(rows)} rows != {EXPECTED_ROWS} spoof utterances in the "
            "MLAAD v10 score files — meta.csv coverage is incomplete."
        )

    if mismatches:
        print(f"[WARN] {len(mismatches)} rows where meta.language != path language:")
        for raw_path, path_lang, meta_lang in mismatches[:20]:
            print(f"  {raw_path}: path={path_lang} meta={meta_lang}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("No language disagreements between meta.csv and the directory path.")

    print(
        f"\n✅ STEP 3 — {args.output} — "
        f"{len(rows)} rows, {len(languages)} languages, "
        f"{len({r[2] for r in rows})} distinct model_name values"
    )


if __name__ == "__main__":
    main()
