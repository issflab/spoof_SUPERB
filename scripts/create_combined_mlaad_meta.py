#!/usr/bin/env python3
"""Combine MLAAD fake/en meta.csv files into a single text file.

The output file contains three pipe-delimited columns:
filename|absolute_path|model_name
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-model MLAAD meta.csv files into one text file."
    )
    parser.add_argument(
        "--meta-root",
        type=Path,
        default=Path("/data/Data/MLAAD/fake/en"),
        help="Directory containing one folder per TTS system.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/Data/MLAAD"),
        help="Root used to resolve relative paths from the meta.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/Data/MLAAD/fake/en/combined_meta.txt"),
        help="Path to the output text file.",
    )
    return parser.parse_args()


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


def collect_rows(meta_root: Path, dataset_root: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    tts_directories = sorted(path for path in meta_root.iterdir() if path.is_dir())

    for tts_dir in tts_directories:
        meta_path = tts_dir / "meta.csv"
        if not meta_path.is_file():
            continue

        with meta_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            for row in reader:
                raw_path = (row.get("path") or "").strip()
                if not raw_path:
                    continue

                model_name = normalize_model_name(
                    raw_model_name=row.get("model_name") or "",
                    fallback_name=meta_path.parent.name,
                )
                absolute_path = resolve_audio_path(
                    raw_path=raw_path,
                    dataset_root=dataset_root,
                    meta_dir=meta_path.parent,
                )

                rows.append((absolute_path.name, str(absolute_path), model_name))

    return rows


def write_output(rows: list[tuple[str, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["filename", "absolute_path", "model_name"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = collect_rows(
        meta_root=args.meta_root.resolve(),
        dataset_root=args.dataset_root.resolve(),
    )
    write_output(rows=rows, output_path=args.output.resolve())
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
