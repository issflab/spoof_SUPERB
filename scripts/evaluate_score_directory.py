from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import calculate_EER


VALID_SCORE_SUFFIXES = {".txt", ".tsv", ".score", ""}
COMBINED_FILE_PREFIX = "combined"
EER_SUMMARY_FILENAME = "combined_ssl_model_eers.txt"

# Update this list to control which datasets are grouped into each combined file.
DATASET_NAMES = [
    "Famous_Figures",
    "Multilingual",
    "asvspoofLD",
    # "Noise_Addition",
    # "Reverberation",
    "asvspoof5",
    "asvspoof2021_DF",
    "asvspoof2021_LA",
    "deepfake_eval_2024",
    "eval_2019",
    "spoofceleb",
    "wild",
]


def iter_score_files(score_dir: Path) -> list[Path]:
    """Return supported non-hidden score files in sorted order."""
    score_paths: list[Path] = []
    for path in sorted(score_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix not in VALID_SCORE_SUFFIXES:
            continue
        score_paths.append(path)
    return score_paths


def extract_ssl_model_name(score_path: Path, dataset_names: list[str]) -> str | None:
    """Return the SSL model name as the part of the filename after a dataset name."""
    stem = score_path.stem
    for dataset_name in sorted(dataset_names, key=len, reverse=True):
        dataset_anchor = f"_{dataset_name}_"
        if dataset_anchor in stem:
            return stem.split(dataset_anchor, 1)[1]
    return None


def group_score_files_by_ssl_model(
    score_files: list[Path], dataset_names: list[str]
) -> tuple[dict[str, list[Path]], list[Path]]:
    """Group score files by SSL model and collect files that do not match any dataset."""
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    unmatched_paths: list[Path] = []

    for score_path in score_files:
        ssl_model_name = extract_ssl_model_name(score_path, dataset_names)
        if ssl_model_name is None:
            unmatched_paths.append(score_path)
            continue
        grouped_paths[ssl_model_name].append(score_path)

    return dict(grouped_paths), unmatched_paths


def combine_score_files(source_paths: list[Path], output_path: Path) -> None:
    """Concatenate multiple score files into one combined score file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for source_path in sorted(source_paths):
            content = source_path.read_text(encoding="utf-8").rstrip()
            if not content:
                continue
            output_handle.write(content)
            output_handle.write("\n")


def evaluate_combined_score_directory(
    score_dir: Path, output_dir: Path, dataset_names: list[str]
) -> None:
    """Combine score files by SSL model, compute EER, and write a summary file."""
    score_files = iter_score_files(score_dir)
    if not score_files:
        raise FileNotFoundError(f"No supported score files found in directory: {score_dir}")

    grouped_paths, unmatched_paths = group_score_files_by_ssl_model(score_files, dataset_names)
    if not grouped_paths:
        raise ValueError(
            "No score files matched the configured dataset names. "
            "Update DATASET_NAMES to match your filename patterns."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    eer_summary_path = output_dir / EER_SUMMARY_FILENAME

    with eer_summary_path.open("w", encoding="utf-8") as summary_handle:
        for ssl_model_name in sorted(grouped_paths):
            combined_score_path = output_dir / f"{COMBINED_FILE_PREFIX}_{ssl_model_name}.txt"
            combine_score_files(grouped_paths[ssl_model_name], combined_score_path)
            eer = round(calculate_EER(str(combined_score_path)), 2)
            summary_handle.write(f"{ssl_model_name} {eer:.2f}\n")
            print(
                f"[INFO] model={ssl_model_name}, files={len(grouped_paths[ssl_model_name])}, "
                f"EER={eer:.2f}, combined_file={combined_score_path.name}"
            )

    if unmatched_paths:
        print("[WARN] Skipped files that did not match any dataset name:")
        for unmatched_path in unmatched_paths:
            print(f"[WARN]   {unmatched_path.name}")

    print(f"[INFO] Saved EER summary: {eer_summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine score files for each SSL model using dataset-name anchors in the "
            "filename, then compute EER on each combined score file."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing the per-dataset score files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where combined score files and the EER summary will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Score directory not found: {args.input_dir}")

    evaluate_combined_score_directory(
        score_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_names=DATASET_NAMES,
    )


if __name__ == "__main__":
    main()


# python3 scripts/evaluate_score_directory.py \
#   --input_dir /data/ssl_anti_spoofing/asd_superb_score_files/linear_head \
#   --output_dir /tmp/linear_head_combined_scores
