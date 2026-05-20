from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import calculate_EER


VALID_SCORE_SUFFIXES = {".txt", ".tsv", ".score", ""}


def _normalize_delim(delim: str | None) -> str | None:
    if delim is None:
        return None
    delimiter_map = {
        r"\t": "\t",
        "\\t": "\t",
        "tab": "\t",
        r"\n": "\n",
        "\\n": "\n",
        "space": " ",
    }
    return delimiter_map.get(delim, delim)


def load_yaml_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def gen_score_file(
    protocol_file_path: Path,
    score_file_path: Path,
    config: dict,
    out_path: Path | None = None,
) -> Path:
    data_config = config["data_config"]
    protocol_delimiter = _normalize_delim(data_config.get("protocol_delimiter"))
    protocol_file_id_column = data_config["protocol_file_id_column"]
    protocol_label_column = data_config["protocol_label_column"]

    protocol_df = pd.read_csv(
        protocol_file_path,
        sep=protocol_delimiter,
        header=None,
        engine="python",
    )

    required_columns = max(protocol_file_id_column, protocol_label_column)
    if protocol_df.shape[1] <= required_columns:
        raise ValueError(
            "Protocol file does not contain the configured file-id/label columns: "
            f"expected indices up to {required_columns}, found {protocol_df.shape[1]} columns."
        )

    protocol_df = protocol_df[[protocol_file_id_column, protocol_label_column]].copy()
    protocol_df.columns = ["AUDIO_FILE_NAME", "KEY"]
    protocol_df["AUDIO_FILE_NAME"] = (
        protocol_df["AUDIO_FILE_NAME"].astype(str).str.split(".").str[0]
    )

    scores_df = pd.read_csv(
        score_file_path,
        sep=r"\s+",
        names=["AUDIO_FILE_NAME", "Scores"],
        engine="python",
    )
    scores_df["AUDIO_FILE_NAME"] = scores_df["AUDIO_FILE_NAME"].astype(str).str.split(".").str[0]

    merged_df = pd.merge(protocol_df, scores_df, on="AUDIO_FILE_NAME", how="inner")
    score_df = merged_df[["AUDIO_FILE_NAME", "KEY", "Scores"]]

    if out_path is None:
        out_path = score_file_path.with_name(f"{score_file_path.stem}-labels.txt")

    score_df.to_csv(out_path, sep=" ", header=None, index=False)
    return out_path


def iter_score_files(score_dir: Path) -> list[Path]:
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


def extract_model_name_from_dataset(score_path: Path, dataset_name: str) -> str | None:
    dataset_anchor = f"_{dataset_name}_"
    stem = score_path.stem
    if dataset_anchor not in stem:
        return None
    return stem.split(dataset_anchor, 1)[1]


def evaluate_single_score_file(
    score_file_path: Path,
    score_file_has_keys: bool,
    config: dict | None,
    protocol_file_path: Path | None,
) -> float:
    if score_file_has_keys:
        return calculate_EER(str(score_file_path))

    if config is None or protocol_file_path is None:
        raise ValueError(
            "--config and --protocol_filepath are required when --score_file_has_keys is not set."
        )

    labeled_score_path = gen_score_file(
        protocol_file_path=protocol_file_path,
        score_file_path=score_file_path,
        config=config,
    )
    return calculate_EER(str(labeled_score_path))


def evaluate_dataset_in_directory(
    score_dir: Path,
    dataset_name: str,
    output_file: Path | None,
    score_file_has_keys: bool,
    config: dict | None,
    protocol_file_path: Path | None,
) -> None:
    matching_paths: list[tuple[str, Path]] = []

    for score_path in iter_score_files(score_dir):
        model_name = extract_model_name_from_dataset(score_path, dataset_name)
        if model_name is None:
            continue
        matching_paths.append((model_name, score_path))

    if not matching_paths:
        raise FileNotFoundError(
            f"No score files found for dataset '{dataset_name}' in directory: {score_dir}"
        )

    lines: list[str] = []
    for model_name, score_path in matching_paths:
        eer = evaluate_single_score_file(
            score_file_path=score_path,
            score_file_has_keys=score_file_has_keys,
            config=config,
            protocol_file_path=protocol_file_path,
        )
        lines.append(f"{model_name} {eer:.2f}")
        print(f"[INFO] dataset={dataset_name}, model={model_name}, EER={eer:.2f}")

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] Saved EER summary: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute EER for one score file, or compute EER for each SSL model "
            "for one dataset name inside a score directory."
        )
    )
    parser.add_argument(
        "--score_file_has_keys",
        action="store_true",
        help="Set this when the score file already contains labels/keys.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML config file used to label raw score files.",
    )
    parser.add_argument(
        "--protocol_filepath",
        type=Path,
        help="Path to the protocol file used to label raw score files.",
    )
    parser.add_argument(
        "--score_filepath",
        type=Path,
        help="Path to one score file to evaluate.",
    )
    parser.add_argument(
        "--score_dir",
        type=Path,
        help="Directory containing score files for multiple datasets and SSL models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name anchor used inside score filenames, for example asvspoofLD.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Optional output summary file for dataset-level evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    has_single_file_mode = args.score_filepath is not None
    has_dataset_mode = args.score_dir is not None or args.dataset_name is not None

    if has_single_file_mode and has_dataset_mode:
        raise ValueError("Use either --score_filepath or (--score_dir with --dataset_name), not both.")

    if not has_single_file_mode and not has_dataset_mode:
        raise ValueError(
            "Provide --score_filepath for single-file evaluation, or provide both "
            "--score_dir and --dataset_name for dataset-level evaluation."
        )

    config = None if args.score_file_has_keys else load_yaml_config(args.config)

    if has_single_file_mode:
        eer = evaluate_single_score_file(
            score_file_path=args.score_filepath.resolve(),
            score_file_has_keys=args.score_file_has_keys,
            config=config,
            protocol_file_path=args.protocol_filepath.resolve() if args.protocol_filepath else None,
        )
        print(f"   EER            = {eer:8.5f} % (Equal error rate for countermeasure)")
        return

    if args.score_dir is None or args.dataset_name is None:
        raise ValueError("Dataset-level evaluation requires both --score_dir and --dataset_name.")

    if not args.score_dir.is_dir():
        raise NotADirectoryError(f"Score directory not found: {args.score_dir}")

    evaluate_dataset_in_directory(
        score_dir=args.score_dir.resolve(),
        dataset_name=args.dataset_name,
        output_file=args.output_file.resolve() if args.output_file else None,
        score_file_has_keys=args.score_file_has_keys,
        config=config,
        protocol_file_path=args.protocol_filepath.resolve() if args.protocol_filepath else None,
    )


if __name__ == "__main__":
    main()


# Single file:
# python3 scripts/evaluate_score_file.py \
#   --score_filepath /path/to/score.txt \
#   --score_file_has_keys
#
# One dataset inside a score directory:
# python3 scripts/evaluate_score_file.py \
#   --score_dir /data/ssl_anti_spoofing/asd_superb_score_files/linear_head \
#   --dataset_name asvspoofLD \
#   --score_file_has_keys \
#   --output_file /tmp/asvspoofLD_eers.txt
