from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROTOCOL_COLUMNS = [
    "SPEAKER_ID",
    "FLAC_FILE_NAME",
    "SPEAKER_GENDER",
    "CODEC",
    "CODEC_Q",
    "CODEC_SEED",
    "ATTACK_TAG",
    "ATTACK_LABEL",
    "KEY",
    "TMP",
]

SCORE_COLUMNS = ["filename", "placeholder", "key", "score"]
VALID_SCORE_SUFFIXES = {".txt", ".tsv", ".score", ""}


def read_protocol(protocol_path: Path) -> pd.DataFrame:
    """Read a whitespace-separated protocol file with fixed columns."""
    print(f"[INFO] Reading protocol: {protocol_path}")
    try:
        df = pd.read_csv(
            protocol_path,
            sep=r"\s+",
            header=None,
            names=PROTOCOL_COLUMNS,
            dtype=str,
            engine="python",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read protocol file: {protocol_path}") from exc

    if df.shape[1] != len(PROTOCOL_COLUMNS):
        raise ValueError(
            f"Protocol file must have {len(PROTOCOL_COLUMNS)} columns, "
            f"but got {df.shape[1]}: {protocol_path}"
        )

    if df["FLAC_FILE_NAME"].isna().any():
        raise ValueError(f"Protocol file contains missing FLAC_FILE_NAME values: {protocol_path}")

    print(f"[INFO] Loaded {len(df)} protocol rows")
    return df


def get_filtered_filenames(protocol_df: pd.DataFrame, filter_mode: str) -> set[str]:
    """Return filenames matching the selected codec-related filter."""
    if filter_mode == "codec_q_zero":
        mask = protocol_df["CODEC_Q"] == "0"
        filter_desc = 'CODEC_Q == "0"'
    elif filter_mode == "codec_dash":
        mask = protocol_df["CODEC"] == "-"
        filter_desc = 'CODEC == "-"'
    else:
        raise ValueError(
            f"Unsupported filter_mode: {filter_mode}. "
            'Expected one of: "codec_q_zero", "codec_dash".'
        )

    filtered = set(protocol_df.loc[mask, "FLAC_FILE_NAME"].astype(str))
    print(f"[INFO] Filter mode: {filter_mode} ({filter_desc})")
    print(f"[INFO] Matched {len(filtered)} filenames from protocol")
    return filtered


def read_score_file(score_path: Path) -> pd.DataFrame:
    """Read a whitespace-separated score file with four columns."""
    print(f"[INFO] Reading score file: {score_path}")
    try:
        df = pd.read_csv(
            score_path,
            sep=r"\s+",
            header=None,
            names=SCORE_COLUMNS,
            dtype=str,
            engine="python",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read score file: {score_path}") from exc

    if df.shape[1] != len(SCORE_COLUMNS):
        raise ValueError(
            f"Score file must have {len(SCORE_COLUMNS)} columns, "
            f"but got {df.shape[1]}: {score_path}"
        )

    if df["filename"].isna().any():
        raise ValueError(f"Score file contains missing filename values: {score_path}")

    return df


def extract_suffix(filename: str, anchor: str) -> str | None:
    """Return the substring after an anchor in the basename, or None if the anchor is absent."""
    basename = Path(filename).name
    if anchor not in basename:
        return None
    return basename.split(anchor, 1)[1]


def extract_model_name(filename: str) -> str:
    """Return the lowercase model name from the full dir3 filename stem."""
    return Path(filename).stem.lower()


def extract_model_name_from_anchor(filename: str, anchor: str) -> str | None:
    """Return the lowercase model name after an anchor, or None if the anchor is absent."""
    suffix = extract_suffix(filename, anchor)
    if suffix is None:
        return None
    return Path(suffix).stem.lower()


def iter_score_files(score_dir: Path) -> Iterable[Path]:
    """Yield non-hidden regular files with supported score-file extensions."""
    for path in sorted(score_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix not in VALID_SCORE_SUFFIXES:
            print(f"[INFO] Skipping unsupported file extension: {path.name}")
            continue
        yield path


def build_suffix_map(score_dir: Path, anchor: str) -> dict[str, Path]:
    """
    Build a map from extracted suffix to file path.

    Raises an error if multiple files map to the same suffix in the same directory.
    """
    suffix_map: dict[str, Path] = {}
    skipped_without_anchor = 0

    for path in iter_score_files(score_dir):
        suffix = extract_suffix(path.name, anchor)
        if suffix is None:
            skipped_without_anchor += 1
            print(f"[INFO] Skipping file without anchor '{anchor}': {path.name}")
            continue

        if suffix in suffix_map:
            raise ValueError(
                f"Duplicate suffix mapping in {score_dir} for suffix '{suffix}' using anchor '{anchor}'. "
                f"Conflicting files: {suffix_map[suffix].name} and {path.name}"
            )
        suffix_map[suffix] = path

    print(
        f"[INFO] Built suffix map for {score_dir}: {len(suffix_map)} matched files, "
        f"{skipped_without_anchor} skipped without anchor '{anchor}'"
    )
    return suffix_map


def build_model_name_map(score_dir: Path) -> dict[str, Path]:
    """Build a case-insensitive map from model name to file path."""
    model_map: dict[str, Path] = {}

    for path in iter_score_files(score_dir):
        model_name = extract_model_name(path.name)
        if model_name in model_map:
            raise ValueError(
                f"Duplicate model mapping in {score_dir} for model '{model_name}'. "
                f"Conflicting files: {model_map[model_name].name} and {path.name}"
            )
        model_map[model_name] = path

    print(f"[INFO] Built model map for {score_dir}: {len(model_map)} matched files")
    return model_map


def warn_for_protocol_misses(
    score_df: pd.DataFrame,
    protocol_filenames: set[str],
    score_path: Path,
) -> None:
    """Warn when score filenames are not present in the protocol file."""
    score_names = set(score_df["filename"].astype(str))
    missing = sorted(score_names - protocol_filenames)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        print(
            f"[WARN] {score_path.name}: {len(missing)} filenames are not present in the "
            f"filtered protocol set. Examples: {preview}{suffix}"
        )


def filter_score_df(score_df: pd.DataFrame, allowed_filenames: set[str], score_path: Path) -> pd.DataFrame:
    """Filter score rows by filename membership in the allowed set."""
    # warn_for_protocol_misses(score_df, allowed_filenames, score_path)
    filtered_df = score_df[score_df["filename"].isin(allowed_filenames)].copy()
    print(
        f"[INFO] {score_path.name}: kept {len(filtered_df)} / {len(score_df)} rows after protocol filtering"
    )
    return filtered_df


def filter_score_df_by_prefix(score_df: pd.DataFrame, prefix: str, score_path: Path) -> pd.DataFrame:
    """Filter score rows by filename prefix."""
    filtered_df = score_df[score_df["filename"].str.startswith(prefix, na=False)].copy()
    print(
        f"[INFO] {score_path.name}: kept {len(filtered_df)} / {len(score_df)} rows with filename prefix '{prefix}'"
    )
    return filtered_df


def merge_score_files(
    dir1_score_df: pd.DataFrame,
    dir2_score_df: pd.DataFrame,
    join_type: str,
) -> pd.DataFrame:
    """Concatenate two score dataframes vertically using shared output columns."""
    _ = join_type
    left_df = dir1_score_df[["filename", "key", "score"]].copy()
    right_df = dir2_score_df[["filename", "key", "score"]].copy()
    merged_df = pd.concat([left_df, right_df], ignore_index=True)
    return merged_df


def build_output_path(output_dir: Path, score_path: Path) -> Path:
    """Create an output path that preserves the original score filename."""
    return output_dir / f"{score_path.stem}.txt"


def process_score_directories(
    score_dir1: Path,
    score_dir2: Path,
    score_dir3: Path,
    allowed_filenames: set[str],
    join_type: str,
    output_dir: Path,
    dir1_anchor: str,
    dir2_anchor: str,
) -> None:
    """
    Match files across three directories by anchor-derived suffix, then filter, concatenate, and save outputs.

    Files in dir1 are filtered using the protocol-derived filename set.
    Files in dir2 are matched to dir1 by exact suffix equality after removing their respective anchors.
    Files in dir3 are matched by the model name that appears after the dir1/dir2 anchor, case-insensitively.
    """
    dir2_suffix_map = build_suffix_map(score_dir2, dir2_anchor)
    dir3_model_map = build_model_name_map(score_dir3)
    dir1_paths = list(iter_score_files(score_dir1))

    # print(dir3_model_map)

    if not dir1_paths:
        print(f"[WARN] No score files found in first directory: {score_dir1}")
        return

    print(
        f"[INFO] Processing {len(dir1_paths)} files from {score_dir1}. "
        f"Multiple dir1 files may share a suffix; each file is handled independently."
    )

    for dir1_path in dir1_paths:
        suffix = extract_suffix(dir1_path.name, dir1_anchor)
        if suffix is None:
            print(f"[WARN] Skipping dir1 file without anchor '{dir1_anchor}': {dir1_path.name}")
            continue

        dir2_path = dir2_suffix_map.get(suffix)
        if dir2_path is None:
            print(
                f"[WARN] No matching dir2 file for {dir1_path.name}. "
                f"Expected suffix after '{dir2_anchor}': {suffix}"
            )
            continue
        model_name = extract_model_name_from_anchor(dir1_path.name, dir1_anchor)
        if model_name is None:
            print(
                f"[WARN] Could not extract model name from dir1 file {dir1_path.name} "
                f"using anchor '{dir1_anchor}'"
            )
            continue

        # print(model_name)
        dir3_path = dir3_model_map.get(model_name)
        if dir3_path is None:
            print(
                f"[WARN] No matching dir3 file for {dir1_path.name}. "
                f"Expected model name: {model_name}"
            )
            continue

        print(f"[INFO] Matched files: {dir1_path.name} <-> {dir2_path.name} <-> {dir3_path.name}")

        dir1_score_df = read_score_file(dir1_path)
        filtered_dir1_df = filter_score_df(dir1_score_df, allowed_filenames, dir1_path)
        if filtered_dir1_df.empty:
            print(f"[WARN] Filtered dir1 score file is empty: {dir1_path.name}")

        dir2_score_df = read_score_file(dir2_path)
        dir3_score_df = read_score_file(dir3_path)
        filtered_dir3_df = filter_score_df_by_prefix(dir3_score_df, "DF_E_", dir3_path)
        if filtered_dir3_df.empty:
            print(f"[WARN] Filtered dir3 score file is empty after DF_E_ prefix filter: {dir3_path.name}")

        merged_df = merge_score_files(filtered_dir1_df, dir2_score_df, join_type)
        merged_df = pd.concat([merged_df, filtered_dir3_df[["filename", "key", "score"]].copy()], ignore_index=True)

        if merged_df.empty:
            print(
                f"[WARN] Merge result is empty for matched files: "
                f"{dir1_path.name} <-> {dir2_path.name} <-> {dir3_path.name}"
            )

        output_path = build_output_path(output_dir, dir1_path)
        merged_df.to_csv(output_path, sep=" ", index=False)
        print(f"[INFO] Saved merged output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter score files from a first directory using a protocol file, "
            "match them to score files in a second directory by anchor-derived suffix, "
            "and save merged outputs."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True, help="Path to the protocol file")
    parser.add_argument(
        "--score_dir1",
        type=Path,
        required=True,
        help="First score directory; these score rows are filtered using the protocol",
    )
    parser.add_argument(
        "--score_dir2",
        type=Path,
        required=True,
        help="Second score directory; files are matched to score_dir1 using suffixes",
    )
    parser.add_argument(
        "--score_dir3",
        type=Path,
        required=True,
        help="Third score directory; files are matched to score_dir1 using suffixes and filtered by DF_E_ prefix",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where merged files will be saved",
    )
    parser.add_argument(
        "--filter_mode",
        choices=["codec_q_zero", "codec_dash"],
        default="codec_q_zero",
        help='Protocol filter rule. Default: "codec_q_zero"',
    )
    parser.add_argument(
        "--join_type",
        choices=["inner", "left"],
        default="inner",
        help='Merge join type. Default: "inner"',
    )
    parser.add_argument(
        "--dir1_anchor",
        type=str,
        default="asvspoof5_",
        help='Anchor used to extract model suffixes from score_dir1 filenames. Default: "asvspoof5_"',
    )
    parser.add_argument(
        "--dir2_anchor",
        type=str,
        default="eval_2019_",
        help='Anchor used to extract model suffixes from score_dir2 filenames. Default: "eval_2019_"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.protocol.is_file():
        raise FileNotFoundError(f"Protocol file not found: {args.protocol}")
    if not args.score_dir1.is_dir():
        raise NotADirectoryError(f"First score directory not found: {args.score_dir1}")
    if not args.score_dir2.is_dir():
        raise NotADirectoryError(f"Second score directory not found: {args.score_dir2}")
    if not args.score_dir3.is_dir():
        raise NotADirectoryError(f"Third score directory not found: {args.score_dir3}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol_df = read_protocol(args.protocol)
    filtered_filenames = get_filtered_filenames(protocol_df, args.filter_mode)
    process_score_directories(
        score_dir1=args.score_dir1,
        score_dir2=args.score_dir2,
        score_dir3=args.score_dir3,
        allowed_filenames=filtered_filenames,
        join_type=args.join_type,
        output_dir=args.output_dir,
        dir1_anchor=args.dir1_anchor,
        dir2_anchor=args.dir2_anchor,
    )


if __name__ == "__main__":
    main()

    # Example:
    # python3 merge_filtered_scores.py \
    #   --protocol /path/to/protocol.txt \
    #   --score_dir1 /path/to/asvspoof5_scores \
    #   --score_dir2 /path/to/eval2019_scores \
    #   --score_dir3 /path/to/third_scores \
    #   --output_dir /path/to/output_dir \
    #   --filter_mode codec_q_zero \
    #   --join_type inner
