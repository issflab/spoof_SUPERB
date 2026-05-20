#!/usr/bin/env python3
"""
Organize SSL score files by canonical TTS system across multiple spoofing datasets.

This script uses:
1. A master TTS CSV containing canonical names and metadata such as AR/NAR labels.
2. A dataset lookup CSV describing how each TTS system appears in each dataset protocol.

For each canonical TTS system, the script:
- collects utterance IDs from each dataset protocol using the lookup rules
- filters matching SSL score files by those utterances
- concatenates filtered rows across datasets per SSL model
- saves outputs under output_root/<AR|NAR|UNKNOWN>/<tts_normalized>/<ssl_model>.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_LOOKUP_COLUMNS = {
    "tts_normalized",
    "dataset",
    "lookup_key",
    "lookup_type",
    "protocol_col",
}

DELIM_MAP = {
    "tab": "\t",
    "comma": ",",
    "pipe": "|",
    "space": None,
    "auto": "__AUTO__",
}

DATASET_PROTOCOL_PATHS = {
    "ASV19": "/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    "ASV5": "/data/Data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv",
    "FamousFigures": "/data/Data/famousfigures/protocol.txt",
    "MLAAD-En": "/data/Data/MLAAD/fake/en/combined_meta.txt",
    "Spoof-Celeb": "/data/Data/SpoofCeleb/metadata/evaluation.csv",
}

DATASET_DELIMITERS = {
    "ASV19": "space",
    "ASV5": "space",
    "FamousFigures": "tab",
    "MLAAD-En": "pipe",
    "Spoof-Celeb": "comma",
}

DATASET_SCORE_TOKENS = {
    "ASV19": "eval_2019",
    "ASV5": "asvspoof5",
    "FamousFigures": "Famous_Figures",
    "MLAAD-En": "Multilingual",
    "Spoof-Celeb": "spoofceleb",
}

DEFAULT_UTT_COLUMNS = {
    "ASV19": 1,
    "ASV5": 2,
    "FamousFigures": 1,
    "MLAAD-En": 1,
    "Spoof-Celeb": 1,
}

VALID_SCORE_SUFFIXES = {".txt", ".tsv", ".score", ""}
SCORE_COLUMNS = ["filename", "placeholder", "key", "score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize SSL score files by canonical TTS system across multiple datasets."
    )
    parser.add_argument("--score_dir", type=Path, required=True, help="Directory containing SSL score files")
    parser.add_argument("--output_root", type=Path, required=True, help="Root output directory")
    parser.add_argument("--tts_master_csv", type=Path, required=True, help="Master TTS metadata CSV")
    parser.add_argument("--tts_lookup_csv", type=Path, required=True, help="Dataset lookup mapping CSV")
    parser.add_argument(
        "--asv19_utt_col",
        type=int,
        default=DEFAULT_UTT_COLUMNS["ASV19"],
        help="1-based utterance ID column for ASV19 protocol",
    )
    parser.add_argument(
        "--asv5_utt_col",
        type=int,
        default=DEFAULT_UTT_COLUMNS["ASV5"],
        help="1-based utterance ID column for ASV5 protocol",
    )
    parser.add_argument(
        "--ff_utt_col",
        type=int,
        default=DEFAULT_UTT_COLUMNS["FamousFigures"],
        help="1-based utterance ID column for Famous Figures protocol",
    )
    parser.add_argument(
        "--mlaad_utt_col",
        type=int,
        default=DEFAULT_UTT_COLUMNS["MLAAD-En"],
        help="1-based utterance ID column for MLAAD-En protocol",
    )
    parser.add_argument(
        "--spoofceleb_utt_col",
        type=int,
        default=DEFAULT_UTT_COLUMNS["Spoof-Celeb"],
        help="1-based utterance ID column for Spoof-Celeb protocol",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for helper CSVs and protocols",
    )
    parser.add_argument(
        "--asv19_protocol",
        type=Path,
        default=Path(DATASET_PROTOCOL_PATHS["ASV19"]),
        help="Optional override for ASV19 protocol path",
    )
    parser.add_argument(
        "--asv5_protocol",
        type=Path,
        default=Path(DATASET_PROTOCOL_PATHS["ASV5"]),
        help="Optional override for ASV5 protocol path",
    )
    parser.add_argument(
        "--ff_protocol",
        type=Path,
        default=Path(DATASET_PROTOCOL_PATHS["FamousFigures"]),
        help="Optional override for Famous Figures protocol path",
    )
    parser.add_argument(
        "--mlaad_protocol",
        type=Path,
        default=Path(DATASET_PROTOCOL_PATHS["MLAAD-En"]),
        help="Optional override for MLAAD-En protocol path",
    )
    parser.add_argument(
        "--spoofceleb_protocol",
        type=Path,
        default=Path(DATASET_PROTOCOL_PATHS["Spoof-Celeb"]),
        help="Optional override for Spoof-Celeb protocol path",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = text.casefold()
    text = re.sub(r"[\s._-]+", "", text)
    return text


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "tts_system"


def human_col_to_index(column_number: int) -> int:
    if int(column_number) < 1:
        raise ValueError(f"Protocol columns must be >= 1, got: {column_number}")
    return int(column_number) - 1


def validate_lookup_columns(lookup_df: pd.DataFrame) -> None:
    missing = REQUIRED_LOOKUP_COLUMNS - set(lookup_df.columns)
    if missing:
        raise ValueError(f"Lookup CSV is missing required columns: {sorted(missing)}")


def canonicalize_dataset_name(dataset: object) -> str:
    raw = str(dataset).strip()
    normalized = normalize_text(raw)

    aliases = {
        "ASV19": {"asv19", "asvspoof2019la", "asvspoof2019", "eval2019"},
        "ASV5": {"asv5", "asvspoof5"},
        "FamousFigures": {"famousfigures", "famousfiguresdataset", "ff"},
        "MLAAD-En": {"mlaaden", "mlaad", "mlaadenglish", "multilingual"},
        "Spoof-Celeb": {"spoofceleb", "spoofcelebdataset", "sc"},
    }

    for canonical, candidates in aliases.items():
        if normalized in candidates:
            return canonical
    return raw


def read_master_tts_csv(path: Path, encoding: str) -> pd.DataFrame:
    master_df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
    if "tts_normalized" not in master_df.columns:
        raise ValueError("Master CSV must contain a 'tts_normalized' column")

    master_df["tts_normalized"] = master_df["tts_normalized"].astype(str).str.strip()
    master_df = master_df[master_df["tts_normalized"] != ""].copy()
    if master_df.empty:
        raise ValueError(f"Master CSV contains no usable TTS rows: {path}")

    if master_df["tts_normalized"].duplicated().any():
        duplicates = sorted(master_df.loc[master_df["tts_normalized"].duplicated(), "tts_normalized"].unique())
        raise ValueError(f"Master CSV contains duplicate tts_normalized values: {duplicates}")

    return master_df


def read_lookup_csv(path: Path, encoding: str) -> pd.DataFrame:
    lookup_df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
    validate_lookup_columns(lookup_df)
    lookup_df = lookup_df.copy()
    lookup_df["tts_normalized"] = lookup_df["tts_normalized"].astype(str).str.strip()
    lookup_df["dataset"] = lookup_df["dataset"].map(canonicalize_dataset_name)
    lookup_df["lookup_key"] = lookup_df["lookup_key"].astype(str).str.strip()
    lookup_df["lookup_type"] = lookup_df["lookup_type"].astype(str).str.strip().str.casefold()
    lookup_df["protocol_col"] = lookup_df["protocol_col"].astype(str).str.strip()

    invalid_protocol_cols = pd.to_numeric(lookup_df["protocol_col"], errors="coerce").isna()
    if invalid_protocol_cols.any():
        bad_rows = lookup_df.loc[invalid_protocol_cols, ["tts_normalized", "dataset", "protocol_col"]]
        raise ValueError(f"Lookup CSV contains non-numeric protocol_col values:\n{bad_rows.to_string(index=False)}")

    return lookup_df


def build_tts_alias_map(master_df: pd.DataFrame) -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    aliases_series = master_df["aliases"] if "aliases" in master_df.columns else pd.Series("", index=master_df.index)

    for tts_name, aliases_value in zip(master_df["tts_normalized"], aliases_series):
        aliases = {normalize_text(tts_name)}
        for raw_alias in re.split(r"[|,;]", str(aliases_value)):
            cleaned = raw_alias.strip()
            if cleaned:
                aliases.add(normalize_text(cleaned))
        alias_map[tts_name] = aliases

    return alias_map


def resolve_ar_nar(master_row: pd.Series) -> str:
    if "ar_nar" not in master_row.index:
        return "UNKNOWN"

    normalized = normalize_text(master_row.get("ar_nar", ""))
    if not normalized:
        return "UNKNOWN"
    if normalized == "ar":
        return "AR"
    if normalized == "nar":
        return "NAR"
    return "UNKNOWN"


def get_dataset_protocol_path(dataset: str, args: argparse.Namespace) -> Path:
    override_map = {
        "ASV19": args.asv19_protocol,
        "ASV5": args.asv5_protocol,
        "FamousFigures": args.ff_protocol,
        "MLAAD-En": args.mlaad_protocol,
        "Spoof-Celeb": args.spoofceleb_protocol,
    }
    return Path(override_map[dataset])


def get_dataset_delimiter(dataset: str) -> str | None:
    label = DATASET_DELIMITERS[dataset]
    delimiter = DELIM_MAP.get(label)
    if delimiter is None and label not in DELIM_MAP:
        raise ValueError(f"Unsupported delimiter label '{label}' for dataset {dataset}")
    return delimiter


def detect_delimiter(line: str) -> str | None:
    counts = {candidate: line.count(candidate) for candidate in ("\t", ",", "|")}
    delimiter = max(counts, key=counts.get)
    return delimiter if counts[delimiter] > 0 else None


def split_protocol_line(line: str, delimiter: str | None) -> list[str]:
    cleaned = line.rstrip("\r\n")
    if delimiter == "__AUTO__":
        delimiter = detect_delimiter(cleaned)
    if delimiter is None:
        return cleaned.split()
    return [part.strip() for part in cleaned.split(delimiter)]


def parse_score_filename(score_path: Path) -> tuple[str, str]:
    stem = score_path.stem
    matches: list[tuple[int, str, str]] = []

    for dataset, token in DATASET_SCORE_TOKENS.items():
        pattern = re.compile(rf"(?:^|_){re.escape(token)}_(.+)$")
        match = pattern.search(stem)
        if not match:
            continue
        ssl_model = match.group(1).strip("_")
        if not ssl_model:
            raise ValueError(f"Missing SSL model suffix in score filename: {score_path.name}")
        matches.append((match.start(), dataset, ssl_model))

    if not matches:
        raise ValueError(
            f"Could not parse dataset token from score filename '{score_path.name}'. "
            f"Expected one of: {sorted(DATASET_SCORE_TOKENS.values())}"
        )

    matches.sort(key=lambda item: item[0], reverse=True)
    _, dataset, ssl_model = matches[0]
    return dataset, ssl_model


def build_score_index(score_dir: Path) -> dict[str, dict[str, Path]]:
    score_index: dict[str, dict[str, Path]] = {}

    for path in sorted(score_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix not in VALID_SCORE_SUFFIXES:
            print(f"[INFO] Skipping unsupported score file extension: {path.name}")
            continue

        dataset, ssl_model = parse_score_filename(path)
        dataset_map = score_index.setdefault(ssl_model, {})
        if dataset in dataset_map:
            raise ValueError(
                f"Duplicate score file for ssl_model='{ssl_model}' and dataset='{dataset}': "
                f"{dataset_map[dataset]} vs {path}"
            )
        dataset_map[dataset] = path

    print(f"[INFO] Indexed {len(score_index)} SSL models from {score_dir}")
    return score_index


def read_protocol_generic(
    protocol_path: Path,
    delimiter: str | None,
    encoding: str,
) -> list[list[str]]:
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    rows: list[list[str]] = []
    with protocol_path.open("r", encoding=encoding) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = split_protocol_line(raw_line, delimiter)
            if not parts:
                continue
            rows.append(parts)

    if not rows:
        raise ValueError(f"Protocol file is empty or unreadable: {protocol_path}")

    print(f"[INFO] Loaded {len(rows)} rows from protocol {protocol_path}")
    return rows


def collect_tts_utterances_for_dataset(
    dataset: str,
    protocol_rows: list[list[str]],
    lookup_rows: pd.DataFrame,
    utt_col_index: int,
    alias_map: dict[str, set[str]],
) -> dict[str, set[str]]:
    utterances_by_tts: dict[str, set[str]] = {}

    for _, row in lookup_rows.iterrows():
        tts_name = row["tts_normalized"]
        lookup_key = str(row["lookup_key"]).strip()
        lookup_type = str(row["lookup_type"]).strip().casefold()
        protocol_col_index = human_col_to_index(row["protocol_col"])
        aliases = set(alias_map.get(tts_name, set()))
        aliases.add(normalize_text(lookup_key))

        matched_utterances: set[str] = set()
        for protocol_row_number, parts in enumerate(protocol_rows, start=1):
            required_max_index = max(utt_col_index, protocol_col_index)
            if required_max_index >= len(parts):
                raise ValueError(
                    f"Protocol row {protocol_row_number} for dataset {dataset} has only {len(parts)} columns, "
                    f"but lookup requires index {required_max_index}."
                )

            utterance_id = str(parts[utt_col_index]).strip().split(".")[0]
            protocol_value = str(parts[protocol_col_index]).strip()

            if lookup_type == "code":
                is_match = protocol_value.casefold() == lookup_key.casefold()
            elif lookup_type == "name":
                is_match = normalize_text(protocol_value) in aliases
            else:
                raise ValueError(
                    f"Unsupported lookup_type '{lookup_type}' for TTS '{tts_name}' in dataset '{dataset}'"
                )

            if is_match and utterance_id:
                matched_utterances.add(utterance_id)

        if matched_utterances:
            utterances_by_tts.setdefault(tts_name, set()).update(matched_utterances)
            print(
                f"[INFO] {dataset}: collected {len(matched_utterances)} utterances for {tts_name} "
                f"using key '{lookup_key}'"
            )
        else:
            print(f"[WARN] {dataset}: no utterances found for {tts_name} using key '{lookup_key}'")

    return utterances_by_tts


def build_utterance_index(
    lookup_df: pd.DataFrame,
    alias_map: dict[str, set[str]],
    args: argparse.Namespace,
) -> dict[str, dict[str, set[str]]]:
    dataset_to_utt_col = {
        "ASV19": human_col_to_index(args.asv19_utt_col),
        "ASV5": human_col_to_index(args.asv5_utt_col),
        "FamousFigures": human_col_to_index(args.ff_utt_col),
        "MLAAD-En": human_col_to_index(args.mlaad_utt_col),
        "Spoof-Celeb": human_col_to_index(args.spoofceleb_utt_col),
    }

    utterance_index: dict[str, dict[str, set[str]]] = {}

    for dataset, dataset_rows in lookup_df.groupby("dataset", sort=True):
        if dataset not in DATASET_PROTOCOL_PATHS:
            print(f"[WARN] Lookup references unsupported dataset '{dataset}'; skipping")
            continue

        protocol_path = get_dataset_protocol_path(dataset, args)
        delimiter = get_dataset_delimiter(dataset)
        protocol_rows = read_protocol_generic(protocol_path, delimiter, args.encoding)

        collected = collect_tts_utterances_for_dataset(
            dataset=dataset,
            protocol_rows=protocol_rows,
            lookup_rows=dataset_rows,
            utt_col_index=dataset_to_utt_col[dataset],
            alias_map=alias_map,
        )

        for tts_name, utterance_ids in collected.items():
            utterance_index.setdefault(tts_name, {})[dataset] = set(utterance_ids)

    return utterance_index


def read_score_file(score_path: Path) -> pd.DataFrame:
    try:
        score_df = pd.read_csv(
            score_path,
            sep=r"\s+",
            header=None,
            names=SCORE_COLUMNS,
            dtype=str,
            engine="python",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read score file: {score_path}") from exc

    if score_df.empty:
        return score_df

    first_row = [str(score_df.iloc[0][column]).strip().casefold() for column in SCORE_COLUMNS]
    if first_row == [column.casefold() for column in SCORE_COLUMNS]:
        score_df = score_df.iloc[1:].reset_index(drop=True)

    score_df["filename"] = score_df["filename"].astype(str).str.strip().str.split(".").str[0]
    return score_df


def filter_score_file(
    score_path: Path,
    utterance_ids: set[str],
    dataset: str,
    tts_normalized: str,
    ssl_model: str,
) -> pd.DataFrame:
    score_df = read_score_file(score_path)
    if score_df.empty:
        print(f"[WARN] Score file is empty: {score_path}")
        return pd.DataFrame(columns=["dataset", "filename", "key", "score", "tts_normalized", "ssl_model"])

    filtered_df = score_df[score_df["filename"].isin(utterance_ids)][["filename", "key", "score"]].copy()
    filtered_df.insert(0, "dataset", dataset)
    filtered_df["tts_normalized"] = tts_normalized
    filtered_df["ssl_model"] = ssl_model
    return filtered_df


def combine_scores_for_tts(
    tts_normalized: str,
    ar_nar_category: str,
    score_index: dict[str, dict[str, Path]],
    utterance_index: dict[str, dict[str, set[str]]],
    output_root: Path,
) -> None:
    datasets_for_tts = utterance_index.get(tts_normalized, {})
    if not datasets_for_tts:
        print(f"[WARN] No utterances collected for {tts_normalized}; skipping")
        return

    output_dir = output_root / ar_nar_category / sanitize_name(tts_normalized)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ssl_model, dataset_paths in sorted(score_index.items()):
        filtered_parts: list[pd.DataFrame] = []

        for dataset, utterance_ids in sorted(datasets_for_tts.items()):
            score_path = dataset_paths.get(dataset)
            if score_path is None:
                print(
                    f"[WARN] Missing score file for tts='{tts_normalized}', ssl_model='{ssl_model}', "
                    f"dataset='{dataset}'"
                )
                continue

            filtered_df = filter_score_file(
                score_path=score_path,
                utterance_ids=utterance_ids,
                dataset=dataset,
                tts_normalized=tts_normalized,
                ssl_model=ssl_model,
            )
            if filtered_df.empty:
                print(
                    f"[WARN] Filtered score file is empty for tts='{tts_normalized}', "
                    f"ssl_model='{ssl_model}', dataset='{dataset}'"
                )
                continue
            filtered_parts.append(filtered_df)

        if not filtered_parts:
            continue

        combined_df = pd.concat(filtered_parts, ignore_index=True)
        output_path = output_dir / f"{ssl_model}.txt"
        combined_df.to_csv(output_path, sep="\t", index=False)
        print(
            f"[INFO] Saved {len(combined_df)} rows for tts='{tts_normalized}', "
            f"ssl_model='{ssl_model}' to {output_path}"
        )


def main() -> None:
    args = parse_args()

    if not args.score_dir.is_dir():
        raise NotADirectoryError(f"Score directory not found: {args.score_dir}")
    if not args.tts_master_csv.is_file():
        raise FileNotFoundError(f"Master CSV not found: {args.tts_master_csv}")
    if not args.tts_lookup_csv.is_file():
        raise FileNotFoundError(f"Lookup CSV not found: {args.tts_lookup_csv}")

    master_df = read_master_tts_csv(args.tts_master_csv, args.encoding)
    lookup_df = read_lookup_csv(args.tts_lookup_csv, args.encoding)
    alias_map = build_tts_alias_map(master_df)

    known_tts = set(master_df["tts_normalized"])
    unknown_lookup_tts = sorted(set(lookup_df["tts_normalized"]) - known_tts)
    if unknown_lookup_tts:
        print(f"[WARN] Lookup CSV references TTS systems not present in master CSV: {unknown_lookup_tts}")

    score_index = build_score_index(args.score_dir)
    utterance_index = build_utterance_index(lookup_df, alias_map, args)

    args.output_root.mkdir(parents=True, exist_ok=True)

    master_by_tts = master_df.set_index("tts_normalized", drop=False)
    for tts_normalized in sorted(master_by_tts.index):
        ar_nar_category = resolve_ar_nar(master_by_tts.loc[tts_normalized])
        if ar_nar_category == "UNKNOWN":
            print(f"[WARN] Missing or ambiguous ar_nar for {tts_normalized}; saving under UNKNOWN")

        combine_scores_for_tts(
            tts_normalized=tts_normalized,
            ar_nar_category=ar_nar_category,
            score_index=score_index,
            utterance_index=utterance_index,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
