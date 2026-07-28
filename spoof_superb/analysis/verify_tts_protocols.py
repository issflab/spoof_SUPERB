#!/usr/bin/env python3
"""
Verify that generated normalized TTS protocol files are consistent with raw dataset protocols.

Expected inputs:
1) Master protocol CSV, e.g. tts_master_protocol_draft.csv
2) Dataset lookup map CSV, e.g. tts_dataset_lookup_map_draft.csv
3) Raw dataset protocol files for datasets such as ASV19, ASV5, Famous Figures,
   MLAAD-En, and Spoof-Celeb, configured below in this script

This script checks:
- Required columns exist in the generated protocol files
- Each dataset lookup row points to a valid raw protocol file
- The lookup key is present in the specified column of the raw dataset protocol
- Duplicate / conflicting rows in the lookup map
- Optional consistency between master protocol and lookup map

Usage example:
python verify_tts_protocols.py \
  --master tts_master_protocol_draft.csv \
  --lookup tts_dataset_lookup_map_draft.csv \
  --output-dir verification_out

Notes:
- protocol_col is interpreted as 1-based column index, matching the convention used in
  your earlier descriptions (e.g. ASV5 -> 8th column, ASV19 -> 4th column).
- lookup_type can be code, model_name, or either.
- By default, the script splits raw protocol rows on whitespace. You can override the
  delimiter per dataset by editing DATASET_DELIMITERS below.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REQUIRED_MASTER_COLUMNS = {
    "tts_normalized",
    "ar_nar",
}

REQUIRED_LOOKUP_COLUMNS = {
    "tts_normalized",
    "dataset",
    "lookup_key",
    "lookup_type",
    "protocol_col",
}

CANONICAL_DATASET_ALIASES = {
    "ASV19": {"ASV19", "ASVSpoof2019LA", "ASVSpoof 2019LA", "ASV Spoof 2019 LA"},
    "ASV5": {"ASV5", "ASVSpoof5", "ASVSpoof 5", "ASV Spoof 5"},
    "FamousFigures": {"FamousFigures", "Famous Figures", "FF"},
    "MLAAD-En": {"MLAAD-En", "MLAAD", "MLAAD_EN", "MLAAD En"},
    "Spoof-Celeb": {"Spoof-Celeb", "SpoofCeleb", "SC", "Spoof Celeb"},
}

DELIM_MAP = {
    "tab": "\t",
    "comma": ",",
    "pipe": "|",
    "space": None,
    "auto": "__AUTO__",
}

# Edit these mappings instead of passing every dataset protocol on the command line.
# Paths may be absolute or relative to the current working directory.
DATASET_PROTOCOL_PATHS = {
    "ASV19": "/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    "ASV5": "/data/Data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv",
    "FamousFigures": "/data/Data/famousfigures/protocol.txt",
    "MLAAD-En": "/data/Data/MLAAD/fake/en/combined_meta.txt",
    "Spoof-Celeb": "/data/Data/SpoofCeleb/metadata/evaluation.csv",
}

# Allowed values: tab, comma, pipe, space, auto
DATASET_DELIMITERS = {
    "ASV19": "space",
    "ASV5": "space",
    "FamousFigures": "tab",
    "MLAAD-En": "pipe",
    "Spoof-Celeb": "comma",
}


@dataclass
class VerificationIssue:
    severity: str
    category: str
    message: str
    row_index: Optional[int] = None
    dataset: Optional[str] = None
    tts_normalized: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "row_index": self.row_index,
            "dataset": self.dataset,
            "tts_normalized": self.tts_normalized,
        }


@dataclass
class MatchResult:
    found: bool
    occurrences: int
    matched_rows: List[int]
    sample_values: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated TTS protocol CSVs against raw dataset protocols.")
    parser.add_argument("--master", required=True, help="Path to master normalized protocol CSV")
    parser.add_argument("--lookup", required=True, help="Path to dataset lookup map CSV")
    parser.add_argument(
        "--output-dir",
        default="verification_out",
        help="Directory to write reports",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for CSV and protocol files",
    )
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_name(text: str) -> str:
    text = text.strip()
    text = text.replace("_", " ")
    text = text.replace("-", "")
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold().replace(" ", "")


def canonicalize_dataset_name(dataset: str) -> str:
    raw = normalize_spaces(dataset)
    folded = raw.casefold()
    for canonical, aliases in CANONICAL_DATASET_ALIASES.items():
        if folded in {a.casefold() for a in aliases}:
            return canonical
    return raw


def read_csv_rows(path: Path, encoding: str) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def detect_delimiter(line: str) -> Optional[str]:
    candidates = ["\t", ",", "|"]
    counts = {c: line.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    return None


def split_protocol_line(line: str, delim: Optional[str]) -> List[str]:
    line = line.rstrip("\n\r")
    if delim == "__AUTO__":
        delim = detect_delimiter(line)
    if delim is None:
        return line.split()
    return [part.strip() for part in line.split(delim)]


def load_protocol_rows(path: Path, delim: Optional[str], encoding: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open("r", encoding=encoding) as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = split_protocol_line(raw, delim)
            if not parts:
                continue
            rows.append(parts)
    return rows


def maybe_int(value: str) -> Optional[int]:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def find_matches(
    rows: List[List[str]],
    protocol_col_1_based: int,
    lookup_key: str,
    lookup_type: str,
) -> MatchResult:
    col_idx = protocol_col_1_based - 1
    matched_rows: List[int] = []
    sample_values: List[str] = []

    key_norm = normalize_name(lookup_key)
    for row_idx, parts in enumerate(rows, start=1):
        if col_idx >= len(parts):
            continue
        value = parts[col_idx]
        exact = value == lookup_key
        norm = normalize_name(value) == key_norm

        found_here = False
        if lookup_type in {"code", "model_name", "either"}:
            found_here = exact or norm
        else:
            found_here = exact or norm

        if found_here:
            matched_rows.append(row_idx)
            if len(sample_values) < 5:
                sample_values.append(value)

    return MatchResult(
        found=bool(matched_rows),
        occurrences=len(matched_rows),
        matched_rows=matched_rows,
        sample_values=sample_values,
    )


def validate_master(rows: List[dict], fieldnames: List[str]) -> List[VerificationIssue]:
    issues: List[VerificationIssue] = []
    missing = REQUIRED_MASTER_COLUMNS - set(fieldnames)
    if missing:
        issues.append(VerificationIssue(
            severity="ERROR",
            category="master_schema",
            message=f"Missing required master columns: {sorted(missing)}",
        ))
        return issues

    seen = {}
    for idx, row in enumerate(rows, start=2):
        tts = normalize_spaces(row.get("tts_normalized", ""))
        ar_nar = normalize_spaces(row.get("ar_nar", ""))
        if not tts:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="master_data",
                message="Empty tts_normalized",
                row_index=idx,
            ))
            continue
        if not ar_nar:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="master_data",
                message="Empty ar_nar",
                row_index=idx,
                tts_normalized=tts,
            ))
            continue
        key = (normalize_name(tts), normalize_name(ar_nar))
        if key in seen:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="master_duplicate",
                message=(
                    f"Duplicate (tts_normalized, ar_nar)=('{tts}', '{ar_nar}') "
                    f"also seen on CSV row {seen[key]}"
                ),
                row_index=idx,
                tts_normalized=tts,
            ))
        else:
            seen[key] = idx
    return issues


def validate_lookup_schema(rows: List[dict], fieldnames: List[str]) -> List[VerificationIssue]:
    issues: List[VerificationIssue] = []
    missing = REQUIRED_LOOKUP_COLUMNS - set(fieldnames)
    if missing:
        issues.append(VerificationIssue(
            severity="ERROR",
            category="lookup_schema",
            message=f"Missing required lookup columns: {sorted(missing)}",
        ))
    return issues


def validate_lookup_rows(
    rows: List[dict],
    master_rows: List[dict],
    dataset_protocols: Dict[str, Path],
    dataset_delims: Dict[str, Optional[str]],
    encoding: str,
) -> Tuple[List[VerificationIssue], List[dict]]:
    issues: List[VerificationIssue] = []
    results: List[dict] = []

    master_names = {normalize_name(r["tts_normalized"]): r["tts_normalized"] for r in master_rows if r.get("tts_normalized")}

    lookup_identity_counter: Counter = Counter()
    for idx, row in enumerate(rows, start=2):
        tts_normalized = normalize_spaces(row.get("tts_normalized", ""))
        dataset = canonicalize_dataset_name(row.get("dataset", ""))
        lookup_key = str(row.get("lookup_key", "")).strip()
        lookup_type = normalize_spaces(str(row.get("lookup_type", ""))).lower()
        protocol_col = maybe_int(str(row.get("protocol_col", "")))

        identity = (normalize_name(tts_normalized), dataset, normalize_name(lookup_key), protocol_col)
        lookup_identity_counter[identity] += 1

        if normalize_name(tts_normalized) not in master_names:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="master_lookup_consistency",
                message=f"tts_normalized '{tts_normalized}' not found in master file",
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))

        if not dataset:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="lookup_data",
                message="Empty dataset",
                row_index=idx,
                tts_normalized=tts_normalized,
            ))
            continue

        if not lookup_key:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="lookup_data",
                message="Empty lookup_key",
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))
            continue

        if protocol_col is None or protocol_col < 1:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="lookup_data",
                message=f"Invalid protocol_col '{row.get('protocol_col')}'",
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))
            continue

        if dataset not in dataset_protocols:
            issues.append(VerificationIssue(
                severity="WARNING",
                category="missing_protocol",
                message=f"No raw protocol path provided for dataset '{dataset}'",
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))
            results.append({
                "csv_row": idx,
                "tts_normalized": tts_normalized,
                "dataset": dataset,
                "lookup_key": lookup_key,
                "lookup_type": lookup_type,
                "protocol_col": protocol_col,
                "status": "SKIPPED_NO_PROTOCOL",
                "occurrences": 0,
                "matched_rows": "",
                "sample_values": "",
            })
            continue

        protocol_path = dataset_protocols[dataset]
        if not protocol_path.exists():
            issues.append(VerificationIssue(
                severity="ERROR",
                category="missing_protocol",
                message=f"Raw protocol file does not exist: {protocol_path}",
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))
            results.append({
                "csv_row": idx,
                "tts_normalized": tts_normalized,
                "dataset": dataset,
                "lookup_key": lookup_key,
                "lookup_type": lookup_type,
                "protocol_col": protocol_col,
                "status": "ERROR_PROTOCOL_NOT_FOUND",
                "occurrences": 0,
                "matched_rows": "",
                "sample_values": "",
            })
            continue

        delim = dataset_delims.get(dataset, None)
        try:
            protocol_rows = load_protocol_rows(protocol_path, delim=delim, encoding=encoding)
        except Exception as exc:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="protocol_read",
                message=f"Failed to read raw protocol '{protocol_path}': {exc}",
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))
            results.append({
                "csv_row": idx,
                "tts_normalized": tts_normalized,
                "dataset": dataset,
                "lookup_key": lookup_key,
                "lookup_type": lookup_type,
                "protocol_col": protocol_col,
                "status": "ERROR_READ_FAILED",
                "occurrences": 0,
                "matched_rows": "",
                "sample_values": "",
            })
            continue

        match = find_matches(protocol_rows, protocol_col_1_based=protocol_col, lookup_key=lookup_key, lookup_type=lookup_type)
        status = "PASS" if match.found else "FAIL_NOT_FOUND"
        if not match.found:
            issues.append(VerificationIssue(
                severity="ERROR",
                category="lookup_miss",
                message=(
                    f"lookup_key '{lookup_key}' not found in dataset '{dataset}' "
                    f"column {protocol_col} of {protocol_path.name}"
                ),
                row_index=idx,
                dataset=dataset,
                tts_normalized=tts_normalized,
            ))

        results.append({
            "csv_row": idx,
            "tts_normalized": tts_normalized,
            "dataset": dataset,
            "lookup_key": lookup_key,
            "lookup_type": lookup_type,
            "protocol_col": protocol_col,
            "status": status,
            "occurrences": match.occurrences,
            "matched_rows": ";".join(map(str, match.matched_rows[:50])),
            "sample_values": ";".join(match.sample_values),
        })

    for identity, count in lookup_identity_counter.items():
        if count > 1:
            tts_key, dataset, lookup_key_norm, protocol_col = identity
            issues.append(VerificationIssue(
                severity="WARNING",
                category="lookup_duplicate",
                message=(
                    f"Duplicate lookup mapping detected for dataset='{dataset}', protocol_col={protocol_col}, "
                    f"normalized_lookup_key='{lookup_key_norm}', count={count}"
                ),
                dataset=dataset,
                tts_normalized=master_names.get(tts_key),
            ))

    return issues, results


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_issues(issues: List[VerificationIssue]) -> dict:
    by_severity = Counter(i.severity for i in issues)
    by_category = Counter(i.category for i in issues)
    return {
        "total_issues": len(issues),
        "by_severity": dict(by_severity),
        "by_category": dict(by_category),
    }


def main() -> int:
    args = parse_args()

    master_path = Path(args.master)
    lookup_path = Path(args.lookup)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_protocols = {
        canonicalize_dataset_name(dataset): Path(path_str)
        for dataset, path_str in DATASET_PROTOCOL_PATHS.items()
        if str(path_str).strip()
    }

    dataset_delims: Dict[str, Optional[str]] = {}
    for dataset, delim_name in DATASET_DELIMITERS.items():
        canonical_dataset = canonicalize_dataset_name(dataset)
        delim_key = str(delim_name).strip()
        if delim_key not in DELIM_MAP:
            raise ValueError(
                f"Invalid delimiter '{delim_name}' configured for dataset '{dataset}'. "
                f"Allowed: {sorted(DELIM_MAP)}"
            )
        dataset_delims[canonical_dataset] = DELIM_MAP[delim_key]

    print(dataset_delims)

    all_issues: List[VerificationIssue] = []

    if not master_path.exists():
        print(f"ERROR: master file not found: {master_path}", file=sys.stderr)
        return 2
    if not lookup_path.exists():
        print(f"ERROR: lookup file not found: {lookup_path}", file=sys.stderr)
        return 2

    master_rows, master_fields = read_csv_rows(master_path, args.encoding)
    lookup_rows, lookup_fields = read_csv_rows(lookup_path, args.encoding)

    all_issues.extend(validate_master(master_rows, master_fields))
    all_issues.extend(validate_lookup_schema(lookup_rows, lookup_fields))

    if not any(i.severity == "ERROR" and i.category in {"master_schema", "lookup_schema"} for i in all_issues):
        lookup_issues, verification_rows = validate_lookup_rows(
            rows=lookup_rows,
            master_rows=master_rows,
            dataset_protocols=dataset_protocols,
            dataset_delims=dataset_delims,
            encoding=args.encoding,
        )
        all_issues.extend(lookup_issues)
    else:
        verification_rows = []

    issue_rows = [issue.as_dict() for issue in all_issues]
    summary = summarize_issues(all_issues)

    write_csv(out_dir / "verification_results.csv", verification_rows)
    write_csv(out_dir / "verification_issues.csv", issue_rows)
    with (out_dir / "verification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pass_count = sum(1 for r in verification_rows if r.get("status") == "PASS")
    fail_count = sum(1 for r in verification_rows if str(r.get("status", "")).startswith("FAIL"))
    skipped_count = sum(1 for r in verification_rows if str(r.get("status", "")).startswith("SKIPPED"))
    error_count = sum(1 for r in verification_rows if str(r.get("status", "")).startswith("ERROR"))

    print("Verification complete")
    print(f"Master file : {master_path}")
    print(f"Lookup file : {lookup_path}")
    print(f"Output dir  : {out_dir}")
    print(f"Rows checked: {len(verification_rows)}")
    print(f"PASS        : {pass_count}")
    print(f"FAIL        : {fail_count}")
    print(f"SKIPPED     : {skipped_count}")
    print(f"ERROR       : {error_count}")
    print(f"Issues      : {summary['total_issues']}")

    return 1 if fail_count > 0 or any(i.severity == "ERROR" for i in all_issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
