"""
Verify and split acoustic degradation score files by source dataset.

Background
----------
The Spoof-SUPERB baseline score files mix utterances from four sources:
  - ASV19 LA eval   : 71 237 LA_E_ IDs  (= ASVLD clean)
  - ASV21 LA:C1     : 25 938 LA_E_ IDs  (codec="none" from ASV21 LA metadata)
  - ASV21 DF:C1     : 17 131 DF_E_ IDs  (codec="nocodec" from ASV21 DF trial_metadata.txt)
  - ASV5 no-codec   : 171 602 E_  IDs  (codec="-" in ASV5 track-1 TSV)

Each degraded condition needs a specific subset of baseline sources appended
so that the augmented condition covers the same data as the baseline (285,908
rows total). The per-condition mapping is defined in CONDITION_SOURCES:
  Additive_Noise / Reverberation / Resampling → add asv21_la_c1, asv21_df_c1, asv5_noc
  Codec_Compression                           → add asv21_la_c1 only
  Channel_Distortions                         → add asv19_eval, asv21_df_c1

Functions
---------
load_protocols(...)     build ID sets for every source/condition
verify_score_file(...)  check a score file against expected ID sets
separate_by_source(...) split a score file's rows into per-source buckets
get_augmented_scores(.) combine a condition file + missing-source rows
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Type alias: one score-file row
# ---------------------------------------------------------------------------
Row = Tuple[str, str, float]   # (utt_id, label, score)

# ---------------------------------------------------------------------------
# Per-condition target composition map
# Keys match the basename of each condition directory.
# Values list every source that the FIXED condition file must contain.
# The script checks which sources are absent (zero ID overlap) and adds only
# those from the matching baseline file.
#
# Why these targets:
#   Additive_Noise / Reverberation / Resampling:
#       ASVLD-based: same ASV19 eval IDs (degraded audio) + three clean sources.
#       asv19_eval is already present (same IDs); the other three are absent.
#   Codec_Compression:
#       DF:C2-C9 (replaces DF:C1) + ASV5:C01-C10 (replaces C00) already present.
#       Only ASV21 LA:C1 is missing — no codec-degraded LA:C1 exists in this set.
#   Channel_Distortions:
#       ASV21 LA:C2-C7 (channel-degraded) + ASV5:C11 already present.
#       C1 (clean) is stripped out (see CONDITION_EXCLUDE below).
#       Missing: ASV19 eval and ASV21 DF:C1.
# ---------------------------------------------------------------------------
CONDITION_TARGET: Dict[str, List[str]] = {
    "Additive_Noise":      ["asv21_la_c1", "asv21_df_c1", "asv5_noc"],
    "Reverberation":       ["asv21_la_c1", "asv21_df_c1", "asv5_noc"],
    "Resampling":          ["asv21_la_c1", "asv21_df_c1", "asv5_noc"],
    "Codec_Compression":   ["asv21_la_c1"],
    "Channel_Distortions": ["asv19_eval", "asv21_df_c1"],
}

# ---------------------------------------------------------------------------
# Per-condition exclusion map
# Sources listed here are stripped from the condition rows BEFORE augmentation.
# Use this when the condition file accidentally contains clean utterances that
# should be replaced by their degraded equivalents.
#
# Channel_Distortions: the raw files contain the full ASV21 LA eval (C1-C7),
# but C1 is the clean/no-channel condition.  It must be removed so only the
# channel-degraded C2-C7 utterances remain.
#
# Codec_Compression: the raw files were scored on the full DF eval set, which
# includes the C1 (nocodec/clean) utterances.  Those clean rows must be removed
# so only the codec-degraded C2-C9 utterances remain.
# ---------------------------------------------------------------------------
CONDITION_EXCLUDE: Dict[str, List[str]] = {
    "Channel_Distortions": ["asv21_la_c1"],
    "Codec_Compression":   ["asv21_df_c1"],
}

# Filename prefix to strip from condition files before baseline matching.
# Only needed when a condition dir uses a different prefix from the default.
CONDITION_PREFIX: Dict[str, str] = {
    "Resampling": "linear_head_resamp_",
}


# ---------------------------------------------------------------------------
# Protocol loaders
# ---------------------------------------------------------------------------

def load_asv19_eval_ids(protocol_path: str) -> Set[str]:
    """Return utt_ids from ASVspoof2019 LA eval protocol (column index 1)."""
    ids: Set[str] = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


def load_asv21_la_c1_ids(protocol_path: str, codec_col: int = 2,
                          c1_value: str = "none") -> Set[str]:
    """
    Return utt_ids for ASV21 LA:C1 (the no-codec condition).
    Protocol format: speaker utt_id codec ... (space-separated).
    """
    ids: Set[str] = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > codec_col and parts[codec_col] == c1_value:
                ids.add(parts[1])
    return ids


def load_asv5_noc_ids(protocol_path: str, codec_col: int = 3,
                      noc_value: str = "-") -> Set[str]:
    """
    Return utt_ids for ASV5 utterances with no codec degradation.
    Protocol format: row_id utt_id gender codec ... (TSV or space).
    """
    ids: Set[str] = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > codec_col and parts[codec_col] == noc_value:
                ids.add(parts[1])
    return ids


def load_asv21_df_c1_ids(protocol_path: str, codec_col: int = 2,
                          c1_value: str = "nocodec") -> Set[str]:
    """
    Return utt_ids for ASV21 DF:C1 (no-codec condition) from trial_metadata.txt.
    Format: speaker utt_id codec source system label ...
    C1 is indicated by codec == "nocodec".
    """
    ids: Set[str] = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > codec_col and parts[codec_col] == c1_value:
                ids.add(parts[1])
    return ids


def load_asvld_condition_ids(protocol_path: str) -> Set[str]:
    """
    Return suffixed utt_ids from an ASVLD per-condition protocol.
    Format: speaker LA_E_<id>_<condition> - system label
    The utt_id (column 1) already has the condition suffix.
    """
    ids: Set[str] = set()
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ids.add(parts[1])
    return ids


# ---------------------------------------------------------------------------
# Score file I/O
# ---------------------------------------------------------------------------

def _strip_ext(utt_id: str) -> str:
    """Strip .flac / .wav extension from an utterance ID if present."""
    for ext in (".flac", ".wav"):
        if utt_id.endswith(ext):
            return utt_id[: -len(ext)]
    return utt_id


_HEADER_TOKENS = {"filename", "utt_id", "uttid", "key", "score", "label"}


def parse_score_file(path: str) -> List[Row]:
    """
    Parse a score file. Two formats are supported automatically:

      4-column (Nithin):  utt_id - label score
      3-column (Hashim):  utt_id label score   (no dash separator)

    Header lines (e.g. "filename key score") are skipped.
    Audio extensions (.flac, .wav) are stripped from the utt_id.
    """
    rows: List[Row] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # Skip header lines whose first token is a known column name
            if parts[0].lower() in _HEADER_TOKENS:
                continue

            if len(parts) == 4:
                # 4-column: utt_id - label score
                utt_id = _strip_ext(parts[0])
                label = parts[2]
                score_str = parts[3]
            elif len(parts) == 3:
                # 3-column: utt_id label score
                utt_id = _strip_ext(parts[0])
                label = parts[1]
                score_str = parts[2]
            else:
                print(f"  [WARN] {path}:{lineno}: expected 3 or 4 columns, "
                      f"got {len(parts)} — skipping")
                continue

            try:
                score = float(score_str)
            except ValueError:
                print(f"  [WARN] {path}:{lineno}: cannot parse score "
                      f"'{score_str}' — skipping")
                continue
            rows.append((utt_id, label, score))
    return rows


def write_score_file(path: str, rows: List[Row]) -> None:
    """Write rows back to a score file in the standard 4-column format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for utt_id, label, score in rows:
            f.write(f"{utt_id} - {label} {score}\n")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_score_file(
    score_path: str,
    expected_id_sets: Dict[str, Set[str]],
    strict: bool = True,
) -> bool:
    """
    Verify a score file against expected utterance sets.

    Parameters
    ----------
    score_path      : path to the score file
    expected_id_sets: mapping of source-name -> set of valid utt_ids for
                      that source. The union is the full valid set.
    strict          : if True, flag utterances in the score file that do not
                      belong to any known source (unexpected IDs).
                      Missing utterances (valid IDs absent from the file) are
                      reported as INFO only, since partial evaluations are
                      expected (e.g. ASV21 DF:C1 has 67 981 nocodec utterances
                      but a score file may only cover a subset).

    Returns True if no unexpected IDs are found, False otherwise.
    """
    rows = parse_score_file(score_path)
    found_ids = {r[0] for r in rows}
    all_valid_ids = set().union(*expected_id_sets.values())

    extra = found_ids - all_valid_ids

    ok = True
    name = os.path.basename(score_path)

    if extra and strict:
        ok = False
        print(f"  [FAIL] {name}: {len(extra)} utterances not in any protocol")
        for uid in sorted(extra)[:5]:
            print(f"         e.g. {uid}")
        if len(extra) > 5:
            print(f"         ... and {len(extra) - 5} more")

    per_source = []
    for src, ids in expected_id_sets.items():
        count = len(found_ids & ids)
        total = len(ids)
        per_source.append(f"{src}={count}/{total}")

    status = "[OK]  " if ok else "[FAIL]"
    print(f"  {status} {name}: {len(found_ids)} rows  "
          f"({', '.join(per_source)})")
    return ok


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------

def separate_by_source(
    rows: List[Row],
    source_id_sets: Dict[str, Set[str]],
) -> Dict[str, List[Row]]:
    """
    Split score-file rows into per-source buckets.

    Parameters
    ----------
    rows           : parsed score rows (utt_id, label, score)
    source_id_sets : mapping of source-name -> set of utt_ids for that source

    Returns a dict of {source_name: [rows]}.
    Rows not matched by any source end up in the '_unmatched' bucket.
    """
    buckets: Dict[str, List[Row]] = defaultdict(list)
    # Build reverse index for O(1) lookup
    id_to_source: Dict[str, str] = {}
    for src, ids in source_id_sets.items():
        for uid in ids:
            id_to_source[uid] = src

    for row in rows:
        uid = row[0]
        src = id_to_source.get(uid, "_unmatched")
        buckets[src].append(row)

    return dict(buckets)


def get_augmented_scores(
    condition_rows: List[Row],
    baseline_rows: List[Row],
    sources_to_add: List[str],
    source_id_sets: Dict[str, Set[str]],
) -> List[Row]:
    """
    Return condition_rows augmented with rows from baseline_rows for each
    source listed in sources_to_add.

    Use this to build a complete score set for a degraded condition:
        augmented = condition_rows (degraded ASVLD)
                  + ASV21_DF_C1 rows from baseline
                  + ASV5_noc    rows from baseline

    Parameters
    ----------
    condition_rows : rows from the degraded condition score file
    baseline_rows  : rows from the baseline score file
    sources_to_add : source names whose baseline rows to append
    source_id_sets : same mapping used for separation

    Returns the combined row list.
    """
    baseline_buckets = separate_by_source(baseline_rows, source_id_sets)
    extra: List[Row] = []
    for src in sources_to_add:
        extra.extend(baseline_buckets.get(src, []))
    return condition_rows + extra


# ---------------------------------------------------------------------------
# Protocols helper — build all source ID sets at once
# ---------------------------------------------------------------------------

def load_protocols(
    asv19_protocol: str,
    asv21_la_protocol: str,
    asv21_df_protocol: str,
    asv5_protocol: str,
) -> Dict[str, Set[str]]:
    """
    Load source ID sets for all four baseline sources.

    Returns
    -------
    Dict with keys: 'asv19_eval', 'asv21_la_c1', 'asv21_df_c1', 'asv5_noc'
    """
    sources: Dict[str, Set[str]] = {}

    print("Loading protocols ...")

    sources["asv19_eval"] = load_asv19_eval_ids(asv19_protocol)
    print(f"  ASV19 LA eval   : {len(sources['asv19_eval']):>7} utterances")

    sources["asv21_la_c1"] = load_asv21_la_c1_ids(asv21_la_protocol)
    print(f"  ASV21 LA:C1     : {len(sources['asv21_la_c1']):>7} utterances")

    sources["asv21_df_c1"] = load_asv21_df_c1_ids(asv21_df_protocol)
    print(f"  ASV21 DF:C1     : {len(sources['asv21_df_c1']):>7} utterances")

    sources["asv5_noc"] = load_asv5_noc_ids(asv5_protocol)
    print(f"  ASV5 no-codec   : {len(sources['asv5_noc']):>7} utterances")

    return sources


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and split acoustic degradation score files by source dataset."
        )
    )

    # Protocol paths
    parser.add_argument(
        "--asv19_protocol", required=True,
        help="ASVspoof2019 LA eval protocol (.trl.txt)",
    )
    parser.add_argument(
        "--asv21_la_protocol", required=True,
        help="ASVspoof2021 LA eval trial_metadata.txt",
    )
    parser.add_argument(
        "--asv21_df_protocol", required=True,
        help="ASVspoof2021 DF eval trial_metadata.txt (codec col 2 == 'nocodec' selects C1)",
    )
    parser.add_argument(
        "--asv5_protocol", required=True,
        help="ASVspoof5 track-1 eval TSV (ASVspoof5.eval.track_1.tsv)",
    )

    # Score file inputs
    parser.add_argument(
        "--baseline_dir", required=True,
        help="Directory containing one baseline score file per SSL model.",
    )
    parser.add_argument(
        "--condition_dirs", nargs="+", default=[],
        help="Directories containing degraded condition score files to verify.",
    )

    # Output
    parser.add_argument(
        "--output_dir", default=None,
        help=(
            "If set, write augmented condition score files here. "
            "Subdirectory per condition dir is created automatically."
        ),
    )
    parser.add_argument(
        "--augment_sources", nargs="+",
        default=["asv21_df_c1", "asv5_noc", "asv21_la_c1"],
        help=(
            "Source buckets from the baseline to append to each degraded "
            "condition file. Default: asv21_df_c1 asv5_noc"
        ),
    )
    parser.add_argument(
        "--no_strict", action="store_true",
        help="Do not flag extra (unexpected) utterances during verification.",
    )
    parser.add_argument(
        "--baseline_prefix", default="linear_head_asvspoof5_",
        help=(
            "Strip this prefix from baseline filenames before name matching. "
            "Defaults to 'linear_head_asvspoof5_' so that "
            "'linear_head_asvspoof5_apc.txt' matches condition file 'APC.txt'."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load protocols
    # ------------------------------------------------------------------
    source_ids = load_protocols(
        asv19_protocol=args.asv19_protocol,
        asv21_la_protocol=args.asv21_la_protocol,
        asv21_df_protocol=args.asv21_df_protocol,
        asv5_protocol=args.asv5_protocol,
    )

    # ------------------------------------------------------------------
    # 2. Build baseline filename index (supports mismatched naming)
    # ------------------------------------------------------------------
    # Maps normalized_stem -> absolute_path for every .txt in baseline_dir.
    # Normalization: strip prefix, lowercase, keep only alphanum chars.
    def _norm(name: str, prefix: str = args.baseline_prefix) -> str:
        stem = os.path.splitext(name)[0]
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix):]
        return "".join(c for c in stem.lower() if c.isalnum())

    baseline_index: Dict[str, str] = {}  # norm_stem -> abs_path
    for bname in os.listdir(args.baseline_dir):
        if bname.endswith(".txt"):
            baseline_index[_norm(bname)] = os.path.join(args.baseline_dir, bname)

    def _find_baseline(cond_fname: str, cond_name: str = ""):
        """Return the baseline path for a condition filename, or None."""
        cond_prefix = CONDITION_PREFIX.get(cond_name, "")
        key = _norm(cond_fname, prefix=cond_prefix) if cond_prefix else _norm(cond_fname)
        # 1. exact normalized match
        if key in baseline_index:
            return baseline_index[key]
        # 2. condition key is a prefix of a baseline key (e.g. 'hubertbase'
        #    matches 'hubertbasell60k')
        matches = [v for k, v in baseline_index.items() if k.startswith(key)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"    [WARN] '{cond_fname}': ambiguous baseline matches — "
                  f"{[os.path.basename(m) for m in matches]}")
        return None

    # ------------------------------------------------------------------
    # 3. Verify baseline files
    # ------------------------------------------------------------------
    print(f"\nVerifying baseline files in: {args.baseline_dir}")
    baseline_expected = {k: v for k, v in source_ids.items()}
    all_ok = True
    for fname in sorted(os.listdir(args.baseline_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(args.baseline_dir, fname)
        ok = verify_score_file(fpath, baseline_expected,
                               strict=not args.no_strict)
        all_ok = all_ok and ok

    # ------------------------------------------------------------------
    # 4. Verify and optionally augment condition files
    # ------------------------------------------------------------------
    for cond_dir in args.condition_dirs:
        cond_name = os.path.basename(cond_dir.rstrip("/"))
        print(f"\nVerifying condition files in: {cond_dir}  [{cond_name}]")

        for fname in sorted(os.listdir(cond_dir)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(cond_dir, fname)
            rows = parse_score_file(fpath)
            found_ids = {r[0] for r in rows}
            print(f"  {fname}: {len(rows)} rows  "
                  f"(LA_E_={sum(1 for uid in found_ids if uid.startswith('LA_E_'))}, "
                  f"DF_E_={sum(1 for uid in found_ids if uid.startswith('DF_E_'))}, "
                  f"E_={sum(1 for uid in found_ids if uid.startswith('E_'))})")

            # Optionally write augmented file
            if args.output_dir:
                baseline_file = _find_baseline(fname, cond_name)
                if baseline_file is None:
                    avail = sorted(baseline_index.keys())
                    cond_prefix = CONDITION_PREFIX.get(cond_name, "")
                    nkey = _norm(fname, cond_prefix) if cond_prefix else _norm(fname)
                    print(f"    [SKIP] no baseline match for '{fname}' "
                          f"(normalized key: '{nkey}'). "
                          f"Available: {avail[:5]}{'...' if len(avail) > 5 else ''}")
                    continue
                baseline_rows = parse_score_file(baseline_file)

                # Strip clean rows that must not appear in this condition
                exclude_sources = CONDITION_EXCLUDE.get(cond_name, [])
                if exclude_sources:
                    exclude_ids = set().union(
                        *(source_ids[s] for s in exclude_sources)
                    )
                    before = len(rows)
                    rows = [r for r in rows if r[0] not in exclude_ids]
                    found_ids = {r[0] for r in rows}
                    print(f"    excluded {before - len(rows)} rows "
                          f"({exclude_sources})")

                target_sources = CONDITION_TARGET.get(
                    cond_name, args.augment_sources
                )
                # Determine which target sources are absent (zero ID overlap)
                missing_sources = [
                    src for src in target_sources
                    if not (source_ids[src] & found_ids)
                ]
                present_sources = [
                    src for src in target_sources
                    if src not in missing_sources
                ]
                print(f"    present: {present_sources}")
                print(f"    missing → adding: {missing_sources}")
                augmented = get_augmented_scores(
                    condition_rows=rows,
                    baseline_rows=baseline_rows,
                    sources_to_add=missing_sources,
                    source_id_sets=source_ids,
                )
                out_dir = os.path.join(args.output_dir, cond_name)
                out_path = os.path.join(out_dir, fname)
                os.makedirs(out_dir, exist_ok=True)
                write_score_file(out_path, augmented)
                matched_base = os.path.basename(baseline_file)
                print(f"    → {matched_base}  +  {len(augmented)} rows  → {out_path}")

    if not args.condition_dirs and all_ok:
        print("\nAll baseline files verified successfully.")


# ---------------------------------------------------------------------------
# Example programmatic usage (importable API)
# ---------------------------------------------------------------------------

def example_separate_baseline(
    baseline_path: str,
    asv19_protocol: str,
    asv21_la_protocol: str,
    asv21_df_protocol: str,
    asv5_protocol: str,
) -> Dict[str, List[Row]]:
    """
    Example: load a single baseline score file and split it into
    per-source buckets. Returns the bucket dict.

        buckets = example_separate_baseline(...)
        df_c1_rows = buckets['asv21_df_c1']   # 17 131 rows
        asv5_rows  = buckets['asv5_noc']       # 171 602 rows
        asv19_rows = buckets['asv19_eval']     #  71 237 rows
        asv21_rows = buckets['asv21_la_c1']    #  25 938 rows
    """
    source_ids = load_protocols(
        asv19_protocol=asv19_protocol,
        asv21_la_protocol=asv21_la_protocol,
        asv21_df_protocol=asv21_df_protocol,
        asv5_protocol=asv5_protocol,
    )
    rows = parse_score_file(baseline_path)
    return separate_by_source(rows, source_ids)


if __name__ == "__main__":
    main()
