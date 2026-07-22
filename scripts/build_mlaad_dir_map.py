#!/usr/bin/env python3
"""Resolve every MLAAD v10 TTS directory name to a canonical TTS system.

MLAAD stores one directory per (language, TTS system) pair, and the directory
names are raw model identifiers (``facebook_mms-tts-deu``,
``tts_models_en_ljspeech_glow-tts``, ``Cartesia.ai (Sonic-3)``).  The published
taxonomy (``mlaad_v10_tts_architecture_groups.csv``) uses 94 canonical system
names.  This script builds the bridge between the two.

Five tiers are applied in order, first match wins; all comparisons are made on
a normalized key (lowercase, non-alphanumerics stripped):

    T1  raw directory name       == a canonical ``tts_system``
    T2  meta.csv ``architecture``== a canonical ``tts_system``
    T3  raw directory name       == a Table IV printed label
    T4  meta.csv ``architecture``== a Table IV printed label
    T5  explicit family table (FAMILY_MAP below)

After tier resolution the ``Dual-AR`` -> ``FishTTS`` alias is applied
(methodology 1.3: Dual-AR is the dual-autoregressive architecture of the
Fish-Speech paper, i.e. the same system as FishTTS), and the three
non-TTS directories are marked EXCLUDED (methodology 1.2 / 1.3).

Output: ``scripts/mlaad_v10_dir_to_system.csv`` with columns
``raw_dir, canonical_system, tier, evidence, n_utts, excluded_reason``.

Usage
-----
    python3 scripts/build_mlaad_dir_map.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SCORE_ROOT = Path("/data/ssl_anti_spoofing/asd_superb_score_files")
TSV_DIR = SCORE_ROOT / "linear_head_MLAAD_v10" / "tsv"
MLAAD_ROOT = Path("/data/Data/MLAAD")

ARCH_CSV = SCORE_ROOT / "mlaad_v10_tts_architecture_groups.csv"
PROVENANCE_CSV = SCORE_ROOT / "mlaad_v10_table4_provenance.csv"

DEFAULT_OUT = Path(__file__).resolve().parent / "mlaad_v10_dir_to_system.csv"

# --- Tier 5: directories with no automatic match ----------------------------
# Keyed by raw directory name; regex families are handled separately below.
FAMILY_EXACT = {
    "tts_models_en_sam_tacotron-DDC": "Tacotron",
    "parler_tts_large_v1": "Parler-TTS",
    "parler_tts_mini_v0.1": "Parler-TTS",
    "parler_tts_mini_v1": "Parler-TTS",
    "suno_bark": "Bark",
    "suno_bark-small": "Bark",
    "tts_models_bn_custom_vits-female": "VITS",
    "tts_models_bn_custom_vits-male": "VITS",
    "sesame_csm": "Sesame-CSM-1B",
    "microsoft_speecht5_tts": "SpeechT5",
    "orpheus-tts-0.1-finetune": "Orpheus-TTS",
    "tts_models_en_blizzard2013_capacitron-t2-c50": "Capacitron",
    "tts_models_en_multi-dataset_tortoise-v2": "Tortoise",
    "Spark-TTS-0.5B": "Spark-TTS",
    "Llasa-1B-Multilingual": "Llasa",
}

FAMILY_PATTERNS = [
    (re.compile(r"^facebook_mms-tts-.+$"), "VITS-MMS"),
    (re.compile(r"^tts_models_.*_tacotron2-.+$"), "Tacotron2"),
]

# --- Alias applied after tier resolution (methodology 1.3) ------------------
ALIASES = {"Dual-AR": "FishTTS"}

# --- Exclusions (methodology 1.2 / 1.3), keyed by raw directory name --------
EXCLUSIONS = {
    "griffin_lim": "phase-reconstruction vocoder, not a TTS system (methodology 1.2)",
    "RVC": "voice-conversion system, not TTS (methodology 1.3)",
    "Voxtral": "audio understanding model, no speech-generation stage (methodology 1.3)",
}

EXPECTED_TOTAL_SPOOF = 456_000
EXPECTED_EXCLUDED = {"griffin_lim": 8_000, "RVC": 8_000, "Voxtral": 9_000}
EXPECTED_RETAINED = 431_000


def norm(text: str) -> str:
    """Case- and punctuation-insensitive comparison key."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_utt_index(tsv_path: Path) -> pd.DataFrame:
    """Return spoof rows with language / raw_dir columns parsed from utt_id.

    The tsv MUST be read with an explicit tab separator: ~8.6% of utt_ids
    contain literal spaces (``Cartesia.ai (Sonic-3)``, ``OpenAI TTS-1 HD``).
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    spoof = df[df["label"] == "spoof"].copy()
    parts = spoof["utt_id"].str.split("/", expand=True)
    if parts.shape[1] != 5:
        sys.exit(f"FATAL: utt_id does not have 5 segments in {tsv_path}")
    spoof["language"] = parts[2]
    spoof["raw_dir"] = parts[3]
    return spoof[["utt_id", "language", "raw_dir"]]


def read_meta_architecture(mlaad_root: Path) -> dict[str, set[str]]:
    """Map raw directory name -> set of ``architecture`` values in its meta.csv.

    A directory name recurs across languages; each occurrence has its own
    meta.csv.  Divergent values across languages are reported, never merged
    silently.
    """
    arch: dict[str, set[str]] = defaultdict(set)
    for meta_path in sorted((mlaad_root / "fake").glob("*/*/meta.csv")):
        with meta_path.open("r", encoding="utf-8", newline="") as handle:
            # QUOTE_NONE: MLAAD transcripts contain bare `"` characters that
            # would otherwise swallow newlines and merge records.
            reader = csv.DictReader(handle, delimiter="|", quoting=csv.QUOTE_NONE)
            first = next(reader, None)
        if first is None:
            continue
        value = (first.get("architecture") or "").strip()
        if value:
            arch[meta_path.parent.name].add(value)
    return arch


def resolve(
    raw_dirs: list[str],
    arch_by_dir: dict[str, set[str]],
    canonical: list[str],
    table4: dict[str, str],
) -> dict[str, tuple[str, str, str]]:
    """Return raw_dir -> (canonical_system, tier, evidence)."""
    canon_by_key = {norm(name): name for name in canonical}
    table4_by_key = {norm(label): canon for label, canon in table4.items()}

    resolved: dict[str, tuple[str, str, str]] = {}
    for raw in raw_dirs:
        archs = sorted(arch_by_dir.get(raw, set()))
        arch_one = archs[0] if len(archs) == 1 else ""

        hit = canon_by_key.get(norm(raw))
        if hit:
            resolved[raw] = (hit, "T1", f"raw_dir=={hit}")
            continue

        if arch_one and (hit := canon_by_key.get(norm(arch_one))):
            resolved[raw] = (hit, "T2", f"meta.architecture='{arch_one}'=={hit}")
            continue

        if hit := table4_by_key.get(norm(raw)):
            resolved[raw] = (hit, "T3", f"raw_dir==TableIV label -> {hit}")
            continue

        if arch_one and (hit := table4_by_key.get(norm(arch_one))):
            resolved[raw] = (
                hit,
                "T4",
                f"meta.architecture='{arch_one}'==TableIV label -> {hit}",
            )
            continue

        if raw in FAMILY_EXACT:
            resolved[raw] = (FAMILY_EXACT[raw], "T5", "family table (exact)")
            continue

        for pattern, target in FAMILY_PATTERNS:
            if pattern.match(raw):
                resolved[raw] = (target, "T5", f"family table ({pattern.pattern})")
                break

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv-dir", type=Path, default=TSV_DIR)
    parser.add_argument("--mlaad-root", type=Path, default=MLAAD_ROOT)
    parser.add_argument("--arch-csv", type=Path, default=ARCH_CSV)
    parser.add_argument("--provenance-csv", type=Path, default=PROVENANCE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tsvs = sorted(args.tsv_dir.glob("*.tsv"))
    if not tsvs:
        sys.exit(f"FATAL: no tsv files under {args.tsv_dir}")
    index = load_utt_index(tsvs[0])
    counts = Counter(index["raw_dir"])
    total_spoof = int(sum(counts.values()))
    print(f"Reference tsv: {tsvs[0].name}")
    print(f"  spoof rows: {total_spoof}   distinct dirs: {len(counts)}")

    if total_spoof != EXPECTED_TOTAL_SPOOF:
        sys.exit(f"FATAL: expected {EXPECTED_TOTAL_SPOOF} spoof rows, got {total_spoof}")

    arch_df = pd.read_csv(args.arch_csv)
    canonical = arch_df["tts_system"].tolist()
    prov_df = pd.read_csv(args.provenance_csv)
    table4 = dict(
        zip(prov_df["table4_label_as_printed"], prov_df["canonical_name"])
    )

    arch_by_dir = read_meta_architecture(args.mlaad_root)
    divergent = {d: v for d, v in arch_by_dir.items() if len(v) > 1 and d in counts}
    if divergent:
        print(f"  [WARN] {len(divergent)} dirs have divergent meta architecture values:")
        for d, v in sorted(divergent.items()):
            print(f"         {d}: {sorted(v)}")

    resolved = resolve(sorted(counts), arch_by_dir, canonical, table4)

    unresolved = sorted(set(counts) - set(resolved))
    if unresolved:
        print(f"\nFATAL: {len(unresolved)} directories unresolved after T1-T5:")
        for d in unresolved:
            print(f"  {d}  (n_utts={counts[d]}, arch={sorted(arch_by_dir.get(d, []))})")
        sys.exit(1)

    rows = []
    for raw in sorted(counts):
        system, tier, evidence = resolved[raw]
        if system in ALIASES:
            evidence = f"{evidence}; alias {system}->{ALIASES[system]} (methodology 1.3)"
            system = ALIASES[system]
        excluded_reason = ""
        if raw in EXCLUSIONS:
            excluded_reason = EXCLUSIONS[raw]
            evidence = f"{evidence}; excluded"
            system = "EXCLUDED"
        rows.append(
            {
                "raw_dir": raw,
                "canonical_system": system,
                "tier": tier,
                "evidence": evidence,
                "n_utts": counts[raw],
                "excluded_reason": excluded_reason,
            }
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # --- verification -------------------------------------------------------
    excluded = out[out["canonical_system"] == "EXCLUDED"]
    retained = out[out["canonical_system"] != "EXCLUDED"]
    n_excluded = int(excluded["n_utts"].sum())
    n_retained = int(retained["n_utts"].sum())

    got_excluded = dict(zip(excluded["raw_dir"], excluded["n_utts"].astype(int)))
    assert got_excluded == EXPECTED_EXCLUDED, (
        f"exclusion counts mismatch: {got_excluded} != {EXPECTED_EXCLUDED}"
    )
    assert n_retained + n_excluded == total_spoof
    assert n_retained == EXPECTED_RETAINED, f"retained {n_retained} != {EXPECTED_RETAINED}"

    systems = sorted(retained["canonical_system"].unique())
    unknown = [s for s in systems if s not in set(canonical)]
    assert not unknown, f"invented canonical names: {unknown}"

    print("\nTier distribution:")
    for tier, n in sorted(Counter(out["tier"]).items()):
        print(f"  {tier}: {n} dirs")

    print("\nExclusions (methodology 1.2 / 1.3):")
    for _, r in excluded.iterrows():
        print(f"  {r['raw_dir']}: {r['n_utts']} utts — {r['excluded_reason']}")

    missing = sorted(set(canonical) - set(systems) - {"Dual-AR", "Griffin Lim"})
    print(f"\nCanonical systems in taxonomy : {len(canonical)}")
    print(f"Canonical systems with utts   : {len(systems)}")
    if missing:
        print(f"Protocol systems with zero utterances: {missing}")

    print(
        f"\n✅ STEP 1 — {args.output} — "
        f"{len(out)} dirs, {len(systems)} canonical systems, "
        f"{n_retained} retained + {n_excluded} excluded = {total_spoof} spoof utts"
    )


if __name__ == "__main__":
    main()
