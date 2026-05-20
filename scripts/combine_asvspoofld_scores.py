from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


SOURCE_DATASET_DIRS = [
    "Additive_Noise",
    "Reverberation",
    "Resampling",
]
TARGET_DATASET_NAME = "asvspoofLD"
OUTPUT_PREFIX = "linear_head"
VALID_SCORE_SUFFIXES = {".txt", ".tsv", ".score", ""}
RESAMPLING_PREFIX = "linear_head_resamp_"
MODEL_NAME_ALIASES = {
    "audioalbert": "audioalbert960hr",
    "byolaudio": "byola2048",
    "data2vec": "data2veclargell60k",
    "hubertlarge": "hubertlargell60k",
    "mrhubert": "multireshubertmultilinguallarge600k",
    "ssast": "ssastframebase",
    "unispeechsat": "unispeechsatlarge",
    "wav2vec2base": "wav2vec2base960",
    "wav2vec2large": "wav2vec2largell60k",
    "wavlablm": "wavlablmek40k",
    "xlsr": "xlsr300m",
}
MODEL_ID_TO_OUTPUT_NAME = {
    "apc": "apc",
    "audioalbert960hr": "audio_albert_960hr",
    "byola2048": "byol_a_2048",
    "data2veclargell60k": "data2vec_large_ll60k",
    "decoar2": "decoar2",
    "fbank": "fbank",
    "hubertbase": "hubert_base",
    "hubertlargell60k": "hubert_large_ll60k",
    "maeastframe": "mae_ast_frame",
    "mockingjay": "mockingjay",
    "mockingjay960hr": "mockingjay_960hr",
    "modifiedcpc": "modified_cpc",
    "multireshubertmultilinguallarge600k": "multires_hubert_multilingual_large600k",
    "npc": "npc",
    "ssastframebase": "ssast_frame_base",
    "tera": "tera",
    "unispeechsatlarge": "unispeech_sat_large",
    "vqapc": "vq_apc",
    "wav2vec": "wav2vec",
    "wav2vec2base960": "wav2vec2_base_960",
    "wav2vec2largell60k": "wav2vec2_large_ll60k",
    "wavlablmek40k": "wavlablm_ek_40k",
    "wavlmlarge": "wavlm_large",
    "xlsr300m": "xls_r_300m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine score files from Additive_Noise, Reverberation, and Resampling "
            "into one per-SSL-model score file for the asvspoofLD dataset."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/data/ssl_anti_spoofing/asd_superb_score_files/scores_by_category"),
        help="Root directory containing the source dataset subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where combined per-model score files will be written.",
    )
    return parser.parse_args()


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


def strip_known_prefixes(stem: str) -> str:
    if stem.startswith(RESAMPLING_PREFIX):
        return stem[len(RESAMPLING_PREFIX) :]
    return stem


def canonical_model_id(score_path: Path) -> str:
    raw_name = strip_known_prefixes(score_path.stem)
    normalized = "".join(ch.lower() for ch in raw_name if ch.isalnum())
    return MODEL_NAME_ALIASES.get(normalized, normalized)


def collect_source_files(
    input_root: Path, source_dataset_dirs: list[str]
) -> dict[str, dict[str, Path]]:
    files_by_dataset_and_model: dict[str, dict[str, Path]] = {}

    for dataset_dir_name in source_dataset_dirs:
        dataset_dir = input_root / dataset_dir_name
        if not dataset_dir.is_dir():
            raise NotADirectoryError(f"Source dataset directory not found: {dataset_dir}")

        model_paths: dict[str, Path] = {}
        for score_path in iter_score_files(dataset_dir):
            model_id = canonical_model_id(score_path)
            if model_id in model_paths:
                raise ValueError(
                    f"Duplicate model mapping in {dataset_dir}: "
                    f"{model_paths[model_id].name} and {score_path.name}"
                )
            model_paths[model_id] = score_path

        files_by_dataset_and_model[dataset_dir_name] = model_paths

    return files_by_dataset_and_model


def choose_output_model_names(
    files_by_dataset_and_model: dict[str, dict[str, Path]],
    source_dataset_dirs: list[str],
) -> dict[str, str]:
    chosen_names: dict[str, str] = {}

    for dataset_dir_name in source_dataset_dirs:
        for model_id, score_path in files_by_dataset_and_model[dataset_dir_name].items():
            chosen_names.setdefault(
                model_id,
                MODEL_ID_TO_OUTPUT_NAME.get(model_id, strip_known_prefixes(score_path.stem).lower()),
            )

    return chosen_names


def group_models(
    files_by_dataset_and_model: dict[str, dict[str, Path]],
    source_dataset_dirs: list[str],
) -> tuple[dict[str, list[Path]], dict[str, list[str]]]:
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    missing_sources: dict[str, list[str]] = {}

    all_model_ids = sorted(
        {
            model_id
            for dataset_files in files_by_dataset_and_model.values()
            for model_id in dataset_files
        }
    )

    for model_id in all_model_ids:
        missing = [
            dataset_dir_name
            for dataset_dir_name in source_dataset_dirs
            if model_id not in files_by_dataset_and_model[dataset_dir_name]
        ]
        if missing:
            missing_sources[model_id] = missing

        for dataset_dir_name in source_dataset_dirs:
            score_path = files_by_dataset_and_model[dataset_dir_name].get(model_id)
            if score_path is not None:
                grouped_paths[model_id].append(score_path)

    return dict(grouped_paths), missing_sources


def combine_score_files(source_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for source_path in source_paths:
            content = source_path.read_text(encoding="utf-8").rstrip()
            if not content:
                continue
            output_handle.write(content)
            output_handle.write("\n")


def combine_datasets(input_root: Path, output_dir: Path) -> None:
    files_by_dataset_and_model = collect_source_files(
        input_root=input_root,
        source_dataset_dirs=SOURCE_DATASET_DIRS,
    )
    grouped_paths, missing_sources = group_models(
        files_by_dataset_and_model=files_by_dataset_and_model,
        source_dataset_dirs=SOURCE_DATASET_DIRS,
    )
    model_names = choose_output_model_names(
        files_by_dataset_and_model=files_by_dataset_and_model,
        source_dataset_dirs=SOURCE_DATASET_DIRS,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_id in sorted(grouped_paths):
        model_name = model_names[model_id]
        output_path = output_dir / f"{OUTPUT_PREFIX}_{TARGET_DATASET_NAME}_{model_name}.txt"
        combine_score_files(grouped_paths[model_id], output_path)
        print(
            f"[INFO] Wrote {output_path.name} from {len(grouped_paths[model_id])} source files"
        )

    if missing_sources:
        print("[WARN] Some SSL models were not present in all source datasets:")
        for model_id in sorted(missing_sources):
            print(
                f"[WARN] model={model_names[model_id]} missing_from={','.join(missing_sources[model_id])}"
            )


def main() -> None:
    args = parse_args()
    combine_datasets(
        input_root=args.input_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
