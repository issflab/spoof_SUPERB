# 9. Analysis: tables and figures

Everything here consumes score files and produces CSVs, then figures from those
CSVs. Run them in dependency order -- the figure scripts do not recompute
metrics, they plot what the metric scripts wrote.

All analysis modules are run with `-m`:

```bash
python -m spoof_superb.analysis.<name> --help
```

Running them by file path (`python spoof_superb/analysis/create_heatmap.py`)
will fail with `ModuleNotFoundError`.

## The paper's main table

```bash
python -m spoof_superb.analysis.recompute_table5_mlaad_v10 --out_dir outputs/table5
```

See [reproducing results](02-reproducing-results.md) -- this is the authority
on what Tables 5 and 6 report.

## Acoustic degradation (Section 5.2)

```bash
python -m spoof_superb.analysis.compute_eer_matrix \
    --baseline_dir  $SCORES/Baseline_by_Hashim \
    --augmented_dir $SCORES/scores_by_acoustic_degradation \
    --output_csv    outputs/eer_matrix.csv

python -m spoof_superb.analysis.create_heatmap \
    --csv outputs/eer_matrix.csv --out_dir outputs/figures
```

Produces absolute and baseline-relative EER heatmaps.

Use `scores_by_acoustic_degradation/`, not `scores_by_category_augmented/`: the
latter carries the old fp16-NaN noise scores and no recompression.

## TTS diversity

The pipeline, in order:

```bash
# 1. build the directory -> canonical TTS system map (MLAAD v10 only)
python -m spoof_superb.analysis.build_mlaad_dir_map

# 2. fan score files out into per-TTS-system trees
python -m spoof_superb.analysis.organize_mlaad_scores
python -m spoof_superb.analysis.organize_tts_scores --help    # non-MLAAD sets

# 3. normalise (needed by the EER-vs-pooled-bonafide path)
python -m spoof_superb.analysis.apply_zscore_and_pool \
    --linear_head_dir ... --tts_dir ... --out_base ...

# 4. metrics
python -m spoof_superb.analysis.compute_eer_tts    --norm_dir ... --tts_dir ... --out_dir outputs/tts_eer
python -m spoof_superb.analysis.compute_far_matrix --tts_dir ... --combined_dir ... --out_dir outputs/tts_far
python -m spoof_superb.analysis.create_mlaad_tts_eer_heatmaps --out-dir outputs/mlaad_tts

# 5. figures
python -m spoof_superb.analysis.create_tts_eer_heatmaps --eer_dir outputs/tts_eer --out_dir outputs/figures
python -m spoof_superb.analysis.create_tts_heatmaps     --far_dir outputs/tts_far --out_dir outputs/figures
python -m spoof_superb.analysis.create_mlaad_group_ranked_figures
python -m spoof_superb.analysis.create_mlaad_tts_system_ranked_figure
```

`create_mlaad_tts_eer_heatmaps` is the one figure script that also *computes*
EERs, rather than reading a CSV.

Note on the aggregates: "Overall Mean" in the TTS matrices is a mean of
per-system means, not a pooled recomputation over utterances.

## Ad-hoc EER

```bash
# one score file
python -m spoof_superb.analysis.evaluate_score_file \
    --score_filepath out.txt --score_file_has_keys

# a directory: concatenate per SSL model, then evaluate
python -m spoof_superb.analysis.evaluate_score_directory \
    --input_dir DIR --output_dir OUT
```

## Score-file organisation utilities

```bash
python -m spoof_superb.analysis.combine_asvspoofld_scores --help
python -m spoof_superb.analysis.merge_filtered_scores --help
python -m spoof_superb.analysis.merge_asv21la_into_hashim_baseline --help
python -m spoof_superb.analysis.strip_bonafide_from_tts --help
python -m spoof_superb.analysis.apply_sigmoid_and_pool --help
```

## Diagnostics

```bash
python -m spoof_superb.analysis.layer_weight_analysis           # SSL layer weights
python -m spoof_superb.analysis.plot_score_distributions --help
python -m spoof_superb.analysis.analyze_benchmark_distribution --help
python -m spoof_superb.analysis.create_taxonomy                 # paper figure
python -m spoof_superb.analysis.create_SSL_taxonomy             # paper figure
```

## Two sharp edges

**Five modules have no CLI.** `layer_weight_analysis`,
`create_mlaad_group_ranked_figures`, `create_mlaad_tts_system_ranked_figure`,
`create_taxonomy` and `create_SSL_taxonomy` read constants defined at the top of
the file. Edit those constants to retarget them.

**Two write to a CWD-relative path.** `create_taxonomy` and
`create_SSL_taxonomy` write to `outputs/figures/...` relative to your current
directory, unlike every other figure script which anchors to the repo root. Run
them from the repo root.
