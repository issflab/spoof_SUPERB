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

## Which tree a script reads

Every script that reads RAW score files takes `--scores_root` and `--layout`,
and resolves paths through `core.scorepath`. Omit them and the configured tree
is used.

```bash
python -m spoof_superb.analysis.recompute_main_results \
    --scores_root /data/ssl_anti_spoofing/spoof_superb_score_files --layout v3
```

Scripts that read DERIVED views -- the per-condition and per-TTS trees -- take
an explicit `--*_dir` instead, because a view is a directory somebody built, not
a path the layout can compute. Those are marked below.

| reads | scripts |
|---|---|
| raw, and builds its own view | `acoustic_degradation`, `tts_systems`, `build_view` |
| raw score files | `recompute_main_results`, `verify_mlaad_column`, `organize_mlaad_scores`, `build_mlaad_dir_map`, `compare_trees`, `verification.scores` |
| raw, then writes a legacy tree | `apply_zscore_and_pool`, `apply_sigmoid_and_pool`, `plot_score_distributions` |
| a LEGACY view tree | `compute_eer_matrix`, `compute_eer_tts`, `compute_far_matrix`, `create_*_heatmaps` |
| neither (checkpoints, corpus metadata) | `layer_weight_analysis`, `create_combined_mlaad_meta_all` |

The last group predates the entry points above and reads view trees that exist
only in the legacy tree. They are kept because they produced published figures,
not because anything should be run through them now.

## The three analyses

Each is one command. Each builds the grouping it reports over, so a number and
the view behind it cannot disagree.

```bash
# the paper's main table
python -m spoof_superb.analysis.recompute_main_results --out_dir outputs/main_results

# Section 4.4.2 -- acoustic degradation
python -m spoof_superb.analysis.acoustic_degradation --out_dir outputs/degradation

# Sections 4.4.3 / 3.2.3 -- TTS systems
python -m spoof_superb.analysis.tts_systems --out_dir outputs/tts
```

The latter two write their view under `{scores_root}/views/` as they go. Use
`--out_root` to put it somewhere else, and `--scores_root` / `--layout` to read
a different tree.

**Where the figures go.** `--out_dir` on any analysis, or set `outputs_root` in
`configs/paths.yaml` to move all three at once -- each writes a sub-directory
under it (`main_results/`, `degradation/`, `tts/`). Unset means the repo's own
`outputs/`, which is what they did before the setting existed.

```yaml
# configs/paths.yaml
outputs_root: /data/ssl_anti_spoofing/spoof_superb_outputs
```

**The figures are the published ones.** Both analyses hand off to the modules
that drew the paper's figures rather than plotting their own: degradation to
`create_heatmap`, TTS to `create_mlaad_group_ranked_figures` and
`create_mlaad_tts_system_ranked_figure`. So the degradation heatmaps carry the
Baseline and Mean columns in grey outside the colour scale, the rows follow
Table 6's order with rules between generative / discriminative / spectrogram,
and the TTS figures show the six representative models with columns and systems
ranked by their Mean, the 91 systems split across two panels.

### Acoustic degradation

Six conditions -- one clean reference and five degraded -- each POOLED from
partitions of four corpora, exactly as `tab:acoustic_degradation` specifies:

| condition | composition |
|---|---|
| Baseline | ASV19 LA eval + ASV21 LA:C1 + ASV21 DF:C1 + ASV5:C00 |
| Codec & Compression | ASVLD (recompr.) + DF:C2--C9 + ASV5:C01--C10 + **LA:C1** |
| Bandwidth | ASVLD (resampled) + **LA:C1** + **DF:C1** + **ASV5:C00** |
| Additive Noise | ASVLD (noise) + **LA:C1** + **DF:C1** + **ASV5:C00** |
| Reverberation | ASVLD (reverb) + **LA:C1** + **DF:C1** + **ASV5:C00** |
| Channel Distortions | LA:C2--C7 + ASV5:C11 + **DF:C1** + **ASV19 LA eval** |

Bold entries are retained unchanged from the Baseline. That retention is the
point: each condition changes only the corpus under degradation, so its EER is
comparable to the reference. Measuring a degradation on its degraded corpus
alone would confound the degradation with that corpus's own difficulty.

Reports absolute EER per condition and the relative change against the
Baseline, `dEER = (EER_deg - EER_clean) / EER_clean`.

Condition codes come from each corpus's own protocol, resolved in
`analysis/conditions.py`. Only the clean partition is load-bearing -- C1 is
`none` for ASV21 LA, `nocodec` for ASV21 DF, and C00 is `-` for ASVspoof 5 --
because every other condition enters as "all the rest" and so cannot drift if a
corpus gains one. ASVLD is the exception: its condition is in the utt_id.

### TTS systems

MLAAD v10 only, 91 systems after merging Dual-AR into FishTTS and excluding
three non-TTS entries (griffin_lim, RVC, Voxtral), leaving 431,000 spoof
utterances. Four groupings, each a heatmap and a CSV: by system, by the 11
architecture groups, by generation mode, by vocoder family.

Only the system is in the view. Architecture, mode and vocoder family are
functions of the system, so they are grouped up at analysis time -- a tree that
hard-coded them would need rebuilding whenever the taxonomy changed, and would
let tree and table disagree meanwhile.

Every group at every level is scored against the same pooled M-AILABS bonafide
reference (584,006 utterances). That is why this analysis is restricted to
MLAAD: every system synthesises from the same bonafide source, so a per-system
EER measures the synthesis system rather than its source corpus.

Coarse levels pool the systems' spoof scores and recompute; they do not average
per-system EERs, which would weight a 1,000-utterance system the same as a
34,000-utterance one.

### The view tree

```
views/{view}/{group}/[{subgroup}/]{frontend}.txt
views/{view}/_bonafide/{frontend}.txt      shared reference pool
views/{view}/_manifest.json                sources, row counts, build time
```

The frontend is always the last component, so `views/*/*/xls_r_300m.txt` finds
one model everywhere -- the same property `raw/` has. To build a view without
running an analysis, or to see what one would write:

```bash
python -m spoof_superb.tools.build_view --view tts_systems
python -m spoof_superb.tools.build_view --view acoustic_degradation --dry-run
```

Each view also carries a `README.md` saying, in sentences, what every group
contains -- which corpus partitions compose each degradation condition, and
which condition codes those are in the corpus's own numbering. `_manifest.json`
is the machine-readable record beside it.

**Superseded.** `compute_eer_matrix` and `compute_eer_tts` read the legacy view
trees and predate these entry points. `create_mlaad_tts_eer_heatmaps` still
computes the four base TTS matrices from raw; `tts_systems` reproduces it
exactly (max|d| = 0.000000 across all 11 architecture groups) and additionally
produces the ranked figures.

## Comparing two score trees

```bash
python -m spoof_superb.tools.compare_trees \
    --a /data/ssl_anti_spoofing/asd_superb_score_files   --a-layout legacy \
    --b /data/ssl_anti_spoofing/spoof_superb_score_files --b-layout v3 \
    --out outputs/tree_comparison "--a-id-rewrite=-=Bonafide"
```

Reports, per (dataset, model) cell, whether the two trees differ because they
scored different utterances or because they assigned different scores to the
same ones. Only the second is a reproducibility problem, and a single EER delta
cannot tell them apart -- so the tool computes both trees' EER restricted to the
utterances they share.

`--a-id-rewrite` renames whole path components before matching. Famous Figures
needs it: the old tree names the bonafide directory `-`, the new one names it
`Bonafide`. It is deliberately not automatic -- asserting that two id
conventions denote the same utterances is a claim the caller makes.

## The paper's main table

```bash
python -m spoof_superb.analysis.recompute_main_results --out_dir outputs/main_results
```

See [reproducing results](02-reproducing-results.md) -- this is the authority
on what the two results tables report.

## The legacy chains

These predate the entry points above and read the legacy view trees. Kept
because they produced published figures; superseded for new work.

```bash
# acoustic degradation, from the legacy scores_by_acoustic_degradation tree
python -m spoof_superb.analysis.compute_eer_matrix \
    --baseline_dir  $SCORES/Baseline_by_Hashim \
    --augmented_dir $SCORES/scores_by_acoustic_degradation \
    --output_csv    outputs/eer_matrix.csv
python -m spoof_superb.analysis.create_heatmap --csv outputs/eer_matrix.csv --out_dir outputs/figures

# TTS, in order: dir map, fan out, normalise, metrics, figures
python -m spoof_superb.analysis.build_mlaad_dir_map
python -m spoof_superb.analysis.organize_mlaad_scores
python -m spoof_superb.analysis.apply_zscore_and_pool --linear_head_dir ... --tts_dir ... --out_base ...
python -m spoof_superb.analysis.compute_eer_tts    --norm_dir ... --tts_dir ... --out_dir outputs/tts_eer
python -m spoof_superb.analysis.compute_far_matrix --tts_dir ... --combined_dir ... --out_dir outputs/tts_far
python -m spoof_superb.analysis.create_tts_eer_heatmaps --eer_dir outputs/tts_eer --out_dir outputs/figures
python -m spoof_superb.analysis.create_mlaad_group_ranked_figures
python -m spoof_superb.analysis.create_mlaad_tts_system_ranked_figure
```

Use `scores_by_acoustic_degradation/`, not `scores_by_category_augmented/`: the
latter carries the old fp16-NaN noise scores and no recompression. That two
trees hold the same view built twice, with this line left to say which to
trust, is the drift the new views exist to prevent.

Note on the aggregates: "Overall Mean" in the legacy TTS matrices is a mean of
per-system means, not a pooled recomputation. `tts_systems` pools instead.

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
