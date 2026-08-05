# Cleanup ledger

Working list for trimming the repo before release. Delete this file when the
list is empty.

Method: every `.py` under `spoof_superb/` was walked as an import graph from the
entry points the README's four paths actually invoke (install / reproduce /
rebuild / train), then each unreachable module was classified by whether it has
a CLI, whether any doc names it, and whether any other module references it.
**"Unreachable" does not mean "dead"** -- most of these are standalone `python -m`
tools, which is why the tiers below exist.

91 modules; 51 reachable by import; 40 not. Of those 40, most are legitimate
entry points.

Status key: `[ ]` open · `[x]` done · `[?]` needs a decision from Hashim

---

## Done

- [x] `core_scripts/` -- 38 files, 336 KB vendored from project-NN-Pytorch-scripts.
      One function reachable, reimplemented as `core.utils.set_seed`, verified
      EQUIVALENT before deleting. Also a licensing problem: BSD-3-Clause code with
      no license file in an Apache-2.0 repo. (P16)
- [x] `verification/driver.py`, `policies.py`, `stats.py`, `tests/test_verification.py`
- [x] `bin/watch_and_run_spoofceleb.sh`
- [x] `tests/test_main_results_regression.py` (P15)
- [x] **Tier 1** -- `analysis/create_tts_heatmaps.py`, `data/prep/make_tsv_mlaad.py`,
      `data/prep/report_mlaad.py`, and the four legacy-tree `bin/` one-offs
      (`run_asvld_model.sh`, `run_noise_rerun.sh`, `run_recompression.sh`,
      `watch_and_run_aasist_mlaad.sh`)
- [x] **`sls` removed entirely** -- `models/sls.py` plus the `--model_arch`
      choice, the `config.py` Literal, the unreachable dispatch branch in
      `main.py`, and the mentions in `bin/train.sh`, `docs/07`, `README.md`.
      The architecture had never been runnable: its import was commented out.

---

## Defects found while auditing -- fix or decide, do not just delete

- [x] **Roster mapping extracted.** The live 3% of
      `tests/baseline_main_results_table.json` -- 21 display-name -> slug
      entries, 672 bytes of a 21 KB file -- is now
      `spoof_superb/scoring/paper_roster.json`. `models.py` reads it via
      `PAPER_ROSTER`. The rest was legacy-tree EERs plus a `reproduction_failures`
      key from the deleted gate.

      The old file is still on disk, UNTRACKED, and there is no git history for
      it -- it was never committed. Deleting it is irreversible, so it was left
      alone. Archive it somewhere outside the repo if the legacy numbers matter;
      otherwise `rm tests/baseline_main_results_table.json`.

- [x] **`seed_worker` deleted.** The ledger entry was WRONG: it claimed the
      DataLoader workers were not deterministically seeded. PyTorch's own
      `_worker_loop` has seeded `random`, `torch` and `numpy` per worker since
      ~1.9, and this repo pins torch 2.7.1. Measured with a dataset drawing
      `np.random` in `__getitem__` exactly as `datasets_ssl.py` does under
      RawBoost (`--algo` defaults to 5): 16/16 distinct draws, and the same seed
      reproduces them across runs. The helper was redundant, not unwired -- no
      behaviour change, no decision needed.

- [x] **`.gitignore` was excluding the package from the repository.** Found while
      moving the roster: `git rm` on the baseline JSON failed with "pathspec did
      not match", because `*.json` had kept it out of git entirely. Widening the
      check found `_*` at line 2 -- unscoped, so matching at every depth --
      excluding all 12 `spoof_superb/**/__init__.py`, `verification/__main__.py`
      and `bin/_common.sh`, plus `models/` excluding
      `spoof_superb/models/__init__.py`. **A fresh clone could not import the
      package, run any `bin/` script, or resolve the roster.** Rules root-scoped
      and re-includes added; verified by cloning: imports, 250 tests pass,
      `bin/orchestrate.sh --list` runs.

---

## Tier 1 -- done

Nothing imported them, no doc names them as current, and no committed data
depends on them.

- [x] `spoof_superb/analysis/create_tts_heatmaps.py` (213 loc) -- not in any doc,
      zero references, superseded by `create_mlaad_tts_eer_heatmaps`
- [x] `spoof_superb/data/prep/make_tsv_mlaad.py` (61 loc) -- no CLI, no doc, no refs
- [x] `spoof_superb/data/prep/report_mlaad.py` (123 loc) -- no CLI, no doc, no refs
- [x] `bin/run_asvld_model.sh`, `bin/run_noise_rerun.sh`,
      `bin/run_recompression.sh`, `bin/watch_and_run_aasist_mlaad.sh` -- one-off
      operational scripts for completed work; every path reference points at the
      **legacy** score tree

### Withdrawn from Tier 1 -- two entries above were wrong

Both were listed as clutter to remove. Checking `git ls-files` and
`git check-ignore` shows neither is in the repository at all, so deleting them
buys nothing at release and only destroys local work.

- **`outputs/` (170 MB) and `spoof_superb_outputs/` (7.4 MB)** -- **0 tracked
  files**; both are gitignored. `spoof_superb_outputs/` is the configured
  `outputs_root` and holds the tables `reference/analysis/` was built from, plus
  every figure and both verification reports. `outputs/` holds the published
  figures. Deleting them would be pure local data loss. Prune stale
  sub-directories by hand if disk matters; nothing needs doing for the release.
- **root PDF** -- untracked, matched by `.gitignore:156 *.pdf`. It is a local
  copy of the paper, not something the repo ships.

---

## Done -- the legacy and v2 layouts are retired

- [x] `LAYOUTS` collapsed to one. `--layout` / `--a-layout` / `--b-layout` /
      `--candidate-layout` / `--ref-layout` removed from **13** commands, the
      `score_layout` setting removed from `config.py` and `configs/paths.yaml`,
      and every `layout=` parameter dropped from the call chain.
- [x] `tools/migrate_layout.py` and `tests/test_migrate_layout.py` deleted -- its
      only job was v2 -> v3.
- [x] The DFEval knowledge that lived in the `legacy` branch of `layout_key` was
      preserved as prose on `COLUMN_KEYS`: the retired tree scored the
      unsegmented set (1,980 trials, one 4 s window per recording), this one
      scores every window (56,481), and those are different measurements. The
      function is now `column_key(dataset)` with no layout argument.

**Layout is not format.** `core/scorefile.py` still reads all three on-disk
shapes -- 4-column space, 4-column tab, 3-column tab with a header -- because
the v3 tree carries a `.tsv` twin beside every `.txt`. None of that changed.

This forces RP-5 (see `docs/internal/humanpending.md`): nothing in the repo can
read the 49 GB legacy tree any more, so keeping it is keeping bytes with no
reader. Archive it or delete it, but do not leave it in between.

---

## Pending the paper update

The paper is being revised to report the **segmented** Deepfake-Eval column. The
pipeline already does -- `reference/analysis/` carries n=56,481 -- so no number
moves. What goes stale is the prose that describes the segmented set as *not*
what the paper reports:

- [ ] `scoring/datasets.py:336` -- "NOT comparable to the published DFEval24
      column (n=1,976)"
- [ ] `analysis/recompute_main_results.py:111` -- "1,976 published trials"
- [ ] `core/scorepath.py` -- the DATASET_KEY_BY_LAYOUT note
- [ ] `docs/04-datasets.md:324`, `docs/05-scoring.md:230`

Each should become "an earlier draft reported the unsegmented column (1,976
trials); the current paper reports the segmented one". The measurement contrast
stays -- the two EERs are still different quantities -- only which one the paper
prints changes.

`test_d5` was already rewritten to assert the RESOLUTION rather than the printed
trial count, so it holds under both drafts.

Note the 1,976 vs 1,980 gap is separate and already explained: the published run
could not read four `.dat` files that are really MP4 containers; ffmpeg recovers
them, so the corpus is 1,980. See `data/prep/segment_deepfake_eval.py`.

---

## Tier 2 -- done, except the committed-data group

Deleted: 14 modules, 4,784 lines. Nothing imported any of them -- checked by
grep for imports across the package, `main.py` and `tests/` before removing.

**The legacy TTS chain** -- replaced by `analysis/tts_systems.py` + `views.py`:
- [x] `compute_eer_tts.py` (531), `compute_far_matrix.py` (502),
      `create_tts_eer_heatmaps.py` (275), `apply_zscore_and_pool.py` (347),
      `apply_sigmoid_and_pool.py` (232), `organize_tts_scores.py` (631),
      `organize_mlaad_scores.py` (205), `strip_bonafide_from_tts.py` (97)
- The z-score/sigmoid poolers existed only for normalised scores, which
  P11-D8 established this project does not use.

**The legacy degradation chain** -- replaced by `analysis/acoustic_degradation.py`:
- [x] `compute_eer_matrix.py` (233), `verify_and_split_condition_scores.py` (625),
      `check_condition_composition.py` (311). Condition composition is now a
      property of `views.py` + `conditions.py`, built from the protocols, rather
      than something asserted after the fact.

**One-off score-file surgery on the legacy tree:**
- [x] `merge_filtered_scores.py` (425), `merge_asv21la_into_hashim_baseline.py` (138),
      `combine_asvspoofld_scores.py` (232)

**Near-miss worth recording.** `create_tts_eer_heatmaps` (deleted) and
`create_mlaad_tts_eer_heatmaps` (LIVE -- `tts_systems` imports `plot_heatmap`
and `auc_bonafide_vs_spoof` from it) differ by one word. The delete list was
checked against the live twins by name before running.

### Kept -- generates committed data, or still has a consumer

- [ ] `analysis/build_mlaad_dir_map.py` (313) writes
      `mlaad_v10_dir_to_system.csv` and `mlaad_v10_table4_provenance.csv`,
      which `views.py` and `tts_systems.py` READ. It is the provenance of live
      committed data, not a superseded step.
- [ ] `analysis/create_combined_mlaad_meta.py` (111),
      `create_combined_mlaad_meta_all.py` (145) -- MLAAD metadata assembly.
- [ ] `analysis/verify_tts_protocols.py` (596) -- the one remaining consumer of
      `create_combined_mlaad_meta.py`'s output.

---

## Tier 3 -- keep

Unreachable by import, but each is a documented `python -m` entry point that a
README path needs.

| Module | Needed for |
|---|---|
| `data/prep/{append_mailabs,build_protocols,balance_mailabs,segment_deepfake_eval}` | path B: preparing corpora |
| `tools/migrate_layout` | moving an old score tree to v3 |
| `tools/compare_trees` | ad-hoc two-tree comparison (docs/08) |
| `tools/select_aasist_ckpt` | checkpoint selection |
| `verification/{noise_rerun_gate,asvld_report}` | the fp32 promotion gate, ASVLD table |
| `analysis/{evaluate_score_file,evaluate_score_directory}` | ad-hoc EER |
| `analysis/verify_mlaad_column` | independent cross-check of the paper's MLAAD column |
| `analysis/{plot_score_distributions,analyze_benchmark_distribution,layer_weight_analysis}` | diagnostics |
| `analysis/{create_taxonomy,create_SSL_taxonomy}` | paper figures |

Three of these read constants at the top of the file instead of taking
arguments (`layer_weight_analysis`, `create_taxonomy`, `create_SSL_taxonomy`),
and the last two write to a CWD-relative path -- noted in docs/09.

---

## tests/ -- audited, nothing redundant

15 files, 225 tests. Every file maps to a live module and pins a distinct
contract; no duplication found.

| File | Exercises |
|---|---|
| `test_aasist_raw.py` | `models.aasist_raw` |
| `test_audio.py` | `scoring.audio` |
| `test_compare_trees.py` | `tools.compare_trees` |
| `test_config.py` | `config` |
| `test_grad_accum.py` | `main.train_epoch` |
| `test_lfcc_frontend.py` | `frontends.lfcc` |
| `test_migrate_layout.py` | `tools.migrate_layout`, `core.scorepath` |
| `test_paper_models.py` | roster, jobs, orchestration surface |
| `test_progress.py` | `orchestration.progress` |
| `test_scorepath.py` / `test_score_reading.py` | `core.scorepath`, `core.scorefile` |
| `test_scoring_driver.py` | `scoring.driver` |
| `test_seeding.py` | `core.utils.set_seed` |
| `test_verification_levels.py` | both verdict ladders |
| `test_views.py` | `analysis.views` |

`test_grad_accum.py` was checked specifically: it imports the real
`train_epoch` from `main.py`, and the loop defined inside the test is the
pre-change **reference** implementation it is compared against. That is correct
design, not a copy under test.

**Gap, not redundancy:** the suite runs entirely on synthetic fixtures and
validates no number in `outputs/`. The composition arithmetic, the
lossless-partition property of the views, and the `max|d| = 0.000000`
equivalence against `create_mlaad_tts_eer_heatmaps` were each verified by
one-off command, never encoded. Worth adding as opt-in slow tests.

---

## Not code, but in the repo

- [?] `PLANNED_CHANGES.md`, `REORG_PLAN.md`, `humanpending.md` -- internal design
      record. Useful to us, noise to a reader. Keep, move to `docs/internal/`,
      or drop at release.
- [ ] `outputs/figures_*`, `outputs/logs/` -- stale working output from earlier
      runs. Local only (gitignored); prune if disk matters, irrelevant to release.
