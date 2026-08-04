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

---

## Defects found while auditing -- fix or decide, do not just delete

- [?] **`--model_arch sls` is broken.** `main.py:434` calls `sls_model(...)`, but
      the import at `main.py:14` is commented out:
      `# from sls_model import Model as sls_model`. Any `sls` run raises
      `NameError`. `spoof_superb/models/sls.py` (293 loc) is therefore dead in
      practice. Either restore the import (it should be
      `from spoof_superb.models.sls import Model as sls_model`) or drop `sls`
      from the `--model_arch` choices and delete the model. **Not a silent
      delete: it changes what the benchmark offers.**

- [?] **`tests/baseline_main_results_table.json` is misfiled.** It is no longer a
      test fixture -- `scoring/models.py::_slug_by_display` reads it, so
      `paper_models()` and every roster decision depend on it. It is the only
      record of which score-file slug produced which printed row. Its name and
      its location under `tests/` describe a job it no longer has. Move to e.g.
      `spoof_superb/scoring/paper_roster.json`, contents unchanged.

- [ ] **`seed_worker` is imported nowhere and wired nowhere.** `main.py`'s
      DataLoaders run `num_workers=8` with no `worker_init_fn`, so the workers
      are not deterministically seeded. Wiring it **changes training results**,
      so it is a decision, not a tidy-up. (Recorded in its docstring.)

---

## Tier 1 -- safe to delete

Nothing imports them, no doc names them as current, and no committed data
depends on them.

- [ ] `spoof_superb/analysis/create_tts_heatmaps.py` (213 loc) -- not in any doc,
      zero references, superseded by `create_mlaad_tts_eer_heatmaps`
- [ ] `spoof_superb/data/prep/make_tsv_mlaad.py` (61 loc) -- no CLI, no doc, no refs
- [ ] `spoof_superb/data/prep/report_mlaad.py` (123 loc) -- no CLI, no doc, no refs
- [ ] `bin/run_asvld_model.sh`, `bin/run_noise_rerun.sh`,
      `bin/run_recompression.sh`, `bin/watch_and_run_aasist_mlaad.sh` -- one-off
      operational scripts for completed work; every path reference points at the
      **legacy** score tree
- [ ] `outputs/` and `spoof_superb_outputs/` working copies (gitignored, local only)
- [ ] root PDF `A_Superb-Style_Benchmark_...pdf` -- ships the paper in the code repo

---

## Tier 2 -- superseded, but deleting loses provenance

These produced published figures or generate committed data. Deleting is
defensible; doing it silently is not. Decide per group.

**The legacy TTS chain** -- replaced by `analysis/tts_systems.py` + `views.py`:
- [?] `analysis/compute_eer_tts.py` (531), `compute_far_matrix.py` (502),
      `create_tts_eer_heatmaps.py` (275), `apply_zscore_and_pool.py` (347),
      `apply_sigmoid_and_pool.py` (232), `organize_tts_scores.py` (631),
      `organize_mlaad_scores.py` (205), `strip_bonafide_from_tts.py` (97)
- Note: the z-score/sigmoid poolers exist only for normalised scores, which
  P11-D8 established this project does not use.

**The legacy degradation chain** -- replaced by `analysis/acoustic_degradation.py`:
- [?] `analysis/compute_eer_matrix.py` (233) -- docs/09 already calls it superseded
- [?] `analysis/verify_and_split_condition_scores.py` (625),
      `check_condition_composition.py` (311) -- condition splitting, now done by
      `views.py` + `conditions.py` from the protocols

**One-off score-file surgery on the legacy tree:**
- [?] `analysis/merge_filtered_scores.py` (425),
      `merge_asv21la_into_hashim_baseline.py` (138),
      `combine_asvspoofld_scores.py` (232)

**Generates committed data -- check before deleting:**
- [?] `analysis/build_mlaad_dir_map.py` (313) writes
      `mlaad_v10_dir_to_system.csv` and `mlaad_v10_table4_provenance.csv`,
      which `views.py` and `tts_systems.py` READ. Deleting it makes those CSVs
      unreproducible. Keep, or record their provenance elsewhere first.
- [?] `analysis/create_combined_mlaad_meta.py` (111),
      `create_combined_mlaad_meta_all.py` (145) -- MLAAD metadata assembly;
      confirm nothing regenerates the taxonomy CSV through them.
- [?] `analysis/verify_tts_protocols.py` (596) -- checks the TTS protocol CSV
      drafts, which the view pipeline no longer uses.

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
- [ ] `outputs/figures_*`, `outputs/logs/` -- stale working output from earlier runs
