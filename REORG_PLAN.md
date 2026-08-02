# REORG_PLAN.md — Phase 0 Audit

> **Names in this document are as they were at audit time (2026-07-27) and are
> deliberately not updated.** Since then: `scripts/recompute_table5_mlaad_v10.py`
> became `spoof_superb/analysis/recompute_main_results.py`,
> `tests/baseline_table5.json` became `tests/baseline_main_results_table.json`,
> and prose that said "Table 5" now says "the main results table", because the
> paper's table number changes between drafts. This file is a record of the
> starting state, not a description of the current one.

Read-only audit of `/home/alhashim/ASD_SUPERB/spoof_SUPERB` at commit `b28fd63` (working tree clean).
No code file was moved, edited, or created for this audit.

Evidence tags per R8: **[executed]** = I ran it and read real output. **[inspected]** = I read the source. **[assumed]** = inference not yet verified by execution.

---

## 0. Executive summary — five findings that change the plan

| # | Finding | Severity |
|---|---|---|
| F1 | `pytest tests/ -q` collects **zero tests** and exits green. The refactor has no safety net today. **[executed]** | Blocker |
| F2 | Four different Python interpreters are hardcoded across launchers, and two of them differ in `soxr` version (1.0.0 vs 0.5.0.post1) — the exact library whose drift `verify_mlaad.py` documents as the cause of the reference logit offset. **[executed]** | Critical |
| F3 | `config.py` is imported by exactly **2 of 47** Python files. It is not the configuration system; it is a `main.py` helper. **[executed]** | High |
| F4 | The four `eval_*.py` files are **not four of a kind**. `evaluation.py` is a pure metrics library; the other three are scoring drivers. Merging all four is wrong. | High |
| F5 | `.asvld_skip` contains the live line `Filtering`, silently no-op'ing that condition in every `eval_asvld.py` run, resolved relative to the *script's own location*. Moving the script silently changes behavior. **[executed]** | High |

---

## 1. Eval convergence matrix

### 1.1 These are three different kinds of file, not four of one kind

| File | What it actually is | Verdict |
|---|---|---|
| `evaluation.py` | Pure metrics library: `compute_eer`, `compute_det_curve`, `compute_tDCF`, `calculate_EER`. No models, no audio, no dataset knowledge. Imported by **17 files** **[executed]**. | Already correct. **Do not merge.** Rename to `core/metrics.py` — the name `evaluation.py` is what invites the confusion. |
| `eval_asvld.py` | Scoring driver: (checkpoint, ASVLD condition) → score file. | Merge target |
| `eval_mlaad.py` | Scoring driver: (checkpoint, MLAAD / M-AILABS / SpoofCeleb) → score file. | Merge target |
| `eval_baselines.py` | Scoring driver for the 2 non-SSL baselines over 10 datasets. **Already has the registry architecture the other two lack** (`DATASETS`, per-dataset `_r_*` resolvers at :117–176, `reference_paths` :221, `read_reference` :229). | **Merge base** — generalize this one, fold the other two into it |

Answering your question 1 directly: yes, `eval_asvld` + `eval_mlaad` + `eval_baselines` collapse into one driver. `evaluation.py` must stay separate — it is the shared math the other three (and 14 scripts) already import.

### 1.2 Shared behavior (safe to unify, no numerical change)

| Behavior | Locations | Note |
|---|---|---|
| `pad(x, max_len=64600)` | `eval_asvld.py:42`, `eval_mlaad.py:45`, `eval_baselines.py:265` (`_pad`) | Byte-identical logic |
| `read_restrict_utts` | `eval_asvld.py:67`, `eval_mlaad.py:77` | Identical |
| `build_model` (LinearHead + `SimpleNamespace(ssl_feature=…, ssl_model=…)`) | `eval_asvld.py:98`, `eval_mlaad.py:138` | Identical intent |
| Score = `batch_out[:, 1]` | `eval_asvld.py:190`, `eval_mlaad.py:255` | Same class index |
| 4-column output `{utt} - {key} {score}` | `eval_asvld.py:200`, `eval_mlaad.py:275` | Same format |
| DataLoader loop, `shuffle=False, drop_last=False` | all three | Order-preserving |

### 1.3 DUPLICATED-BUT-DIVERGENT — must be decided, not silently merged

**D1 — Precision policy. This one changed published numbers already.**
- `eval_mlaad.py:177-180` exposes `--amp` (fp16 autocast); `eval_baselines.py:513` exposes `--amp`; **`eval_asvld.py` has no AMP path and is fp32-only.**
- `verify_noise_rerun.py:1-30` **[inspected]** records the consequence: 4 models (tera, mockingjay, mockingjay_960hr, audio_albert_960hr) carried 384,157 fp16 overflow NaN each (53.93% of rows).
- **Decision required.** My recommendation: fp32 default, `--amp` explicit opt-in, and the precision written into a sidecar `.meta.json` next to every score file so provenance stops being tribal knowledge. I will not pick this for you.

**D2 — CPU fallback policy.**
- `eval_mlaad.py:185-188` refuses to run on CPU and returns rc=2 ("a CPU run is ~25h vs ~20min").
- `eval_asvld.py:136` silently falls back to CPU.
- Recommendation: adopt the refusal. **Behavior change** for anyone running ASVLD on a CPU-only box — flagged, not applied.

**D3 — Atomic output write.**
- `eval_mlaad.py:271-276` writes `.part` then `os.replace`. `eval_asvld.py:198-201` writes in place.
- Recommendation: atomic everywhere. No numerical change; strictly removes a torn-file failure mode.

**D4 — Missing/undecodable audio.**
- `eval_asvld.py:162-170`: pre-pass `os.path.isfile` filter, drops before scoring.
- `eval_mlaad.py:127-135`: try/except inside `__getitem__`, returns silence flagged `ok=False`, dropped after.
- These solve different failures (missing file vs. undecodable file). Unify as both stages, not one.

**D5 — Ground-truth label source.** Three genuinely different mechanisms — keep as per-dataset adapters:
- ASVLD: protocol column 3 (`eval_asvld.py:60-63`)
- MLAAD/M-AILABS: single `--label` flag (`eval_mlaad.py:156`); SpoofCeleb: `--protocol_csv`, `attack=='a00'` ⇒ bonafide (`eval_mlaad.py:89-110`)
- Baselines: copied verbatim from a reference score file (`eval_baselines.py:229`)

**D6 — Batch/worker defaults:** 24/4 (asvld), 32/6 (mlaad), 64/6 (baselines). Models run under `.eval()` so this is **[assumed]** numerically neutral; the migration will assert it on one dataset rather than trust it.

---

## 2. Orchestrator matrix

All four are the same program with different constants. Answering your question 4: yes, one driver + config.

| Aspect | `orchestrate_mlaad` | `orchestrate_mailabs` | `orchestrate_spoofceleb` | `orchestrate_baselines` |
|---|---|---|---|---|
| Scheduling | thread + queue, `GPUS=[0,1,2]`, UUID pinning (:31-44) | same (:40-53) | same (:51-64) | **sequential**, no GPU pool |
| Retry | none | none | `MAX_ATTEMPTS=6`, `CUDA_WAIT_S=10800` (:46-48) | `MAX_ATTEMPTS=3`, `CUDA_WAIT_S=3600` (:174-176) |
| Model SKIP | `{byol_a_2048, mockingjay}` (:29) | same (:37) | **none** — explicitly in scope (docstring) | n/a |
| Resume | via status JSON | via status JSON | `output_is_complete` (:128) | `score_file_ok` (:145) |
| Verify hook | `verify_mlaad.py` | `verify_mlaad.py` | `verify_spoofceleb.py` | inline `eer_from_file` (:132) |
| Interpreter | hardcoded miniconda | hardcoded miniconda | hardcoded miniconda | `sys.executable` (:31) |
| Checkpoint prefix | `model_weighted_CCE_50_64_linear_head_ASV19_` (:27) | same (:36) | same (:39) | `resolve_model_path()` (:41) |

**Genuinely shared and extractable:** GPU UUID pinning, `cuda_healthy` / `wait_for_cuda`, status-JSON writer, worker pool, log-dir convention, summary table.
**Genuinely per-job (→ config):** output dir, protocol/audio roots, SKIP set, expected row count (`EXPECT_LINES=91130`, spoofceleb:40), retry budget, verify check name, dataset list and order.

`orchestrate_baselines.py` being sequential while the other three are parallel is the one real divergence. Recommendation: one pooled driver, `--jobs 1` reproduces the sequential behavior.

---

## 3. Verification matrix — do NOT merge the grades

The four verifiers compute overlapping statistics but encode **four different contracts**. The correct decomposition is one statistics module + a registry of grade policies.

| Script | Contract | Threshold | Side effect |
|---|---|---|---|
| `verify_mlaad.py` | Detection-equivalence | Pearson ≥ 0.99 **AND** Spearman ≥ 0.99 **AND** sign-agreement@0 ≥ 0.999 (:51-52) | none |
| `verify_spoofceleb.py` | Rank-equivalence only | Spearman ≥ 0.99; Pearson diagnostic only (:54) — docstring:1-14 explains Pearson was rejected because tail outliers dragged it to 0.92 on models with Spearman 0.996 | none |
| `verify_noise_rerun.py` | Promotion gate | `MIN_CORR=0.9998`, `MAX_DEER=0.15` pp, `OUTLIER_MEDIAN=0.05` (:68-70) | **`--promote` moves directories** (:89) |
| `verify_asvld.py` | Descriptive only | none — prints a table, no pass/fail | none |

Merging these into one flag-driven script would destroy the reasoning recorded in `verify_spoofceleb.py`'s docstring. Keep the policies; share only the statistics.

---

## 4. Config usage report — answering your question 3

**`config.py` is imported by 2 files: `main.py` and `tools_select_aasist_ckpt.py`. [executed]**
Not by any `eval_*`, any `orchestrate_*`, any `verify_*`, or any of `scripts/`.

Field-by-field:

| Field | Reachable via | Actually used | Problem |
|---|---|---|---|
| `model_arch` | env `SSL_MODEL_ARCH` only | dispatch at `main.py:405,415-428` | **No CLI flag.** Switching architecture requires an env var. |
| `dataset` | hardcoded `'ASV19'` | only string-formatting in `model_tag` (`main.py:373`) | Never validated against actual data |
| `database_path`, `protocols_path` | env, **and** argparse `main.py:240-241` overwrite at `:353-356` | yes | **Two sources of truth with independently maintained defaults** |
| `train_protocol`, `dev_protocol` | class default only | `main.py:443-444` | No CLI, no env |
| `mode` | env `SSL_MODE` only | `main.py:406,501` | No CLI flag |
| `save_dir`, `model_name` | env | `main.py:382` | `model_save_path` property is bypassed — `main.py:382` rebuilds the path from `model_tag` instead |
| `cuda_device` | env `CUDA_DEVICE` | `main.py:387` | Every other script uses `--cuda_device` argparse instead |
| `pretrained_checkpoint` | env + `--model_path` | `main.py:438-440` | ok |
| `eval_protocol` | **does not exist on `Config`** | referenced by main.py's eval branch | This is the documented crash cited in 3 separate docstrings |

Two structural defects:
- **`config.py:125` calls `cfg.prepare_dirs()` at import time**, creating `/data/ssl_anti_spoofing/asd_superb/` as a side effect of `import config`. Any tool that merely imports the module mutates the filesystem.
- The `Config` singleton is **mutated in place** by `main.py:353-358`, so "the config" differs depending on how far into `main()` you are.

So: the config file exists, but the codebase is ~96% argparse-and-hardcoded-constants. Making it authoritative is the single highest-leverage part of this reorg — it is what makes questions 1 and 4 (one eval script, one orchestrator) possible at all.

---

## 5. Environment hazard (F2) — [executed]

| Launcher | Interpreter |
|---|---|
| `run_asvld_model.sh`, `run_recompression.sh` | `/home/alhashim/.conda/envs/ASD_SUPERB/bin/python` |
| `run_noise_rerun.sh`, `watch_and_run_*.sh`, `orchestrate_{mlaad,mailabs,spoofceleb}.py` | `/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python` |
| `orchestrate_baselines.py` | `sys.executable` (whatever is active) |
| `tests/test_lfcc_frontend.py:1` | `/home/alhashim/.conda/envs/SER/bin/python` (Python **3.9.20**, vs 3.10.0 elsewhere) |

Measured difference between the two main envs:

```
spoof_SUPERB : librosa 0.11.0  soxr 1.0.0        numpy 2.2.6  scipy 1.15.3  torch 2.7.1+cu126
ASD_SUPERB   : librosa 0.11.0  soxr 0.5.0.post1  numpy 2.2.6  scipy 1.15.3  torch 2.7.1+cu126
```

`soxr` is librosa's resampler. `verify_mlaad.py:6-11` already attributes the reference-vs-rerun logit offset to exactly this class of drift. **Score files produced through the ASVLD shell scripts and through the MLAAD orchestrators were generated by different resamplers.** For any dataset whose audio is resampled to 16 kHz this is a real, measurable inconsistency; SpoofCeleb is natively 16 kHz and is unaffected (`verify_spoofceleb.py` docstring).

This is out of scope for a file-move refactor, but it belongs in `humanpending.md` and it is a reason the reorg should pin one interpreter in config rather than preserve four.

---

## 6. Test-suite reality (F1) — [executed]

```
$ python -m pytest tests/ -q
2 warnings in 13.13s          # zero tests collected, exit 0
```

`tests/test_aasist_raw.py`, `tests/test_grad_accum.py`, `tests/test_lfcc_frontend.py` are standalone scripts with `def main()` + `if __name__ == "__main__"` — pytest collects nothing from them.

Per R6, the acceptance criterion "pytest passes at every commit" is currently vacuous and would give false confidence through the entire migration. **Before any file moves**, these three need `test_*` functions wrapping their existing assertions (a mechanical change that adds no new test logic), plus one characterization test per merge target. Without this, D1–D6 cannot be verified as behavior-preserving.

---

## 7. Proposed target tree

```
spoof_superb/
  config/
    __init__.py          # layered resolve: dataclass defaults < YAML < env < CLI; no import side effects
    datasets.yaml        # per-dataset roots, protocols, resolver name, expected rows
    jobs.yaml            # per-orchestration-job: GPUs, SKIP, retries, verify check, out dir
  core/
    metrics.py           # <- evaluation.py  (unchanged math)
    scorefile.py         # 4-col read/write, atomic, NaN audit, .meta.json sidecar
    audio.py             # pad(), load, CROP
  models/                # aasist, aasist_raw, linear_head, sls + registry
  frontends/             # lfcc, rawboost
  data/                  # data_utils_SSL, trial-list parsers, _r_* path resolvers
  scoring/
    driver.py            # THE single scoring entry point
    backends.py          # linear_head | aasist_raw | lfcc_gmm loops
  orchestration/
    driver.py            # GPU pool, resume, retry, status JSON
    cuda.py              # cuda_healthy / wait_for_cuda
  verification/
    stats.py             # pearson / spearman / sign-agreement / dEER
    policies.py          # asvld | mlaad | spoofceleb | noise_rerun grade registry
  analysis/              # <- scripts/*.py
  train/                 # <- main.py train path, train_lfcc_gmm.py
bin/                     # shell wrappers over the entry points
```

### File-by-file `git mv` mapping

| Current | New |
|---|---|
| `evaluation.py` | `spoof_superb/core/metrics.py` |
| `eval_baselines.py` | `spoof_superb/scoring/driver.py` (merge base) |
| `eval_asvld.py` | folded into `scoring/driver.py` + `data/trial_lists.py` |
| `eval_mlaad.py` | folded into `scoring/driver.py` + `data/trial_lists.py` |
| `aasist_model.py` | `spoof_superb/models/aasist.py` |
| `aasist_raw_model.py` | `spoof_superb/models/aasist_raw.py` |
| `linear_model.py` | `spoof_superb/models/linear_head.py` |
| `sls_model.py` | `spoof_superb/models/sls.py` |
| `lfcc_frontend.py` | `spoof_superb/frontends/lfcc.py` |
| `lfcc_gmm.py` | `spoof_superb/models/lfcc_gmm.py` |
| `RawBoost.py` | `spoof_superb/frontends/rawboost.py` |
| `data_utils_SSL.py` | `spoof_superb/data/datasets_ssl.py` |
| `orchestrate_*.py` (4) | `spoof_superb/orchestration/driver.py` + `config/jobs.yaml` |
| `verify_*.py` (4) | `spoof_superb/verification/{stats,policies}.py` |
| `main.py` | `spoof_superb/train/ssl.py` + thin root `main.py` shim |
| `train_lfcc_gmm.py` | `spoof_superb/train/lfcc_gmm.py` |
| `config.py` | `spoof_superb/config/__init__.py` |
| `utils.py` | `spoof_superb/core/utils.py` |
| `append_mailabs.py`, `balance_mailabs.py`, `make_tsv_mlaad.py`, `report_mlaad.py` | `spoof_superb/data/prep/` |
| `tools_select_aasist_ckpt.py` | `spoof_superb/tools/` |
| `scripts/*.py` | `spoof_superb/analysis/` |
| `run_*.sh`, `watch_*.sh` | `bin/` |

Answering question 2 (`lfcc_frontend`, `lfcc_gmm`, `aasist_model`, `aasist_raw_model`, `linear_model`): note `lfcc_gmm.py` goes to `models/`, not `frontends/` — it is a classifier (EM over two diagonal GMMs, per `main.py:393`); only `lfcc_frontend.py` is a frontend. `sls_model.py` is imported by **nothing** — it is reachable only through `cfg.model_arch == 'sls'` at `main.py:417`, so it is live-but-unexercised, not dead. It moves; it does not get deleted.

---

## 8. Risk list

| # | Risk | Mitigation |
|---|---|---|
| R1 | `.asvld_skip` is read relative to `eval_asvld.py`'s own directory (`:128`) and **currently contains `Filtering`**. Moving the script relocates the sentinel and silently re-enables a condition. | Move to config as an explicit `skip_conditions` list; delete the sentinel only with your approval |
| R2 | `scripts/recompute_table5_mlaad_v10.py` is the authoritative reproducer for the paper's Table 5. Any change to score-file naming or output paths breaks the published pipeline. | Freeze all output paths; re-run this script as the per-commit gate |
| R3 | `tests/` use `sys.path.insert` + `from main import train_epoch` (`test_grad_accum.py:38`). Moving `main.py` breaks them. | Fix tests first (F1), then move |
| R4 | `orchestrate_*.py` hardcode `REPO=` and pass `cwd=REPO` to subprocesses. | Derive from `__file__`; keep an env override |
| R5 | 62 references across 39 files point at `/data/.../asd_superb_score_files`. | Out of scope this pass — deferred with the score-dir reorg |
| R6 | `main.py:34-37` derives `CONFIG_PATH`/`LOGS_ROOT` from `Path(__file__).parent`. Moving `main.py` one level deeper silently relocates `outputs/logs`. | Anchor to a single `REPO_ROOT` in config |
| R7 | Four `sys.path.insert` hacks (`verify_noise_rerun.py:58`, `scripts/compute_eer_tts.py:47`, `scripts/evaluate_score_file.py:12`, `scripts/evaluate_score_directory.py:9`) exist only because the repo is not a package. | They become deletable once the package exists — that is the cleanup, not extra scope |
| R8 | Interpreter/`soxr` divergence (§5). | Log to `humanpending.md`; pin one interpreter in config |

---

## 9. Recommended execution order

Each step is one commit, gated by the Table 5 reproducer rather than the (currently empty) test suite.

0. **Fix the test suite** so it actually collects — precondition for everything below.
1. Capture a numerical baseline: run `scripts/recompute_table5_mlaad_v10.py`, store the output as `tests/baseline_table5.json`.
2. `core/` + `models/` + `frontends/` — pure `git mv` + import rewrites, no logic change.
3. `data/` + trial-list parsers.
4. `scoring/driver.py` merge — **the only step with real numerical risk (D1–D6)**.
5. `orchestration/` merge + `jobs.yaml`.
6. `verification/` split into stats + policies.
7. `config/` becomes authoritative; strip hardcoded paths.
8. `bin/` shell wrappers; `analysis/` move; README table.

---

## 10. Decisions I need from you before Phase 1

1. **D1 precision policy** — fp32 default with opt-in `--amp`? (my recommendation)
2. **D2 CPU fallback** — adopt `eval_mlaad`'s refusal for ASVLD too? (behavior change)
3. **F1 test suite** — approve fixing the three test files as step 0? It is outside a literal reading of "only make changes directly requested", but without it no acceptance criterion in this plan is verifiable.
4. **`sls_model.py`** — keep as live-but-unexercised, or mark deprecated?
5. **`.asvld_skip`** — is `Filtering` still meant to be skipped?

## 11. `scripts/` inventory — answering your question 6

32 `.py` files (not ~30). Categories: score organization/merging (9), EER/FAR computation (7), figures (8), protocol/metadata construction (4), verification/QA (3), other (1).

### 11.1 The good news: the EER math is *not* duplicated

Every EER-producing script routes through `evaluation.compute_eer`. There is exactly one independent reimplementation — `verify_mlaad_column.py:91-98` (sklearn `roc_curve` + `scipy.optimize.brentq`) — and it is **deliberate**: it exists so a bug in `compute_det_curve` would not be silently reproduced by its own verifier, and it compares the two estimators against `TOL_ESTIMATOR = 0.01` pp. **This must survive the reorg untouched.** A naive "remove duplicate EER implementations" pass would delete the only independent check on the paper's numbers.

### 11.2 What *is* duplicated

| # | Duplication | Sites |
|---|---|---|
| S1 | **Score-file → (bonafide, spoof) reader** — each re-derives 3-col vs 4-col, header skip, float coercion, ×100 | 6 metric sites: `evaluation.py:7-29`, `compute_eer_matrix.py:130-166`, `compute_far_matrix.py:281-302`, `compute_eer_tts.py:333-366`, `recompute_table5_mlaad_v10.py:247-255`, `analyze_benchmark_distribution.py:380-390` — plus 3 non-metric parsers |
| S2 | **Pooling/aggregation math, line-for-line identical** | `compute_far_matrix.py:388-394/397-441/355-382` vs `compute_eer_tts.py:405-411/414-460/372-399`. Differ only in the inner metric call |
| S3 | **5-entry dataset protocol table, byte-identical** | `organize_tts_scores.py:42-46` and `verify_tts_protocols.py:77-81` |
| S4 | **`sys.path.insert(0, REPO_ROOT)` boilerplate** | 8 sites — exists *only* because the repo is not a package; deleted for free by the reorg |

### 11.3 Two metrics with no library home

- **FAR** is implemented once, at `compute_far_matrix.py:325-352`, inside a matrix-builder. `evaluation.py` has no FAR entry point. Note the polarity asymmetry: FAR counts `score > threshold` on spoof rows, while `compute_det_curve` treats bonafide as target.
- **Rank-AUC** `auc_bonafide_vs_spoof` at `create_mlaad_tts_eer_heatmaps.py:86-95`, used as a polarity diagnostic when EER > 50.

Both belong in `core/metrics.py` beside `compute_eer`. This is a move, not a rewrite.

### 11.4 Near-duplicate clusters

| Cluster | Members | What actually differs |
|---|---|---|
| A — heatmaps | `create_heatmap`, `create_tts_heatmaps`, `create_tts_eer_heatmaps`, `create_mlaad_tts_eer_heatmaps` | First three are pure CSV→PNG and share the same `ORDERED_MODELS` + `SEPARATOR_ROWS = [1,7,17]`; differ only in input flag, figure count, and colour clamp. The fourth is the outlier — it *computes* EERs and uses a different 22-model list |
| B — ranked re-plotters | `create_mlaad_group_ranked_figures`, `create_mlaad_tts_system_ranked_figure` | Same 6-model `REPRESENTATIVE` list, same `VMAX`, neither has a CLI. Differ in input CSV, orientation (transposed), and sort key |
| C — score-tree organizers | `organize_mlaad_scores` (177 L), `organize_tts_scores` (629 L) | Same output shape `<AR\|NAR>/<system>/<ssl>.txt`; differ in system-resolution source, unknown-bucket name, and MLAAD's extra language tree |
| D — metric matrices | `compute_eer_matrix`, `compute_far_matrix`, `compute_eer_tts` | `compute_eer_matrix` is the odd one (condition axis, no pandas). The other two are **the same program** over the same tree — see S2 |
| E — MLAAD meta combiners | `create_combined_mlaad_meta`, `create_combined_mlaad_meta_all` | `_all` is explicitly "modelled on" the other; two functions byte-identical. See §11.5 |
| F — normalize-and-pool | `apply_sigmoid_and_pool`, `apply_zscore_and_pool` | Identical CLI/DATASETS/structure; sigmoid is stateless, z-score adds a stats pass + `MIN_STD` guard |
| G — score-file evaluators | `evaluate_score_file`, `evaluate_score_directory` | Both call `calculate_EER`; the directory one concatenates per SSL model first, the file one can synthesize labels from a protocol via YAML |
| H — condition verifiers | `verify_and_split_condition_scores`, `check_condition_composition` | **Sequential, not parallel** — the first produces augmented files, the second checks them. Keep both |

Consolidation targets: D (merge the two TTS matrix builders behind one metric parameter), A/B (one heatmap renderer + config), E and F (one script + a `--transform` / `--scope` flag). C and H should stay separate — their differences are substantive.

### 11.5 A real bug found during the sweep — not a reorg issue

`create_combined_mlaad_meta_all.py:27-33` sets `csv.field_size_limit(sys.maxsize)` and `QUOTE_NONE`, with a comment naming the exact failure it fixes: `ja/kokoro` silently loses 53 of 1000 rows under default quoting. **`create_combined_mlaad_meta.py:69` still uses plain `delimiter="|"` and still has that bug.** Its output feeds `organize_tts_scores.py:45` and `verify_tts_protocols.py:80` as the MLAAD protocol.

This is outside the reorg scope. Flagging it rather than fixing it — see decision 6.

### 11.6 Verification scripts in `scripts/` — contracts

- **`verify_and_split_condition_scores.py`** — set-membership on utterance IDs: every ID in a condition file must be attributable to exactly one of five protocol-derived sources. Checks membership, *not* counts.
- **`check_condition_composition.py`** — the count check the above omits: exact expected row counts per condition from `CONDITION_COMPOSITION:49-104`.
- **`verify_mlaad_column.py`** — three-way agreement on Table 5's MLAAD column: transcription vs `access.tex` (≤ 0.0005), repo estimator vs independent estimator (≤ 0.01 pp), full pool vs balanced pool (≤ 0.2 pp). Exits non-zero on failure.

Combined with the four root `verify_*.py`, that is **7 verifiers across 3 contract families**: score-file equivalence (root four), protocol/set integrity (`verify_and_split`, `check_condition_composition`, `verify_tts_protocols`), and published-number agreement (`verify_mlaad_column`, `recompute_table5_mlaad_v10`). The reorg should group them by family, share the statistics layer, and leave every threshold and grade policy exactly where it is.

### 11.7 Path findings

Two roots dominate: `/data/ssl_anti_spoofing/…` (9 files) and `/data/Data/…` (6 files). Additionally, 14 files carry absolute paths in usage docstrings that will go stale the moment entry points change, and **two figure scripts write to CWD-relative paths** — `create_taxonomy.py:7` and `create_SSL_taxonomy.py:8` use `Path("outputs/figures/…")` while every other figure script anchors to `Path(__file__).resolve().parent.parent / "outputs"`. Those two silently write to the wrong place if run from anywhere but the repo root.

---

## 12. Additional decision

6. **`create_combined_mlaad_meta.py` quoting bug (§11.5)** — fix it in this pass, or log it to `humanpending.md` and leave it? It affects a protocol file consumed downstream, so fixing it could change results; leaving it means the reorg ships a known-defective script.

