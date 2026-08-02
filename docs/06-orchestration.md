# 6. Orchestration

Scoring handles one model on one set. Orchestration runs the whole matrix
across GPUs, unattended, with resume, retry, a live status file and automatic
verification.

It does not score anything itself -- it launches
`python -m spoof_superb.scoring.driver` as a subprocess per task. That boundary
is forced by CUDA: `CUDA_VISIBLE_DEVICES` must be set before torch initialises
in a process, and torch caches a failed CUDA init, so a retry is only
meaningful in a fresh process.

## Run a job

Edit the settings block at the top of `bin/orchestrate.sh`, then:

```bash
bin/orchestrate.sh
bin/orchestrate.sh --list      # enumerate the tasks and exit, running nothing
```

Always `--list` first. It shows exactly which tasks will run and where each
output will land.

## Every option

The settings block is the intended way to configure a run: it is editable,
self-documenting, and what you ran is recoverable from the file afterwards.
But every setting also has a flag, and **flags are appended after the settings
block, so a flag wins over the file.** The script echoes the full command
before executing, so you can always see which won.

| Flag | Setting | Values | Meaning |
|---|---|---|---|
| `--job` | `JOB` | `all` `linear_head` `baselines` | which back-ends; see below |
| `--systems` | `SYSTEMS` | `linear_head` `aasist_raw` `lfcc_gmm` | restrict back-ends |
| `--datasets` | `DATASETS` | any registry key | restrict datasets |
| `--models` | `MODELS` | s3prl upstream names | restrict SSL upstreams (`linear_head` only) |
| `--gpus` | `GPUS` | e.g. `0 1 2` | devices to spread over |
| `--workers` | `WORKERS` | `0` = one per GPU, `1` = sequential | parallel workers. **Not** `--job` -- see below |
| `--force` | `FORCE="yes"` | flag | re-score even when a complete NaN-free output exists |
| `--progress` | `PROGRESS` | `auto` `bar` `plain` `none` | live display; see [Watching a run](#watching-a-run) |
| `--all-models` | `PAPER_ONLY="no"` | flag | score all 24 trained heads, not just the 21 the paper reports |
| `--verify-against` | `VERIFY_AGAINST` | path to a score tree | compare each finished column against that tree. Off by default |
| `--verify-layout` | `VERIFY_LAYOUT` | `legacy` `v2` | layout of the tree above |
| `--run-name` | `RUN_NAME` | any string | identity for this run; defaults to a timestamp |
| `--list` | — | flag | print the tasks and exit, running nothing |
| `--python` | — | path | interpreter for the scoring subprocesses; defaults to `cfg.python` |

Empty settings mean "the defaults", which for `DATASETS` is
`DEFAULT_DATASETS` -- 11 of the 12 scoreable sets. See
[Deepfake-Eval: segmented by default](#deepfake-eval-segmented-by-default).

### `--job` vs `--workers`

`--jobs` used to mean "worker threads", one character away from `--job`, which
picks the named job. A third spelling, `Job.n_jobs`, meant the LFCC-GMM pool
size. All three were unrelated. They are now:

| | Is |
|---|---|
| `--job all` | which named `Job` to run |
| `--workers 3` | how many worker threads the orchestrator runs (`WORKERS`) |
| `Job.gmm_processes` (=16) | processes in the LFCC-GMM pool, forwarded to the scoring driver as `--n_jobs` |

`--jobs` still works as a hidden deprecated alias for `--workers` and prints a
notice, so existing scripts keep running. `Job.gmm_processes` is a `jobs.py`
constant and affects the CPU baseline only. The scoring driver's own `--n_jobs`
flag is unchanged -- in that context "GMM worker processes" is unambiguous.

### Only the paper's models are scored by default

24 trained linear heads are on disk. The paper's main results table --
`\label{tab:results_main}` -- prints **19** of them, plus the two non-SSL
reference systems. Five were trained and scored but never reported:

    audio_albert_960hr   byol_a_2048   fbank   mockingjay   modified_cpc

Scoring those five everywhere spends 60 of 288 tasks on columns nobody reads,
including the two most expensive corpora.

The roster lives in `spoof_superb/scoring/models.py` as `PAPER_TABLE_ROWS`,
stated once as the display names the table prints. It is **not** derived from
`tests/baseline_main_results_table.json`: that file tracks 21 models, two of which
(`FBANK`, `Mockingjay`) the paper does not print, and nothing in it distinguishes
them. The regression gate deliberately guards more columns than the paper
reports, so the baseline is not the roster.

Instead, `tests/test_paper_models.py::test_m1_roster_matches_the_paper_table_exactly`
parses `access.tex` and asserts the two agree, whenever the paper repo is checked
out beside this one. Drift is detected rather than assumed impossible. The
name-to-slug mapping still comes from the baseline, so slugs cannot disagree with
the gate.

```bash
bin/orchestrate.sh --list                       # 19 heads + 2 baselines
bin/orchestrate.sh --all-models --list          # all 24 + 2
bin/orchestrate.sh --models fbank --list        # naming one always works
```

Naming a model with `--models` overrides the filter, in the paper or not -- an
explicit request is never silently dropped.

### Deepfake-Eval: segmented by default

Deepfake-Eval 2024 is registered twice, because it can be measured two ways:

| dataset key | trials | what one trial is |
|---|---|---|
| `deepfake_eval_2024` | 1,980 | one 4 s window per recording |
| `deepfake_eval_2024_segmented` | 56,481 | every 4 s window of every recording |

The model's input is 4 s, so the unsegmented column looks at the first four
seconds of each file and never sees the rest -- and these recordings run to
minutes. Segmenting cuts each one into 4 s pieces and scores all of them.

**The new score tree uses the segmented set.** It is in `DEFAULT_DATASETS`; the
unsegmented one is not, though it stays scoreable by name and the two write to
different paths, so both can sit on disk.

```bash
bin/orchestrate.sh                                     # segmented
bin/orchestrate.sh --datasets deepfake_eval_2024       # the published column
```

**This is not a bug fix and the two numbers are not comparable.** The paper
reports `n=1976` for DFEval24; per-segment trials weight long recordings more
heavily, so the segmented EER is a different quantity, not a corrected one. Any
write-up of the new tree needs to say which it reports.

### Three ways to lose work

**`--list` is not the default.** `bin/orchestrate.sh --datasets wild --models
xls_r_300m` *runs*; the same line with `--list` looks. There is no confirmation
prompt. Read the echoed command before pressing return.

**`--force` cannot be switched off from the command line.** It is a boolean
flag, so if `FORCE="yes"` is in the settings block it is already in the argument
list and nothing you pass removes it. Edit the file.

**Two runs of the same `--job` used to overwrite each other's record.** Fixed:
run state now lives at `{scores_root}/_runs/{job}/{run}/`, where `{run}` is a
timestamp unless you pass `--run-name`. Concurrent runs of one job no longer
share a `run_status.json` or interleave their logs.

### Detached runs

A sweep is hours to days. If the terminal that launched it goes away, the
orchestrator and every scoring subprocess it owns die together, mid-task, and
whatever they had not yet written is lost.

```bash
nohup bin/orchestrate.sh > run.log 2>&1 &
```

`--progress auto` detects the redirection and switches to periodic status lines,
so `run.log` stays readable. `tmux` or `screen` works equally well and lets you
reattach to the live bar.

## The jobs

A job is one `Job` dataclass in `jobs.py`. There is no separate policy object --
"policy" below just names the fields that are not selection. The thirteen fields
do five unrelated things:

| Role | Fields | Overridable from the CLI? |
|---|---|---|
| **Selection** -- which tasks exist | `systems` `datasets` `skip` | yes, by `--systems` `--datasets` `--models` |
| **Resources** -- how much machine | `gpus` `batch_size` `num_workers` `gmm_processes` | only `--gpus` |
| **Failure handling** | `max_attempts` `cuda_wait_s` | no |
| **Post-check** | `verify` | no |
| **Identity** -- where the record goes | `name` `run` `log_dir` | `--job` picks the job, `--run-name` the run |

Selection is fully expressible with the three filters. Nothing else is, and
that is the only reason the dataset-specific jobs exist.

The two override paths differ, which matters if you read the code: `--gpus`
mutates `job.gpus` in place, while `--systems/--datasets/--models` are passed as
arguments to `enumerate_tasks()` and never touch the job. So a filter narrows
what a job produces; it does not redefine the job.

Counts below are at the default (paper models only); the figure in brackets is
what `--all-models` gives -- every trained head on every dataset, with nothing
excluded.

| `JOB` | Selects | Tasks | Policy it adds |
|---|---|---|---|
| `all` | every system on every dataset | 231 (286) | — |
| `linear_head` | every SSL head on every dataset | 209 (264) | — |
| `baselines` | `aasist_raw` + `lfcc_gmm` everywhere | 22 (22) | `batch_size=64` |

**There is no per-dataset job.** There used to be three -- `mlaad`, `mailabs`,
`spoofceleb`. Each was a dataset name plus three facts about that dataset: which
policy grades it, its retry budget, and which upstreams to skip on it. Once
`--datasets` existed the name was redundant, and the facts belonged to the
dataset, so they moved to the registry:

```python
# spoof_superb/scoring/datasets.py
VERIFY_POLICY = {"Multilingual": "mlaad", "MAILABS": "mlaad",
                 "spoofceleb": "spoofceleb"}
```

This fixed a real hole rather than just tidying: because the grading policy lived
on `--job mlaad`, **`--job all` used to score MLAAD and SpoofCeleb with no
policy attached at all** -- the sweep anyone would actually run was the one that
skipped the check. The policy now travels with the dataset, so
`--datasets Multilingual` carries it regardless of which job selected it.

There used to be a per-dataset `SKIP_MODELS` here too, holding `mockingjay` and
`byol_a_2048` off MLAAD and M-AILABS. It was removed: both are outside the
19-model roster, so it excluded nothing `paper_only` did not. `--all-models` now
really does mean every trained head on every dataset.

`mockingjay_960hr` is unaffected and **is** scored on MLAAD -- it is the variant
the paper's MLAAD table lists. Plain `mockingjay` is not, because it is not one
of the 19 reported models, and `--models mockingjay` scores it deliberately.

## Verification is a separate step

**Scoring never reads a score file it did not just write.** Trial lists come
from protocols; no reference file is consulted, and there is no configuration
that changes this. A tree built from scratch therefore cannot inherit an older
tree's coverage -- which is exactly how the published ASV21-DF column ended up
at 152,955 rows of a 611,829-row protocol.

That leaves comparison as something you ask for, in one of two places.

**During the sweep**, if you want each column checked as it lands:

```bash
# in bin/orchestrate.sh
VERIFY_AGAINST="/data/ssl_anti_spoofing/asd_superb_score_files"
VERIFY_LAYOUT="legacy"
```

**Afterwards**, which costs nothing extra and does not slow the sweep:

```bash
python -m spoof_superb.verification.driver \
    --check mlaad \
    --new  {new_root}/raw/linear_head/mlaad_v10/xls_r_300m.txt \
    --ref  {old_root}/linear_head_MLAAD_v10/linear_head_MLAAD_v10_xls_r_300m.txt
```

Either way the reference tree is **named explicitly**. It is never derived from
`scores_root`, because `scores_root` is where the new files go -- comparing a
tree against itself is vacuous, and defaulting to some other tree is how a
"fresh" build silently stops being fresh.

### Building a new tree and promoting it

The order matters, and it is the reason verification is not automatic:

1. **Build from scratch.** `bin/orchestrate.sh` with `VERIFY_AGAINST=""`.
   Every column comes from its protocol. Coverage may legitimately *exceed* the
   old tree's.
2. **Compare against the old tree.** Per column, with the dataset's own policy.
   Expect differences in row counts where the old column was a subset; expect
   agreement in ranking and EER where it was not.
3. **Confirm, then promote.** Once the new tree is accepted, build the release
   manifest from it (`spoof_superb.tools.build_release_manifest`). From that
   point the new files are the reference and the old tree is history.

A column whose policy is `None` has no published twin to compare against; it
records `not compared` and that is not a failure.

## Watching a run

### In the terminal

A sweep prints a live display, redrawn in place:

```
[all] 41/312 tasks   13.4%  elapsed 6:12:40  eta 1 day, 8:19:02
  [###.......................]
  gpu0      linear_head/asvspoof2021_DF/xls_r_300m     62.3%  of 611,829 trials   2:41:09
  gpu1      linear_head/asvspoof2021_DF/wavlm_large    58.9%  of 611,829 trials   2:41:07
  gpu2      linear_head/Famous_Figures/hubert_large     4.1%  of 348,135 trials   0:11:22
OK       linear_head/wild/xls_r_300m: 31,779 lines, 0 NaN, no reference
```

Read it as three separate facts:

* **`41/312 tasks`** is exact. The percentage beside it is not the same number
  -- it includes the fraction of each running task, so the bar keeps moving
  during a three-hour column instead of sitting still for hours.
* **The per-task percentage** comes from the scoring subprocess's own progress
  counter, read back out of its log. **`of 611,829 trials`** comes from the
  protocol. They are shown side by side rather than multiplied together,
  because the subprocess counts batches, not trials.
* **`eta`** is throughput-based: measured progress so far, projected over the
  work left. It is unreliable for the first few minutes and tightens after
  that. It does not know that ASVLD is 60x larger than in-the-wild, so it
  jumps when the sweep reaches a big dataset.

`...` in place of a percentage means the task has started but has not yet
written a progress line -- model loading and protocol parsing happen first.

`PROGRESS` in `bin/orchestrate.sh` (or `--progress`) selects the display:

| value | behaviour |
|---|---|
| `auto` | bar on a terminal, periodic lines when redirected. The default. |
| `bar` | force the redrawing bar |
| `plain` | one status line every 60 s, no escape codes |
| `none` | per-task result lines only |

`auto` is what makes `nohup bin/orchestrate.sh > run.log &` safe: a redrawing
bar would fill that log with cursor-movement codes, so redirection silently
switches to `plain`.

### On disk

```
{out_dir}/run_status.json    live, one entry per task
{out_dir}/SUMMARY.txt        final table
{out_dir}/logs/              per-task stdout, plus per-task verify logs
```

`run_status.json` records status, GPU, wall time, attempts, row counts, NaN
count and the verification verdict for each task. Tail it while a sweep runs,
or from another shell if the sweep is detached.

`logs/{job}_{system}_{dataset}_{frontend}.log` is the full subprocess output for
one task -- the same file the progress display reads its percentage from. Go
there when a task is `suspect` or `failed`.

A task ends `ok` only if the row count matches the job's expectation (where one
is declared) and there are no NaN. Anything else is `suspect` or `failed`, and
the driver exits non-zero if any task is not `ok`.

## Unattended runs

`bin/watch_and_run_spoofceleb.sh` and `bin/watch_and_run_aasist_mlaad.sh` wait
for CUDA to become available, then launch a job, then re-check. They exist for
running through a driver outage; the retry logic inside the orchestrator covers
the ordinary case.
