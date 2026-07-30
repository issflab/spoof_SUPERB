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
| `--job` | `JOB` | `all` `linear_head` `baselines` `mlaad` `mailabs` `spoofceleb` | which selection + policy; see below |
| `--systems` | `SYSTEMS` | `linear_head` `aasist_raw` `lfcc_gmm` | restrict back-ends |
| `--datasets` | `DATASETS` | any registry key | restrict datasets |
| `--models` | `MODELS` | s3prl upstream names | restrict SSL upstreams (`linear_head` only) |
| `--gpus` | `GPUS` | e.g. `0 1 2` | devices to spread over |
| `--workers` | `WORKERS` | `0` = one per GPU, `1` = sequential | parallel workers. **Not** `--job` -- see below |
| `--force` | `FORCE="yes"` | flag | re-score even when a complete NaN-free output exists |
| `--progress` | `PROGRESS` | `auto` `bar` `plain` `none` | live display; see [Watching a run](#watching-a-run) |
| `--all-models` | `PAPER_ONLY="no"` | flag | score all 24 trained heads, not just the 21 Table 5 reports |
| `--run-name` | `RUN_NAME` | any string | identity for this run; defaults to a timestamp |
| `--list` | — | flag | print the tasks and exit, running nothing |
| `--python` | — | path | interpreter for the scoring subprocesses; defaults to `cfg.python` |

Empty settings mean "all of them", so `DATASETS=""` is not a restriction.

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

24 trained linear heads are on disk; Table 5 reports 21. `audio_albert_960hr`,
`byol_a_2048` and `modified_cpc` were trained and scored but never reported, so
scoring them everywhere spends 36 of 288 tasks on columns nobody reads.

The roster is **read from `tests/baseline_table5.json`**, the same file the
zero-tolerance regression gate compares against, so it cannot drift from the
paper. Add a row to Table 5 and it becomes scoreable with no code change; lose
the file and the orchestrator refuses to start rather than silently widening
back to 24.

```bash
bin/orchestrate.sh --list                       # 21 heads + 2 baselines
bin/orchestrate.sh --all-models --list          # all 24 + 2
bin/orchestrate.sh --models byol_a_2048 --list  # naming one always works
```

Naming a model with `--models` overrides the filter, in the paper or not -- an
explicit request is never silently dropped.

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
what `--all-models` gives.

| `JOB` | Selects | Tasks | Policy it adds |
|---|---|---|---|
| `all` | every system on every dataset | 276 (312) | — |
| `linear_head` | every SSL head on every dataset | 252 (288) | — |
| `baselines` | `aasist_raw` + `lfcc_gmm` everywhere | 24 (24) | `batch_size=64` |
| `mlaad` | every SSL head on MLAAD v10 | 20 (22) | verify `mlaad`; 1 attempt; skips `byol_a_2048`, `mockingjay` |
| `mailabs` | every SSL head on M-AILABS | 20 (22) | as `mlaad` |
| `spoofceleb` | every SSL head on SpoofCeleb | 21 (24) | verify `spoofceleb`; 6 attempts; 3 h CUDA wait |

So `--job mlaad` is **not** the same as `--job all --datasets Multilingual`. The
latter is 23 tasks: it adds the two baselines and `mockingjay`, and runs no
verification.

Why the policies differ:

* **Verification.** Only these three datasets have a published score file to
  compare a fresh run against, and the two comparisons grade differently --
  `mlaad` requires Pearson *and* Spearman *and* sign agreement at 0 and
  tolerates 1% NaN; `spoofceleb` uses Spearman alone and tolerates none. See
  [verification](08-verification.md).
* **Retry budget.** `spoofceleb` was run through a period of driver instability
  and retries 6 times over 3 hours. `mlaad` and `mailabs` use a single attempt
  because those runs are short: failing fast beats a 3 h wait on a 20 min task.
* **Skip lists.** `mlaad` and `mailabs` exclude `byol_a_2048` and `mockingjay`
  by an earlier request. `spoofceleb` deliberately includes them -- fp32 fixes
  byol's fp16 STFT crash, and mockingjay's SpoofCeleb reference is usable. The
  exclusion is attached to the jobs where it applies rather than made a global
  default, so it is visible instead of silent.

These three are also how the published runs were organised, and their names are
what `_runs/{job}/` is keyed on, so each keeps a separate audit trail.

## Narrowing a sweep

A task is one `(system, dataset, model)`, and each axis has its own filter, so
any slice can be named directly:

```bash
# one model on one dataset -- the smallest useful unit of work
bin/orchestrate.sh --datasets wild --models xls_r_300m

# one model everywhere
bin/orchestrate.sh --models xls_r_300m

# every model on one dataset
bin/orchestrate.sh --datasets spoofceleb

# just the CPU baseline, one dataset
bin/orchestrate.sh --systems lfcc_gmm --datasets wild
```

| Filter | Selects | Notes |
|---|---|---|
| `--systems` | `linear_head`, `aasist_raw`, `lfcc_gmm` | |
| `--datasets` | any registry key | see `bin/score.sh --list_datasets` |
| `--models` | s3prl upstreams | applies to `linear_head` only |

The same three are settable in `bin/orchestrate.sh`'s settings block. A filter
combination that selects nothing produces nothing -- `--systems lfcc_gmm
--models xls_r_300m` is empty, because that back-end has no upstream.

Filters compose with `--job`, so a named job can be narrowed the same way:
`--job spoofceleb --models xls_r_300m` keeps that job's verification and retry
budget while running one upstream.

`--only` is the deprecated spelling of `--models`; it used to mean SSL model
for the SSL sweep and dataset for the baselines, which was a trap.

`all` is not a special case -- it is the selection with nothing excluded.
Expected row counts come from each dataset's protocol rather than being written
down, so the resume check is correct for every dataset.

Before a sweep starts, any dataset whose protocol is missing is reported with
the command that builds it, rather than failing 200 tasks in.

Job definitions live in `spoof_superb/orchestration/jobs.py`. A `Job` declares
only its policy -- skip list, retry budget, CUDA wait, batch size, verification
check. Output paths come from `core/scorepath.py` and expected row counts from
each dataset's protocol, so neither is written down per job. To add a dataset
sweep, add a `Job` there rather than writing a new script; to run an existing
one over a different slice, use the filters.

## Behaviour worth knowing

**Resume is automatic.** A complete, NaN-free output is not recomputed, only
re-verified. Set `FORCE="yes"` to override. This means an interrupted overnight
run can simply be restarted.

**GPUs are pinned by UUID, not index.** Index-based `CUDA_VISIBLE_DEVICES`
fails to initialise on this host once another process holds a different device,
which previously sent whole models to CPU without anyone noticing.

**rc=2 is retried in a fresh process.** That return code is the scoring
driver's "CUDA requested but unavailable" guard -- an environment fault, not a
model fault. Per-job retry budgets are in [the jobs table](#the-jobs).

**`WORKERS=1` runs sequentially**, which is what the old
`orchestrate_baselines.py` did.

**ASVspoof2021 needs `av` installed** or it decodes ~14x slower. libsndfile
cannot read 36-43% of those FLAC files and librosa falls back to a subprocess
per file. See [troubleshooting](11-troubleshooting.md).

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
