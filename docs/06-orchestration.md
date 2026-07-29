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

## The jobs

| `JOB` | What it scores | Verify policy |
|---|---|---|
| `all` | every system on every dataset (312 tasks) | none |
| `linear_head` | every SSL linear head on every dataset (288) | none |
| `mlaad` | every linear head on MLAAD v10 fake | `mlaad` |
| `mailabs` | every linear head on M-AILABS bonafide, into a staging dir | `mlaad` |
| `spoofceleb` | every linear head on the SpoofCeleb eval set | `spoofceleb` |
| `baselines` | `aasist_raw` and `lfcc_gmm` on every dataset | none |

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

`--only` is the deprecated spelling of `--models`; it used to mean SSL model
for the SSL sweep and dataset for the baselines, which was a trap.

A job is a selection over (system x dataset x frontend) plus its runtime
policy, so `all` is not a special case -- it is the selection with nothing
excluded. Expected row counts come from each dataset's protocol rather than
being written down, so the resume check is correct for every dataset.

Before a sweep starts, any dataset whose protocol is missing is reported with
the command that builds it, rather than failing 200 tasks in.

Job definitions live in `spoof_superb/orchestration/jobs.py`. Each declares its
output directory, reference directory, skip list, retry budget, expected row
count and verification policy. To add a dataset sweep, add a `Job` there rather
than writing a new script.

## Behaviour worth knowing

**Resume is automatic.** A complete, NaN-free output is not recomputed, only
re-verified. Set `FORCE="yes"` to override. This means an interrupted overnight
run can simply be restarted.

**GPUs are pinned by UUID, not index.** Index-based `CUDA_VISIBLE_DEVICES`
fails to initialise on this host once another process holds a different device,
which previously sent whole models to CPU without anyone noticing.

**rc=2 is retried in a fresh process.** That return code is the scoring
driver's "CUDA requested but unavailable" guard -- an environment fault, not a
model fault. Retry budgets differ per job: `spoofceleb` retries 6 times over a
3-hour window, `baselines` 3 times over 1 hour.

**Skip lists differ on purpose.** `mlaad` and `mailabs` skip `byol_a_2048` and
`mockingjay`; `spoofceleb` deliberately includes them, because fp32 fixes byol's
fp16 STFT crash and mockingjay's SpoofCeleb reference is usable.

**`WORKERS=1` runs sequentially**, which is what the old
`orchestrate_baselines.py` did.

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
