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

```
{out_dir}/run_status.json    live, one entry per task
{out_dir}/SUMMARY.txt        final table
{out_dir}/logs/              per-task stdout, plus per-task verify logs
```

`run_status.json` records status, GPU, wall time, attempts, row counts, NaN
count and the verification verdict for each task. Tail it while a sweep runs.

A task ends `ok` only if the row count matches the job's expectation (where one
is declared) and there are no NaN. Anything else is `suspect` or `failed`, and
the driver exits non-zero if any task is not `ok`.

## Unattended runs

`bin/watch_and_run_spoofceleb.sh` and `bin/watch_and_run_aasist_mlaad.sh` wait
for CUDA to become available, then launch a job, then re-check. They exist for
running through a driver outage; the retry logic inside the orchestrator covers
the ordinary case.
