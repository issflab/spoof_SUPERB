# Planned changes

Work that is agreed or proposed but not yet done. Distinct from
`humanpending.md`, which records decisions blocked on a human; these are
changes with a known shape, listed so none is lost between sessions.

Ordered by priority. Each item states what is wrong, what it costs, and what it
needs from a human before it can start.

Status: **open** = not started, **needs decision** = cannot start until a
question is answered, **queued** = decided, waiting on the current sweep.

---

## P1  Score only the models the paper reports  [needs decision]

24 trained linear heads exist; Table 5 reports 21. Three are scored on every
dataset and never appear in the paper:

    audio_albert_960hr
    byol_a_2048
    modified_cpc

All 21 Table 5 rows have a trained head, so nothing is missing in the other
direction.

**Cost of the status quo:** 288 tasks where 252 tasks would do -- 36 wasted
(12%). Of
those, 15 are already on disk (each of the three has `wild`, `eval_2019`,
`asvspoof2021_LA`, `asvspoof2021_DF`, `asvspoof5`), so **21 remain avoidable**,
including the two most expensive columns.

**Shape of the fix.** Do not hand-maintain a 21-name list. `tests/baseline_table5.json`
already maps each Table 5 row to its slug and is the authority the regression
gate uses, so derive the roster from it:

```python
# spoof_superb/scoring/models.py
def paper_models():
    """The 21 slugs Table 5 reports, read from the regression baseline."""
```

Then `discover_linear_heads` gains a `paper_only` filter, and the jobs default
to it. `--models` stays an explicit override, so scoring an excluded model
remains one flag away.

**Needs a human decision:**

1. Excluded, or merely deprioritised (scored last, after the paper matrix is
   complete)?
2. Delete the 15 existing non-paper score files, or leave them? They are valid
   scores, just not reported. Leaving them costs disk and makes coverage
   arithmetic ambiguous; deleting them is irreversible.
3. Should the same restriction apply to `verification` and the analysis
   scripts, or only to scoring?

---

## P2  Give each run its own identity  [open]

Run state is keyed on the job name alone:

    {scores_root}/_runs/{job}/run_status.json
    {scores_root}/_runs/{job}/SUMMARY.txt
    {scores_root}/_runs/{job}/logs/{job}_{system}_{dataset}_{frontend}.log

So two runs of the same `--job` collide. Observed both ways on 2026-07-29:

* `run_status.json` was **overwritten** -- a 19-task record replaced by a
  1-task record from an unrelated invocation.
* `logs/` silently **mixes** runs: 42 logs from Jul 28 and 78 from Jul 29 in one
  directory, because log filenames carry no run identifier. A re-run of the same
  task overwrites its predecessor's log.

Score files are unaffected and resume reads the score files, so this costs the
audit trail rather than the work -- but the audit trail is what a paper's
provenance rests on.

**Shape of the fix.** A `--run-name` (default: timestamp) separate from both the
job and the selection, so `_runs/{job}/{run}/` is unique per invocation. Keep a
`latest` symlink so existing tooling and documented paths keep working.

---

## P3  Record why the audio loader fell back  [queued]

`load_wave` swallows the PyAV failure:

```python
    except Exception:
        pass
```

When the live sweep silently used audioread despite `av` being installed, the
log said only "PySoundFile failed" and diagnosing it needed a separate
reproduction. The loader should report, once per process, which decoder handled
what and why a fallback happened.

Small and contained, but it is on the decode path, so it waits for the current
sweep rather than being applied under it.

---

## P4  "job" names three unrelated things  [needs decision]

    --job all        which named Job
    --jobs 3         worker threads in the orchestrator
    Job.n_jobs=16    processes in the LFCC-GMM pool (forwarded as --n_jobs)

`--job` and `--jobs` differ by one character and mean unrelated things. This is
the same class of trap as the `--only` overload already removed, where one flag
meant "SSL model" in one job and "dataset" in another.

**Shape of the fix.** Rename `--jobs` to `--workers` (keep `--jobs` as a hidden
deprecated alias, as was done for `--only`), and `Job.n_jobs` to
`Job.gmm_processes`.

**Needs a human decision:** this changes the CLI surface. Any script or note of
yours that passes `--jobs` keeps working via the alias, but the documented
spelling changes.

---

## P5  A Job conflates selection with policy  [needs decision]

`Job`'s thirteen fields do five unrelated things -- selection (`systems`,
`datasets`, `skip`), resources (`gpus`, `batch_size`, `num_workers`, `n_jobs`),
failure handling (`max_attempts`, `cuda_wait_s`), post-check (`verify`), and
identity (`name`, `log_dir`).

Selection is now fully expressible with `--systems/--datasets/--models`, so the
`datasets=` field inside `mlaad`, `mailabs` and `spoofceleb` duplicates a filter.
What those jobs uniquely carry is dataset-specific policy: a verification check,
a retry budget, a CUDA wait, a skip list.

**Shape of the fix.** Move the policy onto the dataset registry, which already
owns each dataset's trial source and resolver -- the same move that closed
RP-6/7/8. The three named jobs then disappear, leaving `all` plus filters.

**Blocked on P2.** The job name is currently also the run's on-disk identity, so
collapsing the jobs would funnel every run into `_runs/all/` -- exactly the
collision P2 fixes. P2 must land first.

**Needs a human decision:** whether to do this at all. It is a real
simplification, but the named jobs are how the published runs were organised and
the current arrangement is documented and working.

---

## P6  `Job.extra` is dead  [open]

Declared in the dataclass, consumed nowhere. Delete it. Trivial; listed only so
it is not rediscovered.

---

## Carried over from `humanpending.md`

Still open there, unchanged by the reorganisation:

* **RP-1** -- two environments disagreed on `soxr` (1.0.0 vs 0.5.0.post1),
  librosa's resampler. Existing score files were produced by both. Blocking for
  provenance; cannot be fixed retroactively.
* **RP-2** -- `create_combined_mlaad_meta.py` still has the quoting bug its
  `_all` sibling fixes: `ja/kokoro` loses 53 of 1000 rows.
* **RP-3** -- confirm `Filtering` should stay in the default ASVLD skip list.
* **RP-4** -- `compute_far_matrix` and `compute_eer_tts` duplicate their
  aggregation; FAR has no home in `core/metrics.py`.
* **RP-5** -- the old score directory holds ~19 GB of duplicated or regenerable
  views. Reorganisation deferred.
