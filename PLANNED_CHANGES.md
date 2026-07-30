# Planned changes

Work that is agreed or proposed but not yet done. Distinct from
`humanpending.md`, which records decisions blocked on a human; these are
changes with a known shape, listed so none is lost between sessions.

Ordered by priority. Each item states what is wrong, what it costs, and what it
needs from a human before it can start.

Status: **open** = not started, **needs decision** = cannot start until a
question is answered, **queued** = decided and waiting on something,
**DONE** = shipped, kept for the record of what changed and why.

---

## P1  Score only the models the paper reports  [DONE 2026-07-29]

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

**Decided as implemented**, on the reading that "only work on models which are
relevant for the paper" means excluded rather than reordered:

1. **Excluded**, not deprioritised. `paper_only=True` is the enumerator default.
2. **The 15 existing non-paper score files were left in place.** They are valid
   scores and deleting them is irreversible; nothing asked for that. They are
   simply no longer produced or extended. Say the word to remove them.
3. **Scoring only.** Verification and the analysis scripts read whatever score
   files exist; narrowing them was not implied and would hide a column someone
   deliberately produced with `--models`.

Delivered as `spoof_superb/scoring/models.py` (`paper_models()` reading
`tests/baseline_table5.json`), a `paper_only` argument threaded through
`discover_linear_heads` / `_frontends` / `enumerate_tasks` defaulting to True,
`--all-models` to opt out, and `PAPER_ONLY` in `bin/orchestrate.sh`. Naming a
model with `--models` overrides the filter in either direction.

Task counts moved: `all` 312 -> 276, `linear_head` 288 -> 252, `mlaad`/`mailabs`
22 -> 20, `spoofceleb` 24 -> 21.

---

## P2  Give each run its own identity  [DONE 2026-07-29]

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

**Done.** `Job.run` (default: a `YYYYmmdd-HHMMSS` timestamp) with `--run-name`
to set it, `Job.job_dir` for the across-runs parent, and `link_latest()`
maintaining `{job_dir}/latest`. Documented paths move one level deeper:
`{job_dir}/run_status.json` becomes `{job_dir}/latest/run_status.json`, still a
fixed location. The symlink is best-effort -- a status link is never worth
failing a sweep for.

---

## P3  Record why the audio loader fell back  [DONE 2026-07-29]

`load_wave` swallows the PyAV failure:

```python
    except Exception:
        pass
```

When the live sweep silently used audioread despite `av` being installed, the
log said only "PySoundFile failed" and diagnosing it needed a separate
reproduction. The loader should report, once per process, which decoder handled
what and why a fallback happened.

**Done.** `_announce()` prints, once per (decoder, reason) per process, which
decoder ran and why the fallback happened -- and marks the audioread path
`(SLOW)`, because avoiding it is the entire reason `av` is a dependency.

Implementing it surfaced a second, worse bug: the reporting call sat *inside* the
`try`, so a `NameError` in it was caught by `except Exception`, recorded as "av
failed", and silently demoted the whole corpus to the slow path. The loader now
uses `try/except/else` throughout, with all bookkeeping in `else` where the
except clauses cannot reach it. `test_a12` guards exactly that.

---

## P4  "job" names three unrelated things  [DONE 2026-07-29]

    --job all        which named Job
    --jobs 3         worker threads in the orchestrator
    Job.n_jobs=16    processes in the LFCC-GMM pool (forwarded as --n_jobs)

`--job` and `--jobs` differ by one character and mean unrelated things. This is
the same class of trap as the `--only` overload already removed, where one flag
meant "SSL model" in one job and "dataset" in another.

**Done.** `--workers` is the spelling; `--jobs` remains a hidden alias that
prints a deprecation notice, so existing scripts keep working. `Job.n_jobs` is
now `Job.gmm_processes`. The scoring driver's own `--n_jobs` flag is unchanged --
in that context "GMM worker processes" is unambiguous, and it is used by
`bin/score.sh`, `bin/train_lfcc_gmm.sh` and two docs.

---

## P5  A Job conflates selection with policy  [needs decision]

`Job`'s thirteen fields do five unrelated things -- selection (`systems`,
`datasets`, `skip`), resources (`gpus`, `batch_size`, `num_workers`,
`gmm_processes`), failure handling (`max_attempts`, `cuda_wait_s`), post-check
(`verify`), and identity (`name`, `run`, `log_dir`).

Selection is now fully expressible with `--systems/--datasets/--models`, so the
`datasets=` field inside `mlaad`, `mailabs` and `spoofceleb` duplicates a filter.
What those jobs uniquely carry is dataset-specific policy: a verification check,
a retry budget, a CUDA wait, a skip list.

**Shape of the fix.** Move the policy onto the dataset registry, which already
owns each dataset's trial source and resolver -- the same move that closed
RP-6/7/8. The three named jobs then disappear, leaving `all` plus filters.

**No longer blocked.** P2 has landed, so collapsing the jobs would no longer
funnel every run into one status directory. This is now purely a question of
whether to do it.

**Needs a human decision:** whether to do this at all. It is a real
simplification, but the named jobs are how the published runs were organised and
the current arrangement is documented and working.

---

## P6  `Job.extra` is dead  [DONE 2026-07-29]

**Done.** Deleted. `Job` still has thirteen fields: `run` replaced it.

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
