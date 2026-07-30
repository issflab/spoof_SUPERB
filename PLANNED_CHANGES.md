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

## P1  Score only the models the paper reports  [DONE 2026-07-29, corrected]

24 trained linear heads exist. The paper's main results table -- **Table 6**,
`\label{tab:results_main}` in access.tex -- prints **19**, plus the two non-SSL
reference systems. Five were trained and scored but never reported:

    audio_albert_960hr   byol_a_2048   fbank   mockingjay   modified_cpc

That is 60 of 288 tasks (21%) on columns nobody reads, including the two most
expensive corpora.

**The first implementation got the roster wrong, and the reason is worth keeping.**
It derived the 21 names from `tests/baseline_table5.json`, arguing that a
hand-maintained list would drift from the paper while the regression baseline
could not. The argument was right about the risk and wrong about the remedy: the
baseline had *already* drifted. It tracks 21 models, two of which the paper does
not print --

    FBANK        has mean/pooled, not printed
    Mockingjay   no MLAAD cell, not printed

-- and no key in the JSON distinguishes them from the 19 that are printed, so no
filter over that file could have recovered the real roster. The regression gate
deliberately guards more columns than the paper reports; the baseline is a
superset, not the roster.

**As shipped:** `PAPER_TABLE_ROWS` in `spoof_superb/scoring/models.py` states the
19 display names once, from the table. `test_m1_roster_matches_the_paper_table_exactly`
parses `access.tex` and asserts the two agree, skipping only when the paper repo
is not checked out beside this one. Drift is now detected rather than assumed
impossible -- the only honest arrangement when the authority lives in another
repository. The name-to-slug mapping still comes from the baseline, so slugs
cannot disagree with the gate.

**Decided as implemented**, reading "only work on models which are relevant for
the paper" as excluded rather than reordered:

1. **Excluded**, not deprioritised. `paper_only=True` is the enumerator default.
2. **Existing non-paper score files were left in place.** Valid scores; deleting
   is irreversible and was not asked for. They are simply no longer extended.
3. **Scoring only.** Verification and analysis read whatever exists; narrowing
   them would hide a column someone produced deliberately with `--models`.

`--all-models` opts out. `--models` overrides in either direction -- a default
filter that dropped an explicit request would be the `--only` trap in a new
costume.

Task counts: `all` 312 -> 252, `linear_head` 288 -> 228.

A side effect worth recording: `SKIP_MODELS` (`mockingjay`, `byol_a_2048` off
MLAAD/M-AILABS) is now redundant at the default, since both are unreported
anyway. Kept because it still bites under `--all-models` and states an intent
that outlives the current roster. `test_j4b` pins that relationship.

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

## P5  Jobs pick back-ends; datasets carry their own policy  [DONE 2026-07-29]

`mlaad`, `mailabs` and `spoofceleb` were each a dataset name plus three facts
about that dataset: which policy grades it, its retry budget, and which
upstreams to skip. Once `--datasets` existed the name was redundant and the
facts belonged to the dataset.

**Done, and it closed a real hole rather than just tidying.** Because the
grading policy lived on `--job mlaad`, `--job all` -- the sweep anyone would
actually run -- had `verify=None` for MLAAD and SpoofCeleb. The check that
exists for those two corpora only ran if you happened to invoke the specially
named job.

    JOBS         = {all, linear_head, baselines}
    VERIFY_POLICY = {Multilingual: mlaad, MAILABS: mlaad, spoofceleb: spoofceleb}
    SKIP_MODELS   = {Multilingual: {byol_a_2048, mockingjay}, MAILABS: same}

`Job.verify` and `Job.skip` are gone; `Task` carries `verify`, `system` and
`frontend` instead, the way it already carried `expect_lines`.

**Scoring no longer resolves a reference file at all.** This was the deciding
requirement: a new tree must be buildable from protocols alone, or it can only
ever reproduce the old tree's coverage -- which is how the published ASV21-DF
column came to hold 152,955 rows of a 611,829-row protocol. Comparison is now
opt-in against an explicitly named tree (`--verify-against ROOT`), never derived
from `scores_root`.

The intended sequence, which is why verification is not automatic:

1. build from scratch, no comparison
2. compare the new tree against the old one, per column, with each dataset's
   own policy
3. once accepted, build the release manifest from the new tree -- from then on
   the new files are the reference

Counts moved again: `all` 276 -> 274, `linear_head` 252 -> 250, because
`SKIP_MODELS` now applies under `--job all`, which it never did before.
`mockingjay` is excluded from MLAAD and M-AILABS because the paper's MLAAD table
reports no cell for it -- the plain variant only; `mockingjay_960hr` is scored on
MLAAD and is the one that table lists. `mockingjay` remains in the paper roster,
since the main results table reports it on nine of ten datasets.
`--models mockingjay` fills the empty MLAAD cell deliberately.

A test caught a real gap while writing this: `skip` was evaluated before `only`,
so `--models mockingjay --datasets Multilingual` silently produced nothing. An
explicit request now overrides both `SKIP_MODELS` and `paper_only`.

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
