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

24 trained linear heads exist. The paper's main results table --
`\label{tab:results_main}` in access.tex -- prints **19**, plus the two non-SSL
reference systems. Five were trained and scored but never reported:

    audio_albert_960hr   byol_a_2048   fbank   mockingjay   modified_cpc

That is 60 of 288 tasks (21%) on columns nobody reads, including the two most
expensive corpora.

**The first implementation got the roster wrong, and the reason is worth keeping.**
It derived the 21 names from `tests/baseline_main_results_table.json`, arguing that a
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

A side effect: `SKIP_MODELS` (`mockingjay`, `byol_a_2048` off MLAAD/M-AILABS)
became redundant, since both are unreported models that `paper_only` already
excludes. **Removed**, along with `discover_linear_heads(skip=...)`, whose only
caller it was. `--all-models` now really means every trained head on every
dataset: 312 tasks rather than 308.

The original exclusion was an early request, and byol's half of it was a
workaround for an fp16 STFT crash that fp32 scoring already fixes. Neither
reason survived, so keeping the mechanism would have been keeping a comment in
executable form.

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
    (SKIP_MODELS also moved here, then was removed as redundant -- see P1)

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

`mockingjay_960hr` is scored on MLAAD -- the variant the paper's MLAAD table
lists. Plain `mockingjay` is not, because it is not one of the 19 reported
models; `--models mockingjay` scores it deliberately.

A test caught a real gap while writing this: the skip list was evaluated before
`only`, so `--models mockingjay --datasets Multilingual` silently produced
nothing. An explicit request now overrides the default roster.

---

## P6  `Job.extra` is dead  [DONE 2026-07-29]

**Done.** Deleted. `Job` still has thirteen fields: `run` replaced it.

---

## P7  Deepfake-Eval: segmented is the new tree's column  [DONE 2026-07-30]

Deepfake-Eval 2024 is registered twice because it can be measured two ways:

    deepfake_eval_2024              1,980 trials -- one 4 s window per recording
    deepfake_eval_2024_segmented   56,481 trials -- every 4 s window

The model's input is 4 s, so the unsegmented column looks at the first four
seconds of each file and never sees the rest; these recordings run to minutes.

**Done.** `DEFAULT_DATASETS` holds 11 of the 12 scoreable sets, excluding
`deepfake_eval_2024`. `ordered_datasets(None)` reads it, so a sweep with no
`--datasets` scores the segmented set. The unsegmented one stays scoreable by
name and the two write to different paths, so both can sit on disk.

Counts: `all` 252 -> 231, `linear_head` 228 -> 209, `baselines` 24 -> 22.

**Not a bug fix, and it needs saying in the paper.** The published DFEval24
column is n=1,976. Per-segment trials weight long recordings more heavily, so the
segmented EER is a different quantity, not a corrected one. `test_d5` pins the
order-of-magnitude gap so nobody compares the two by accident.

---

## P8  MLAAD has no EER until M-AILABS is pooled into it  [DONE 2026-08-03]

Closed by `scorepath.mlaad_pool_paths`, which returns the score files composing
the MLAAD column: one pre-pooled tsv under legacy, and the two single-class
files under v2/v3, concatenated at read time by `scorefile.read_scored`.

Verified by execution -- both layouts yield the same pool:

    legacy  1,040,006 rows   456,000 spoof + 584,006 bonafide   0 NaN
    v3      1,040,006 rows   456,000 spoof + 584,006 bonafide   0 NaN

and 39,000 utt_ids containing spaces are read intact on both, because the spoof
half is taken from its tab-separated twin.

Pooling at read time rather than on disk is the better arrangement, and not for
tidiness: a file that silently contains two corpora cannot be re-pooled at a
different ratio, cannot have either corpus's row count checked independently,
and answers "how many MLAAD utterances are there" with the wrong number.

Consumed by `recompute_main_results`, `verify_mlaad_column`,
`organize_mlaad_scores`, `build_mlaad_dir_map`, `create_mlaad_tts_eer_heatmaps`
and `tools/compare_trees`. Contracts in `tests/test_score_reading.py` (R3).

The MLAAD column now reproduces the published table from the v3 tree: 20 of 21
rows match to 3 decimals, AASIST is the single exception (published 21.942 vs
recomputed 26.336).

---

## P8-original  [superseded by the entry above]

Both are single-class, so neither yields an EER alone:

    Multilingual   456,000 spoof,    0 bonafide
    MAILABS        584,006 bonafide, 0 spoof
    pooled       1,040,006 -- exactly the paper's MLAAD row (n, n_bonafide, n_spoof)

The orchestrator will score both and mark them `ok`; any per-dataset EER for
either is undefined. **No tooling in the repo pools them** -- checked
`analysis/` and `recompute_main_results.py`. `MAILABS` is in
`NON_BENCHMARK`, which records the fact, but nothing consumes it.

This blocks the MLAAD column, not the sweep. Scoring can proceed; the merge has
to exist before that column can be reported.

---

## P9  The migration comparison reaches only 3 of 12 columns  [DONE 2026-08-03]

Resolved by the second option the entry proposed -- explicit per-dataset paths --
rather than by filling in `scorepath._LEGACY_LINEAR_HEAD` for the nine.

`tools/compare_trees` builds the legacy path for a column itself, from the tree
it was given, and does the same on the other side through `score_path`. So the
old tree stays readable without teaching `scorepath` to WRITE nine paths it
should never write again: the legacy convention is a fact about a tree that
already exists, not a layout the code should be able to produce.

All 10 benchmark columns now resolve on both sides. Two required normalising an
id convention rather than a path, which is recorded in P12.

---

## P9-original  [superseded by the entry above]

`score_path(layout="legacy")` raises `KeyError` for nine datasets -- the old tree
had no single naming convention for them, which `scorepath.py` documents
deliberately. So `--verify-against` silently reports "no reference" for `wild`,
`eval_2019`, both ASV21 columns, `asvspoof5`, `Famous_Figures`, `asvspoofLD` and
both Deepfake-Eval sets. Only `spoofceleb`, `Multilingual` and `MAILABS` resolve.

That undercuts step 2 of the promote sequence in P5. Either fill in
`scorepath._LEGACY_LINEAR_HEAD` for the nine, or let the comparison take
explicit per-dataset reference paths.

Blocks verification, not scoring.

---

## P10  ASVLD rebuilds a condition index the protocol already contains  [OPEN]

`_r_asvld` parses all five condition protocols into a 2,065,873-entry
utt_id -> condition map, because a bare utt_id does not say which
`ASVspoofLD/{cond}/flac/` directory holds its audio.

The combined protocol built by `build_protocols asvld` already carries a
`condition` column, but `trials_from_protocol` reads only `utt_col` and
`label_col` and discards the rest, so the resolver re-derives what it was handed.

Harmless -- the index is built once and cached -- but it is ~2M rows of parsing
per process for information already in hand, and the two sources could in
principle disagree. The fix is to let a dataset's parsed protocol carry extra
columns through to its resolver. Left open rather than changed under a running
sweep, since it touches the path every ASVLD trial resolves through.

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

---

## P12  Provenance of the published columns  [DONE 2026-08-03 -- record, not a change]

Measured by `tools/compare_trees` over all 10 benchmark columns x 19 paper
models (190 cells), legacy vs v3. Both levels the comparison needs are
reported: whether the two trees scored the same UTTERANCES, and whether they
assigned the same SCORES to the ones they share.

Reproduction is graded on the EER over the SHARED trials, because that is the
number the paper prints and the only one a reproduction claim is about.

### Result

| column     | reproduce | coverage change | verdict |
|---|---|---|---|
| MLAAD      | 19/19 | none            | bit-identical |
| ASV21 LA   | 18/19 | none            | reproduces |
| ASVLD      | 17/19 | +430,942 trials | reproduces |
| SpoofCeleb | 16/19 | none            | reproduces |
| ASV5 Eval  | 10/19 | none            | mixed |
| ASV21 DF   |  9/19 | +458,874 trials | mixed |
| FF         |  3/19 | +1,665 trials   | mostly differs |
| ASV19 LA   |  0/19 | none            | **does not reproduce** |
| ITW        |  0/19 | none            | **does not reproduce** |
| DFEval24   |  0/19 | 1,976 -> 56,481 | not comparable by design |

### What each row means

**MLAAD reproduces exactly.** All 19 cells are bit-identical, and 20 of the 21
published MLAAD values in `tab:results_main` recompute to 3 decimals from the
v3 tree. AASIST is the one exception: published 21.942, recomputed 26.336.

**Coverage growth is not divergence.** ASV21 DF, ASVLD and FF grew, and in
every one of those the old trials are a strict SUBSET of the new (`onlyA = 0`,
except one FF utterance). Restricted to the shared trials, ASVLD reproduces
17/19 and ASV21 DF 9/19. The old measurement is recoverable from the new tree;
the reverse is not.

**DFEval24 is a different measurement, not a broken one.** 1,976 whole-file
trials versus 56,481 four-second windows, zero overlap by construction. The
numbers are not comparable and should not be presented as a correction.

**ASV19 LA and In-the-Wild do not reproduce, and coverage does not explain it.**
Identical trial counts, identical ids, zero label disagreement -- and every one
of the 38 cells moves. Score correlation runs 0.72-0.95 and the EER moves by up
to 14.46 pp (ITW/tera). Resampling is ruled out: both corpora are natively
16 kHz, so RP-1 cannot apply. The pattern is per (dataset, model) cell rather
than per dataset, which points at the legacy tree having accreted from several
runs, checkpoints or precisions rather than at a single systematic cause.

**Consequence.** The published ASV19 LA and ITW columns cannot be regenerated
by the current pipeline. That is a provenance gap in the OLD tree, not a defect
in the new one -- but it means those two columns have no reproducible source.

### A caveat this surfaced, which matters for the paper

Score agreement and EER agreement are not the same question. Five cells have
score correlation >= 0.99999 and still move the EER by more than 0.05 pp;
SpoofCeleb/tera moves 4.15 pp on a maximum score difference of 0.043. Those are
models operating near chance, where the DET curve is flat at the crossing point
and a hair's movement reorders many trials.

Reporting a three-decimal EER for a model that cannot separate the classes
implies a precision that is not there. The `groups_mean_gt_50` and AUC
reporting already in `create_mlaad_tts_eer_heatmaps` makes the same point for
MLAAD.

### Reproducing this record

    python -m spoof_superb.tools.compare_trees \
        --a /data/ssl_anti_spoofing/asd_superb_score_files   --a-layout legacy \
        --b /data/ssl_anti_spoofing/spoof_superb_score_files --b-layout v3 \
        --out outputs/tree_comparison "--a-id-rewrite=-=Bonafide"

Two id conventions had to be reconciled first, and neither is a score
difference:

* Famous Figures records ABSOLUTE paths in the old tree and corpus-relative
  ones in the new. Stripped automatically -- a shared absolute prefix is
  detectable, and relative ids are never touched.
* Famous Figures names the bonafide directory `-` in the old tree and
  `Bonafide` in the new. NOT automatic: asserting that two conventions denote
  the same utterances is a claim the caller makes, so it is a command-line flag.

Without the second, the FF intersection is all-spoof and every FF cell reports
"not comparable" -- true, but it hides that the spoof half matched perfectly.

---

## P11  Analysis views for the v3 tree  [DONE 2026-08-03 -- D1-D7, D9 approved; D8 dropped]

Approved as proposed, less D8: normalised scores are not used, so no normalised
view exists and `apply_zscore_and_pool`, `apply_sigmoid_and_pool` and
`compute_eer_tts` stay legacy-only.

**Nothing has been written to the score tree.** It still holds `raw/` and
`_runs/` only. The builder defaults to writing beside `raw/`, but every run so
far used `--out_root` to a scratch directory or `--dry-run`. Building into the
score tree is a one-line command whenever you want it.

Shipped: `analysis/views.py` (the registry and `load_view`) and
`tools/build_view.py` (the materialiser). Contracts in `tests/test_views.py`
(V0-V5), 22 tests.

Verified by execution against v3, per model:

    asvld_conditions   29 groups   2,065,873 rows   2 levels
    mlaad_language     54 groups     456,000 rows   1 level   + 584,006 bonafide
    mlaad_tts          91 groups     431,000 rows   2 levels  + 584,006 bonafide

The MLAAD counts match `organize_mlaad_scores`' asserts exactly (431,000
retained, 25,000 excluded, 584,006 bonafide, 91 systems, 54 languages), and the
ASVLD view is a LOSSLESS PARTITION: concatenating all 29 groups and sorting
gives a file byte-identical to the raw score file. No row invented, none lost.

utt_ids containing spaces survive the round trip -- `Cartesia.ai (Sonic-3)` is
written and read back intact, because views hold the same canonical format raw
does rather than inventing one.

Two corpus metadata CSVs moved from the score tree into the repo, beside the
`mlaad_v10_dir_to_system.csv` that was already there:
`mlaad_v10_tts_architecture_groups.csv` and `mlaad_v10_table4_provenance.csv`.
They describe the corpus, not any model's scores, and keeping them in the score
tree meant they vanished the moment you pointed the tools at a different one --
which is exactly how the `mlaad_tts` view failed on first run.

### What the legacy tree did, and what it cost

Eight view trees, 23.1 GB, against 7.5 GB of raw score files -- roughly three
copies of the data, in trees that can drift from their source and from each
other:

    scores_by_acoustic_degradation  3.3G     scores_by_TTS_MLAAD      2.9G
    scores_by_category_augmented    2.6G     scores_by_MLAAD_language 2.9G
    scores_by_TTS                   2.6G     linear_head_normalized_scores  3.5G
    scores_by_TTS_norm              1.8G     normalized_scores_by_ssl_model 3.5G

The first two are the same view built twice, and `docs/09` already has to warn
which one to use ("the latter carries the old fp16-NaN noise scores and no
recompression"). That is the drift, already realised.

### The finding that decides the design

**A legacy view is a GROUP-BY over a raw file, and the grouping key is already
in the utt_id** -- for the MLAAD views. There is proof rather than an argument:
at Phase 2c `create_mlaad_tts_eer_heatmaps` was ported to read `raw/` plus the
directory map and produced all four figures with `scores_by_TTS_MLAAD` not
existing at all.

So a view is a QUERY, not a second copy. This is the same argument
`scorepath.py` already makes for ASVLD conditions -- "splitting by directory
would add a level that carries no information" -- applied one layer out.

### CORRECTION, twice: what the legacy degradation tree actually is

This entry got the degradation view wrong twice, and both are recorded because
the second correction reverses the first's implication.

**First claim (wrong):** the legacy degradation tree is a lossy view of raw
ASVLD. It is not -- its files carry `LA_E`, `DF_E` and `E_00` ids, so it spans
three corpora, and v3 holds `DF_E` and `E_00` clean only.

**Second claim (also wrong):** that the mixture was accretion, and a defect.
It is neither. It is the composition the paper specifies in Section 4.4.2 and
`tab:acoustic_degradation`, and the arithmetic confirms it exactly:

    condition        Table 5 composition                                observed
    Reverberation    ASVLD RT 210,191 + LA:C1 25,938 + DF:C1 17,131
                     + ASV5:C00 171,602                 =    424,862     424,862
    Bandwidth        ASVLD resample 284,948 + the same three
                                                        =    499,619     499,619
    Additive Noise   ASVLD noise 712,370 + the same three
                                                        =    927,041     927,041
    Channel          LA:C2-C7 155,628 + ASV19 71,237 + DF:C1 17,131
                     + ASV5:C11 46,610                  =    290,606     290,606
    Codec            ASVLD recompr 427,422 + LA:C1 + DF:C2-C9 137,048
                     + ASV5:C01-C10 466,100             =  1,056,508   1,051,746

Four exact, and Codec short by 4,762 (0.45%) -- decode failures, the same kind
of gap as the reverberation shortfall below.

Pooling clean corpora into every degraded condition is the point of the design,
not a flaw in it: each condition changes only the corpus under degradation, so
its EER is comparable to the Baseline. Measuring a degradation on its degraded
corpus alone would confound the degradation with that corpus's difficulty.

The one real finding from the audit stands: the legacy tree's reverberation
conditions are unequally populated -- RT_0_9 71,237, RT_0_6 70,533, RT_0_3
68,421, so 3,520 utterances are missing with no record of why. v3 has the full
71,237 for all three.

**What does not carry over.** The paper's DF:C1 is 17,131 utterances, and it is
not a protocol partition -- it is the `nocodec` subset of the DF trial list the
main benchmark column happened to score (152,955 of 611,829). Verified by
lookup: the two sets are identical. v3 scores the full DF set, so its DF:C1 is
all 67,981 `nocodec` trials. Every v3 condition count therefore exceeds the
paper's, for the coverage reason P12 already records, not a compositional one.

### Proposal

**Views are declared in code, materialised only on request.**

    {scores_root}/
      raw/                                 the only source of truth
      views/{view}/{group}/[{subgroup}/]{frontend}.txt
      views/{view}/_bonafide/{frontend}.txt
      views/{view}/_manifest.json

with a `VIEW_SPECS` registry naming, per view: its source dataset(s), the
function deriving the group key from the utt_id, and whether a shared bonafide
pool is attached. One command builds any of them.

**D1  `views/` is a sibling of `raw/`, never inside it.** `raw/*/*/model.txt`
stays exact.

**D2  The frontend is always the LAST path component.** So `views/*/*/xls_r_300m.txt`
still finds one model everywhere, exactly as in raw. Legacy broke this: the
degradation tree names files `APC.txt`, `Audio_Albert.txt`, `Byol-Audio.txt`
under four conditions and `linear_head_resamp_apc.txt` under the fifth. That
single inconsistency is why `compute_eer_matrix` carries three separate stem
dictionaries -- 60 lines of hand-maintained mapping that exist only to undo it.

**D3  Depth is 1 or 2 levels, chosen for readability, frontend last.**

    views/acoustic_degradation/Additive_Noise/babble_10/apc.txt
    views/mlaad_tts/AR/Bark/apc.txt
    views/mlaad_language/de/apc.txt

The coarse family is kept as a browsable level because it is what a reader
looks for, and the fine condition is kept as the leaf because it is recoverable
and the coarse one is not. Legacy had to choose one and chose the lossy one.

**D4  The group key is derived by code, never hand-built.** That is what makes
a view checkable against raw, rebuildable, and impossible to half-update.

**D5  `_bonafide/`, with the underscore.** Legacy used `bonafide/`, which is
indistinguishable from a TTS system named "bonafide" and sorts into the middle
of the systems.

**D6  Derived tables leave the score tree.** `eer_matrix.csv` currently sits
INSIDE `scores_by_acoustic_degradation/`. Results go to `outputs/`; the score
tree holds scores.

**D7  Every view carries `_manifest.json`** -- source dataset, layout, row count
per group, the raw files read with size and mtime, spec version, build time. So
"is this view stale" is answerable, which is the failure mode materialised
views always eventually hit and the one the two degradation trees already hit.

**D8  Normalised scores are a different kind of view and get raw's shape.**
z-score and sigmoid outputs are not a grouping -- same rows, different values --
so they mirror raw exactly:

    views/normalized/{method}/{method}/{dataset}/{frontend}.txt

which keeps them diffable against the raw file they came from, row for row.

**D9  DROPPED.** Views were proposed as optional, computed from raw by default.
Rejected: an analysis and the grouping it reports over must not be able to
disagree, and "the view on disk is stale" is a failure mode with no upside.
Each analysis entry point BUILDS its view and then reports over it, in one
command. `tools/build_view` remains for building one without running an
analysis, and `--dry-run` for seeing what it would write.

### As shipped

Two views, one per analysis beyond the main table:

    acoustic_degradation   6 conditions   Section 4.4.2, tab:acoustic_degradation
    tts_systems           91 systems      Sections 4.4.3 and 3.2.3

An earlier draft listed `asvld_conditions` and `mlaad_language` instead. Both
are gone: the first was a stand-in built while the degradation composition was
misread as ASVLD-only, and nothing reports over per-language MLAAD.

The degradation view needed a shape the first design could not express. Its
groups are POOLS of partitions of four corpora, named in advance because the
paper names them, where `tts_systems` PARTITIONS one dataset into groups
discovered from the data. Both yield {group: rows}, so the materialiser and
everything downstream is still written once.

Modules:

    analysis/conditions.py   which condition an utterance is in, per corpus
    analysis/views.py        the two specs, and load_view
    tools/build_view.py      the materialiser, a generator over models
    analysis/acoustic_degradation.py   entry point: build, then EER + dEER
    analysis/tts_systems.py            entry point: build, then 4 groupings

Condition codes come from each corpus's own protocol and were verified to cover
the v3 tree exactly, with nothing unmatched:

    asvspoof2021_la    181,566 rows    7 conditions x 25,938   C1  = none
    asvspoof2021_df    611,829 rows    9 conditions x 67,981   C1  = nocodec
    asvspoof5          680,774 rows   12 conditions            C00 = '-'
    asvspoof_ld      2,065,873 rows   29 conditions x 71,237   in the utt_id

The composed conditions, per model, on v3:

    Baseline               336,758        Additive_Noise      1,334,076
    Codec_Compression    1,459,770        Reverberation         479,232
    Bandwidth              550,469        Channel_Distortions   341,456

Each matches the Table 5 composition arithmetic exactly.

**The view changed no number.** `tts_systems` reproduces the older raw-reading
`create_mlaad_tts_eer_heatmaps` to max|d| = 0.000000 across all 11 architecture
groups, for both models checked.

### Run against v3, 19 models

    acoustic_degradation   6 conditions   4,501,761 rows/model   view 3.7 GB
    tts_systems           91 systems        431,000 rows/model   view 2.0 GB
                          + 584,006 shared bonafide

Both views are now on disk under `{scores_root}/views/`, each with a manifest
recording its sources, per-group row counts and build time.

Two results worth carrying into the paper:

* **Codec compression collapses the ranking.** The four strongest models under
  the Baseline (XLS-R 8.09, WavLM 8.73, MR-HuBERT 10.30, UniSpeech-SAT 10.55)
  degrade by +108% to +166%, while the weakest barely move (NPC +4%, VQ-APC
  +8%). Under codec every model lands between 21% and 36%. Bandwidth reduction
  does the opposite -- most models IMPROVE slightly.
* **Autoregressive systems are consistently harder to detect than
  non-autoregressive**, for 19 of 19 models. Among architecture groups,
  Flow + LLM is hardest (mean EER 43.69) and LLM -- the largest group, 34
  systems -- is 40.57.

One bug this surfaced: `_figures` wrote its absolute matrix as
`eer_by_condition.csv`, the same name the report used, so the figure step
clobbered the report -- no Model column, no dEER columns, no error. Fixed by
naming the figure CSVs `figure_*.csv`.

### Still open

`compute_eer_matrix` and `compute_eer_tts` are not ported. They read the legacy
view trees, and the two entry points above replace what they were for -- so the
question is whether to delete them rather than port them. `compute_eer_matrix`'s
three hand-maintained stem dictionaries exist only to undo the legacy naming
inconsistency that D2 forbids, so a port would delete them anyway.

---

## P13  The AASIST row cannot be reproduced; LFCC-GMM reproduces exactly  [DONE 2026-08-03 -- record]

`recompute_main_results` now computes the table's two non-SSL reference rows.
Comparing them against the published values separates cleanly, and the reason
is in the paper's own training protocol.

**LFCC-GMM reproduces exactly.** On ASV19 LA the two trees are bit-identical:
71,237 rows both sides, zero label disagreement, correlation 1.00000,
max|d| 0.0000, 100% of scores equal, EER 3.700 in both. Six of its ten columns
match the paper to three decimals; the four that move are exactly the four with
documented coverage growth (ASV21 DF, DFEval24, FF, ASVLD).

That is expected: "LFCC-GMM involves no gradient-based training: two
512-component diagonal-covariance mixtures are fitted by expectation
maximization". Refitting is deterministic given the same data.

**AASIST does not reproduce.** Same 71,237 trials, same labels, but correlation
0.91859, max|d| 10.42, and not one score equal. EER 1.659 published vs 3.223
recomputed. Every one of its ten columns moves, ASV5 Eval by 12.7 pp.

Also expected, and not a defect: "AASIST is trained end-to-end from random
initialization ... and the checkpoint with the lowest development EER is
retained". A re-run is a different model. The published AASIST column is a
property of a checkpoint, and cannot be regenerated without that checkpoint.

**Consequence.** Of the two reference rows, only LFCC-GMM is reproducible from
score files. Reporting the AASIST row alongside recomputed SSL rows compares a
retained checkpoint against a fresh one. Either the original AASIST checkpoint
is scored into the new tree, or the row is republished from the re-run with the
change stated.

This is the same class as the ASV19 LA / ITW finding in P12 -- scores moving on
identical trials -- but here the cause is known rather than unexplained.

## P14  Verification is a separate step, at two levels  [DONE 2026-08-03]

**The problem.** Comparison was embedded in three producers, and in each case
the producer graded itself:

* **Scoring.** `orchestration/driver.py` ran `verification.driver` on every
  finished file, against `--verify-against OLD_ROOT --verify-layout legacy`.
  A build that reads a score file it did not just write can only reproduce the
  older tree's coverage. The verdict landed in `run_status.json` and `SUMMARY.txt`
  and was never read again, and the grade came from per-dataset thresholds
  (`VERIFY_POLICY`) tuned to the legacy environment's constant logit offset.
* **Analysis.** `recompute_main_results` carried a "REPRODUCTION GATE": a dict
  of published values, in its own source, that it compared its own output
  against. An analysis marking its own homework cannot distinguish *the code
  changed* from *the scores changed*, and the reference was a literal nobody
  could refresh.
* Both pinned the **legacy layout**, which is no longer the authoritative tree.

**What replaced it.** One command, `python -m spoof_superb.verification`, with
two levels that fail independently.

### Level 1 -- score files

Two reference modes. Manifest mode reads `reference/manifest.json` and costs no
download; tree mode compares every utterance.

The manifest gained a `cells` block beside its existing per-file block, because
they answer different questions: `files` is what a DOWNLOAD needs (path, size,
sha256), `cells` is what VERIFICATION needs -- the POOLED counts and EER for a
benchmark column (MLAAD and ASVLD are two files each, so a per-file EER is not
the number the paper prints) plus a **digest of the sorted trial list**.

That digest is what makes offline verification meaningful rather than
suggestive. Matching row counts prove nothing: two different 71,237-trial sets
are still different trial sets, and without trial-set identity comparing two
EERs is meaningless. `build_release_manifest` also had to be made layout-aware
-- it enumerated through `reference_paths`, which is legacy-only, so it could
not index the v3 tree at all.

**What is reported.** Coverage in both directions, label agreement, sha256 and
`frac_exact`, non-finite counts, Pearson/Spearman/offset, and four EERs: each
tree's own, and each tree's restricted to the shared trials. The last pair is
the one a reproduction claim rests on -- `eer_a` vs `eer_a_common` is coverage
alone, `eer_a_common` vs `eer_b_common` is the scores alone.

**The verdict is a ladder, not a boolean.** `IDENTICAL`, `EQUIVALENT`,
`SENSITIVE`, `COVERAGE_DIFFERS`, `SCORES_DIFFER`, `LABELS_DIFFER`,
`CANDIDATE_INVALID`, `MISSING`, `ERROR`. Two of these are load-bearing:

* **`EQUIVALENT` is the target outcome, not `IDENTICAL`.** A different
  GPU/cuDNN/torch shifts every logit by a near-constant offset. A check that
  demands bit-exactness fails every honest reproduction and gets ignored.
* **`SENSITIVE` exists because P12 measured it.** Five cells have corr >= 0.99999
  and still move the EER past 0.05 pp; SpoofCeleb/`tera` moves 4.15 pp on a
  maximum score difference of 0.043. Near-chance models sit where the DET curve
  is flat. That is a caveat on the metric, not a defect in the run, so it does
  not fail.

Manifest mode cannot separate `SENSITIVE` from `SCORES_DIFFER` -- rank agreement
is not in the manifest. It reports `SCORES_DIFFER` and names `--ref-root` as the
way to decide, rather than guessing.

### Level 2 -- analysis tables

Reference is `reference/analysis/` -- the six tables the three analyses produce,
frozen by `tools/build_reference.py`. **Deliberately not the paper's LaTeX:**
P12 established that ASV19 LA and ITW do not regenerate from any score file in
either tree. A reference nobody, including us, can reproduce is not a reference.

**Grading on `max |delta|` would be wrong in both directions.** A run can miss
every cell by 0.3 pp and support every sentence in the paper; a run can miss one
cell by 0.4 pp and change which model is best on a column -- which is a
sentence. So three layers are reported: structure, cells (diagnostic), and
claims (the grade).

The claims checked are the paper's own: best-in-column with the non-SSL rows
excluded as the caption specifies, the top-five set under Mean, the **ordering
of columns by mean EER** (which IS the sentence in 4.4.2 / 4.4.3), per-column
rank correlation, **sign flips against the degradation Baseline**, and -- where
the CSV carries them -- the `*` emphasis markers compared directly, so the
published bolding is checked as published with no rule restated to drift from.

Verdicts: `IDENTICAL`, `EQUIVALENT`, `CONCLUSIONS_HOLD`, `CONCLUSIONS_DIFFER`,
`STRUCTURE_DIFFERS`, `MISSING`, `ERROR`.

### Verified by execution

Level 2 on four perturbations of the real reference tables, all six verdicts as
designed: jitter <= 0.02 pp -> `EQUIVALENT`; a runner-up promoted in ASV19 LA
-> `CONCLUSIONS_DIFFER` naming both the changed best-in-column and the moved
emphasis marker; AR/NAR swapped -> `CONCLUSIONS_DIFFER` on the column ordering;
a dropped model row -> `STRUCTURE_DIFFERS`; the two untouched tables
`IDENTICAL`. 23 new unit tests pin the ladder boundaries.

### Removed

`--verify-against` / `--verify-layout` and `_verify` / `reference_for` from the
orchestrator; `Task.verify` from `jobs.py`; `VERIFY_POLICY` / `verify_policy`
from `scoring/datasets.py`; `VERIFY_AGAINST` from `bin/orchestrate.sh`; and
`PUBLISHED` / `CHANGED_COLS` / `REPRO_TOL` / the gate from
`recompute_main_results`. `tests/test_paper_models.py` J3/J7/J8 now assert the
ABSENCE of the hook rather than its behaviour.

`verification/driver.py` and `verification/policies.py` are orphaned but not
deleted (deletion is the user's call); both carry a SUPERSEDED header.
`tests/test_verification.py` still pins the reasoning in them.

`tests/test_main_results_regression.py` keeps its baseline JSON untouched. It is
now explicitly a **code**-regression gate -- it pins the reproducer to the
legacy tree the baseline was captured on so a refactor cannot move a number --
not the reproducibility check a benchmark user runs. Only
`test_reproduction_gate_still_passes` was dropped, because the field it read no
longer exists.

### P14 addendum -- level 1 validated against the legacy tree

The self-check (v3 vs its own manifest) is 190/190 `IDENTICAL` and 6/6
`IDENTICAL`, exit 0 -- correct, but it only proves the plumbing.

The real test is tree mode against the legacy tree, which reproduces P12's
findings through the new ladder:

```
SpoofCeleb   16 EQUIVALENT   3 SENSITIVE
ASV19 LA     19 SCORES_DIFFER
```

| cell | trials | labels | spearman | max abs d | dEER pp | verdict |
|---|---|---|---|---|---|---|
| SpoofCeleb/`tera` | identical | agree | 1.000000 | 0.0428 | **4.149** | `SENSITIVE` |
| SpoofCeleb/`mockingjay_960hr` | identical | agree | 1.000000 | 0.0311 | 0.117 | `SENSITIVE` |
| SpoofCeleb/`wavlablm_ek_40k` | identical | agree | 0.999999 | 0.0469 | 0.052 | `SENSITIVE` |
| ASV19 LA/`wav2vec` | identical | agree | 0.81015 | -- | **3.170** | `SCORES_DIFFER` |

**The magnitude ordering is inverted relative to the verdict.** `tera` moves the
EER by 4.149 pp and PASSES; `wav2vec` moves it by 3.170 pp and FAILS. That is
the entire argument against grading on dEER alone, measured on real data rather
than argued: the two cells are distinguishable only by rank agreement, and rank
agreement is what a detection claim is about.

All 19 ASV19 LA cells have identical trial sets and zero label disagreement, so
the divergence is neither coverage nor protocol -- the same conclusion P12
reached, now produced automatically by a command anyone can run.

**Efficiency note.** Manifest mode originally parsed every candidate file before
comparing its sha256, which made the cheapest possible answer -- "byte-identical"
-- cost a full read of ~15 GB. It now hashes first and only parses when the hash
differs. On a fetched or re-verified tree that is every cell.

## P15  Retire the superseded verification unit and the legacy regression gate  [DONE 2026-08-03]

**Deleted.** `verification/driver.py`, `verification/policies.py`,
`verification/stats.py`, `tests/test_verification.py`,
`bin/watch_and_run_spoofceleb.sh`, `tests/test_main_results_regression.py`.

`stats.py` was not on the approved list but goes with the pair: nothing else
imported it. It existed only to supply `Comparison` to `policies.py` and
`compare()` to `driver.py`, so it was orphaned by their removal rather than by a
separate decision. `test_verification.py` tested exactly those three modules.

`watch_and_run_spoofceleb.sh` was a completed one-off whose canary gate called
`verification.driver` against the legacy tree.

**The regression gate.** `test_main_results_regression.py` re-ran the reproducer
against the LEGACY tree and diffed every EER against
`tests/baseline_main_results_table.json` at zero tolerance. It was answering the
right question in the wrong place: the reference was a capture of one tree at
one commit, unrefreshable without contradicting its own purpose, and it pinned
the reproducer to a tree that is no longer authoritative. P14's level 2 asks the
same question against a published artefact anyone can rebuild, and grades on
whether the paper's claims survive rather than on a tolerance.

**`tests/baseline_main_results_table.json` was NOT deleted, and must not be.**
This is the finding that changed the plan: `scoring/models.py::_slug_by_display`
reads it, so `paper_models()` -- and therefore every roster decision in
orchestration and all three analyses -- depends on it. It is the only record of
which score-file slug produced which printed row. Deleting it alongside its
test would have broken the repo.

That leaves a file under `tests/`, named for a job it no longer has, read by
production code. Recorded in `models.py` beside the constant. Moving and
renaming it is worth doing and is deliberately NOT bundled here: it is unrelated
to what retired the test, and the standing instruction is not to touch that
file's contents.

**Rewired.** `bin/reproduce_main_results.sh` `MODE="check"` ran the deleted
test. It now recomputes and then runs level-2 verification -- compute and check
as two visible steps in one script, rather than a pytest invocation that looked
like a unit test. Verified end to end: 6/6 `IDENTICAL`.

**Also committed:** `tests/test_migrate_layout.py`, which had been untracked.

Suite: 240 passed (was 246 passed / 3 skipped; the six removed were
`test_verification.py`, and the three skips were the opt-in regression gate).

## P16  Remove the vendored core_scripts tree  [DONE 2026-08-04]

**What it was.** A 38-file, 336 KB copy of Xin Wang's
project-NN-Pytorch-scripts (NII) -- a complete training framework: data IO,
`nn_manager` training loops, `op_manager` optimisers and LR schedulers, config
parsing. Every file carries `Copyright 2020, Xin Wang`.

**What was reachable.** One file. `main.py` imported `set_random_seed` from
`startup_config.py` and called it once. `startup_config.py` imports nothing
from `core_scripts` -- only stdlib, torch and numpy -- so the other 37 files
were unreachable: nothing outside the tree imported them, and the one reachable
file did not either. Every other `core_scripts` import in the repo was
`core_scripts` importing itself.

**It was already duplicated.** `main.py` imported TWO seeding functions:

    line 18  from spoof_superb.core.utils import create_optimizer, seed_worker, set_seed, str_to_bool
    line 24  from core_scripts.startup_config import set_random_seed

and called only the vendored one. `set_seed`, `seed_worker` and `str_to_bool`
were all dead imports.

They could not be swapped, and the reason is worth keeping: `str_to_bool` calls
`.lower()`, so the local `set_seed` -- which pushed `config["..."]` through it
-- raised `AttributeError` on argparse's real booleans, while the vendored one
read attributes directly. Two implementations of one thing, each unable to
accept the other's argument. That is why both survived.

**The licensing argument, which is the decisive one.** This repo is Apache-2.0.
`core_scripts/` carried another author's copyright headers and shipped NO
license file. project-NN-Pytorch-scripts is BSD-3-Clause, whose terms require
the notice to travel with the code. Shipping it in a public release was a real
problem independent of tidiness.

**What replaced it.** `core.utils.set_seed` now accepts None, an argparse
Namespace, or a mapping (`_toggle` converts strings only), sets PYTHONHASHSEED,
and preserves the two stdout notices.

**Verified by execution, before deleting anything.** A probe seeded with each
function over three configurations -- default toggles, flipped toggles, and
`config=None` -- and compared five draws each from torch, numpy and random,
plus PYTHONHASHSEED and both cuDNN flags. EQUIVALENT on every field. A training
run seeded after this change draws the same numbers as before it.

(The probe failed on the first attempt because the vendored function prints its
cuDNN notices to STDOUT, which corrupted the JSON it was writing there. The new
one preserves that behaviour, and `test_seeding.py` pins it.)

**Left alone, deliberately.** `seed_worker` is still not wired into main.py's
DataLoaders, which run `num_workers=8` with no `worker_init_fn` -- so the eight
workers are seeded by torch's default rather than by it. Passing it would
CHANGE TRAINING RESULTS, so it is a decision to take on purpose, not a tidy-up
to fold into a deletion. Recorded in the function's docstring.

Suite: 251 passed (11 new in `tests/test_seeding.py`).

## P17  Tier-1 cleanup, and the removal of the `sls` architecture  [DONE 2026-08-04]

**Deleted.** `analysis/create_tts_heatmaps.py`, `data/prep/make_tsv_mlaad.py`,
`data/prep/report_mlaad.py`, and the four one-off operational scripts
`bin/run_asvld_model.sh`, `run_noise_rerun.sh`, `run_recompression.sh`,
`watch_and_run_aasist_mlaad.sh`. Nothing imported any of them; the only
remaining mentions were in the historical records. Every path in the four shell
scripts pointed at the legacy score tree.

**`sls` removed entirely.** `models/sls.py` (293 loc) plus the `--model_arch`
choice, the `config.py` Literal, the dispatch branch in `main.py`, and the
mentions in `bin/train.sh`, `docs/07` and `README.md`.

The architecture had **never been runnable**. `main.py` called
`sls_model(args, device)` while its import sat commented out three hundred lines
above:

    # from sls_model import Model as sls_model

so `--model_arch sls` raised `NameError` on every invocation. `REORG_PLAN.md`
had recorded it as "live-but-unexercised, not dead" and moved it rather than
deleting it -- that judgement was made from the dispatch branch alone, without
checking that the name it called was bound. It was dead, and the reorganisation
carried it forward.

Removing it narrows what the benchmark advertises, which is why it was raised as
a decision rather than folded into a tidy-up.

**Two Tier-1 entries were withdrawn as wrong, not executed.** Both had been
listed as clutter; `git ls-files` and `git check-ignore` show neither is in the
repository:

* `outputs/` (170 MB) and `spoof_superb_outputs/` (7.4 MB) have **zero tracked
  files** and are gitignored. `spoof_superb_outputs/` is the configured
  `outputs_root` and holds the tables `reference/analysis/` was built from, plus
  every figure and both verification reports. Deleting them would have been pure
  local data loss with no effect on what ships.
* The root PDF is untracked, matched by `.gitignore` `*.pdf`.

Recorded in `CLEANUP.md` under "Withdrawn from Tier 1" so the error is visible
rather than silently dropped.

Suite: 251 passed.

## P18  The repository did not contain the package  [DONE 2026-08-04]

Found while relocating the roster mapping. `git rm tests/baseline_main_results_table.json`
failed with *pathspec did not match any files* -- the file every roster decision
depends on had never been committed, because `.gitignore`'s `*.json` caught it.
`paper_models()` therefore raised `FileNotFoundError` for everyone except the
author, on whose disk the untracked file happened to sit.

Widening the check found the larger problem. `.gitignore` line 2 was:

    _*

Git applies an unanchored pattern at EVERY depth, so this excluded:

  * all 12 `spoof_superb/**/__init__.py`  -- the package does not import
  * `spoof_superb/verification/__main__.py` -- the verification CLI
  * `bin/_common.sh` -- the preamble every shell script sources

and line 5's `models/` excluded `spoof_superb/models/__init__.py`. That
directory's other files survived only because they had been force-added before
the rule landed.

**A fresh clone could not import `spoof_superb`, could not run a single `bin/`
script, and had no model roster.** Confirmed by cloning: 0 of 12 `__init__.py`
present, no `_common.sh`.

Fixed by anchoring both rules to the repo root (`/_*`, `/models/`) -- where the
scratch and checkpoint directories they were written for actually live -- plus
explicit `!**/__init__.py`, `!**/__main__.py`, `!bin/_common.sh` and
`!/spoof_superb/scoring/paper_roster.json` re-includes, so a future broad rule
cannot remove the package again.

Verified against a fresh clone: package imports, roster resolves to 19 slugs,
250 passed / 1 skipped, `bin/orchestrate.sh --list` enumerates tasks.

**Why every check missed it.** Tests, analyses and verification all ran against
a working tree containing files the repository did not. Nothing in the suite
distinguished "present on disk" from "present in the repo", and no test cloned.
That gap is the actual defect; the two `.gitignore` lines are just where it
showed up.
