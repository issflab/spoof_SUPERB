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

### CORRECTION: acoustic degradation is NOT ASVLD

An earlier version of this proposal claimed the legacy degradation tree was a
lossy view of `raw/linear_head/asvspoof_ld/`. **That is wrong**, and it was
load-bearing, so it is corrected here rather than quietly edited.

The v3 ASVLD decomposition is right: 29 condition suffixes, 71,237 utterances
each, totalling exactly the file's 2,065,873 rows.

    babble|cafe|street|volvo|white _ {0,10,20}   15
    RT_{0_3,0_6,0_9}                              3
    resample_{8000,11025,22050,44100}             4
    recompression_{16k..320k}                     6
    lpf_7000                                      1

What is wrong is the claim that the legacy view is the same population. It is a
MIXTURE OF THREE CORPORA, and mostly untagged:

    file                  LA_E      DF_E      E_00      condition-tagged?
    Additive_Noise      738,308    17,131   171,602     no
    Channel_Distortions 226,865    17,131    46,610     no
    Codec_Compression   453,360   135,824   462,562     partly (6 x 71,237)
    Reverberation             --        --        --    yes (3 x ~70,000)
    Resampling                --        --        --    yes (4 x 71,237)

`DF_E_*` is ASVspoof2021 DF and `E_00*` is ASVspoof5. The v3 tree has scores
for those corpora **clean only** -- 0 of their utt_ids carry any condition
suffix. The degraded audio for them was never scored into v3.

So the legacy Section 5.2 population **cannot be rebuilt from the v3 raw tree**.
Doing so needs the degraded ASV19/DF/ASV5 audio re-scored, which is a scoring
job, not an analysis one, and is not started here.

Two further things the audit turned up, both properties of the legacy tree:

* Its reverberation conditions are not equally populated -- RT_0_9 has 71,237
  rows but RT_0_6 has 70,533 and RT_0_3 has 68,421. 3,520 utterances are
  missing, silently, with no record of why.
* `Codec_Compression` mixes 624,324 untagged rows in with the six tagged
  recompression conditions, so an EER computed over that file is over a
  population that cannot be described.

### What this changes in the proposal

D1-D7 and D9 are unaffected -- they are about shape, and the shape is right.

The view list changes. Two views are fully derivable and are built now; the
third is renamed to what it actually is:

    mlaad_tts          from mlaad_v10 + mailabs      derivable, verified
    mlaad_language     from mlaad_v10 + mailabs      derivable, verified
    asvld_conditions   from asvspoof_ld              derivable, but NOT the
                       (29 conditions, 5 families)   legacy 5.2 population

`asvld_conditions`, not `acoustic_degradation`, for the reason `scorepath.py`
already gives for writing `mlaad_v10` rather than `mlaad`: two different
populations must not share a name. A reader who sees `acoustic_degradation` in
the v3 tree would reasonably assume it reproduces the published Section 5.2,
and it does not -- it is a cleaner, per-condition measurement over one corpus.

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

**D9  Materialising stays optional.** Analysis that can compute from raw should,
as the MLAAD figures now do. Build a view to browse it, to feed a tool that
needs directories, or to publish a subset -- not as a precondition for a number.

### What this costs and what it needs

Roughly one new module (`tools/build_view.py` plus the `VIEW_SPECS` registry),
and a follow-up port of `compute_eer_matrix` and `compute_eer_tts` onto the
view paths -- at which point `compute_eer_matrix`'s three stem dictionaries
delete themselves, because D2 makes them unnecessary.

**Questions that need answering before any of it starts:**

1. **Materialise, or compute from raw?** D9 says both, defaulting to computing.
   If you would rather the views simply exist on disk as before, say so -- it is
   a different design, not a tuning knob.
2. **Fine or coarse degradation leaf?** D3 proposes both levels. The fine level
   is new capability (per-SNR, per-bitrate) the paper does not currently report.
3. **Do the normalised trees carry over at all?** They feed `compute_eer_tts`
   and `plot_score_distributions`. Nothing in the main results depends on them.
4. **Is `views/` the right word?** It is yours, from the request. `derived/`
   is the alternative.
