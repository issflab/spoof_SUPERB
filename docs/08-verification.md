# 8. Verification

Verification is a **separate step**. Nothing in scoring, orchestration or
analysis compares itself against anything.

That is a design decision, not an omission. A comparison that runs inside the
producer has three faults, and this repo had all three:

* **Scoring** graded each finished file against an older tree as the sweep ran.
  A build that reads a score file it did not just write can only ever reproduce
  the older tree's coverage, and the verdict landed in a run status file nobody
  opened again.
* **`recompute_main_results`** carried a "REPRODUCTION GATE" comparing its own
  output against a dict of published values in its own source. An analysis
  marking its own homework cannot distinguish *the code changed* from *the
  scores changed*, and the reference was a literal that no one could refresh.
* Both graded against the **legacy layout**, which pinned a tree that is no
  longer authoritative. That layout, and the intermediate `v2`, have since been
  retired entirely -- see `core.scorepath`.

All of that is gone. One command replaces it:

```bash
python -m spoof_superb.verification all      # or: bin/verify.sh
```

## The two levels

| | asks | reference | fails when |
|---|---|---|---|
| **1. Score files** | did the pipeline produce the same scores? | `reference/manifest.json`, or a reference score tree | scores disagree on identical trials |
| **2. Analysis** | do the same conclusions come out? | `reference/analysis/*.csv` | a claim in the paper changed |

They fail **independently**, and both are worth knowing. Identical scores with a
changed table means the *analysis code* moved. Drifting scores with an intact
table means the *finding is robust* to the drift. A single pass/fail cannot say
either.

```bash
python -m spoof_superb.verification scores      # level 1 only
python -m spoof_superb.verification analysis    # level 2 only
python -m spoof_superb.verification all         # both; non-zero if either fails
```

Each writes `{outputs_root}/verification/{level}/*.md` and `*.json` — the
Markdown is what you paste into an issue, the JSON is what a script reads.

## Level 1 — score files

```bash
# offline: reference/manifest.json only, no download
python -m spoof_superb.verification scores

# full: every utterance compared against a reference tree
bin/fetch_scores.sh
python -m spoof_superb.verification scores \
    --ref-root /path/to/reference/tree
```

### What is reported, and why each field is there

The whole problem is that two trees disagree for two unrelated reasons — they
**scored different utterances**, or they **assigned different scores to the same
ones** — and a single EER delta cannot tell them apart. So the report separates
them by construction:

| reported | answers |
|---|---|
| `n_a`, `n_b`, `n_common`, `n_only_a`, `n_only_b` | coverage, in both directions |
| `label_mismatch` | do the two runs agree on ground truth at all |
| `sha256`, `frac_exact`, `max_abs_diff` | integrity, and how far individual scores moved |
| `nan_a`, `nan_b` | is either side's own output usable |
| `corr`, `spearman`, `mean_offset ± std` | agreement on value and on rank |
| `eer_a`, `eer_b` | what each side would publish, on its own trials |
| **`eer_a_common`, `eer_b_common`** | **each side's EER on the shared trials** |

The last row is the one a reproduction claim rests on. `eer_a` vs
`eer_a_common` is the effect of coverage alone; `eer_a_common` vs
`eer_b_common` is the effect of the scores alone, trial set held fixed.

### The verdict ladder

Best to worst. Only some of these are anybody's fault.

| verdict | means | fails |
|---|---|---|
| `IDENTICAL` | byte-for-byte the reference | |
| `EQUIVALENT` | same trials, EER agrees within 0.05 pp | |
| `SENSITIVE` | same trials, ranks agree, EER still moved | |
| `COVERAGE_DIFFERS` | different trial sets; EERs below are on the intersection | |
| `SCORES_DIFFER` | same trials, scores genuinely disagree | ✗ |
| `LABELS_DIFFER` | a shared utt_id carries a different key | ✗ |
| `CANDIDATE_INVALID` | your output has > 1% NaN/inf | ✗ |
| `MISSING` / `ERROR` | no candidate file / the comparison raised | ✗ |

**`EQUIVALENT` is the target outcome, not `IDENTICAL`.** A different
GPU/cuDNN/torch shifts every logit by a near-constant offset. Demanding
bit-exactness would fail every honest reproduction, and a check people learn to
ignore is worse than no check.

**`SENSITIVE` exists because it was measured.** Across 190 cells:

```
corr >= 0.99999   n=92   median dEER 0.0009 pp   max 4.15 pp
corr <  0.99999   n=79   median dEER 1.6345 pp   max 14.46 pp
```

Five cells have essentially perfect score correlation and still move the EER
past tolerance — SpoofCeleb/`tera` moves **4.15 pp on a maximum score difference
of 0.043**. Those are models operating near chance, where the DET curve is flat
at the crossing point, so a hair's movement reorders many trials. That is a
caveat on reporting a three-decimal EER for a model that cannot separate the
classes; it is not a defect in the run, so it does not fail.

### Manifest mode vs tree mode

Manifest mode costs no download. It works because the manifest carries a
**digest of each cell's sorted trial list** — matching row counts prove nothing,
since two different 71,237-trial sets are still different trial sets, and
without trial-set identity comparing two EERs is meaningless.

What manifest mode *cannot* do is separate `SENSITIVE` from `SCORES_DIFFER`,
because that needs rank agreement between two score vectors and the manifest
holds no per-utterance scores. It reports `SCORES_DIFFER` and says so, naming
`--ref-root` as the way to decide. It never guesses.

## Level 2 — analysis tables

```bash
python -m spoof_superb.verification analysis --candidate outputs
```

Six tables, produced by the three analyses:

```
main_results/main_results_table.csv
degradation/eer_matrix.csv
tts/eer_by_tts_system.csv        tts/eer_by_architecture.csv
tts/eer_by_generation_mode.csv   tts/eer_by_vocoder_family.csv
```

### Why not just diff the numbers

Grading on `max |delta|` is wrong in both directions. A run can miss every cell
by 0.3 pp and still support every sentence in the paper. A run can miss one cell
by 0.4 pp and change which model is best on a column — which *is* a sentence. So
three layers are reported, in this order:

1. **Structure** — which models and columns each side has. A missing row is an
   absent measurement, not a small number, and no delta describes it.
2. **Cells** — per-cell |Δ| in pp: max, median, and the count over tolerance,
   with the worst named. This is the *diagnostic* layer: it says where to look,
   not whether you passed.
3. **Claims** — the things the paper asserts, one at a time:
   * which model is best in each column (an argmin per column, with the non-SSL
     reference rows excluded, as the caption specifies)
   * the top-five set under Mean
   * **the ordering of the columns by their mean** — "which degradation hurts
     most", "which architecture group is hardest" are literally the sentences in
     §4.4.2 and §4.4.3
   * the model ordering within each column (rank correlation)
   * **sign flips against the Baseline** in the degradation table — "this
     condition hurts" reversing is a finding, however small the cells moved
   * where the CSV carries the paper's own `*` emphasis markers, the marked
     cells are compared directly, so the published bolding is checked *as
     published* with no rule restated here to drift from the one that wrote it

| verdict | means | fails |
|---|---|---|
| `IDENTICAL` | every cell within 0.005 pp | |
| `EQUIVALENT` | cells drift, every claim and ranking survives | |
| `CONCLUSIONS_HOLD` | rankings shift among near-ties, headline claims survive | |
| `CONCLUSIONS_DIFFER` | a best-in-column, a top-5 set, or an ordering changed | ✗ |
| `STRUCTURE_DIFFERS` | different models or columns, or a one-sided cell | ✗ |

The cell tolerance (0.005 pp) is tighter than level 1's (0.05 pp) because these
are recomputations over *fixed* score files — arithmetic, not inference — so the
only source of drift is the analysis code itself.

## Publishing a reference

Run once when releasing a score tree. Both artefacts go in git; the 6 GB of
score files do not.

```bash
# reference/manifest.json -- per-file integrity + per-cell trial digests and EER
python -m spoof_superb.tools.build_release_manifest --dry-run
python -m spoof_superb.tools.build_release_manifest --archive_url https://...

# reference/analysis/ -- the six tables, plus provenance.json and REFERENCE.md
python -m spoof_superb.tools.build_reference --from outputs
```

`build_reference` copies exactly the list `verification.analysis.TABLES`
verifies, so the published set and the checked set cannot drift apart. It
refuses to write a partial reference: a missing table would silently exempt
that table from verification.

**The reference is not the paper's LaTeX.** Two published columns (ASV19 LA and
ITW) do not regenerate from any score file in either tree, on identical trials
with zero label disagreement. A reference nobody — including us — can reproduce
is not a reference. The tables here were computed by code that ships in this
repo from score files whose sha256 is published, so every number has a path back
to bytes anyone can download.

## Getting the reference score files

```bash
bin/fetch_scores.sh --list        # what the manifest offers, fetch nothing
bin/fetch_scores.sh               # edit DATASET / MODEL at the top first
```

Files are fetched individually, so checking one model on one dataset costs a
megabyte rather than a gigabyte. Every download is checked against its sha256
and written atomically; a file already present and matching is skipped.

Nothing derived from the score files is committed beyond the manifest. Trial
lists and score subsamples were considered and rejected: they are recomputable,
so shipping them would put two copies of the same information under version
control with no record of which is authoritative — the pattern that produced
the duplication in the score directory.

## Comparing two arbitrary trees

For anything that is not "candidate vs published reference":

```bash
python -m spoof_superb.tools.compare_trees \
    --a /path/to/reference/tree \
    --b /path/to/your/tree \
    --out outputs/tree_comparison "--a-id-rewrite=-=Bonafide"
```

Same measurement code (`verification.cells`), an older verdict vocabulary, and
no notion of a published reference. `--a-id-rewrite` renames whole path
components before matching; Famous Figures needs it because the old tree names
the bonafide directory `-` and the new one names it `Bonafide`. It is
deliberately never inferred — asserting that two id conventions denote the same
utterances is a claim the caller makes.

## Other checks, unchanged

### The fp32 ASVLD noise re-run promotion gate

```bash
python -m spoof_superb.verification.noise_rerun_gate            # verify only
python -m spoof_superb.verification.noise_rerun_gate --promote  # verify, then swap
```

Gates replacing archived score files with the fp32 re-run. Five contracts: same
utterance sequence and order, same labels, no NaN, Pearson ≥ 0.9998 on
utterances the archive scored finitely, and |ΔEER| ≤ 0.15 pp for models that had
no archived NaN. Models *with* archived NaN are exempt by construction —
correcting them is the point, and their EER moves by 5.8 to 8.0 pp.

**`--promote` moves directories.** Run without it first and read the table.

### Descriptive and structural checks

```bash
# ASVLD rerun vs reference, per (model, condition). No pass/fail, just a table.
python -m spoof_superb.verification.asvld_report

# the TTS protocol CSVs actually match the raw dataset protocols
python -m spoof_superb.analysis.verify_tts_protocols --master M.csv --lookup L.csv

# the paper's MLAAD column, three ways
python -m spoof_superb.analysis.verify_mlaad_column --tex access.tex
```

`verify_mlaad_column` checks the number printed in the paper against a fresh
recomputation (≤ 0.0005), the repo's EER estimator against an independent
sklearn/Brent one (≤ 0.01 pp), and the full pool against the balanced pool
(≤ 0.2 pp). **Its duplicate EER implementation is deliberate** — a bug in
`core/metrics.py::compute_det_curve` must not be reproducible by the very code
verifying it. Do not "deduplicate" it.

## Which one do I want?

| Situation | Use |
|---|---|
| I rebuilt the score tree and want to know if it reproduces | `python -m spoof_superb.verification scores` |
| I re-ran the analyses and want to know if the paper's claims hold | `python -m spoof_superb.verification analysis` |
| Both | `python -m spoof_superb.verification all` |
| I have two arbitrary trees to compare | `tools.compare_trees` |
| I re-ran ASVLD noise in fp32 and want to replace the archive | `noise_rerun_gate` |
| I changed code and want to know if it still runs | [tests](10-testing.md) |
| I want to know if my protocol CSVs are self-consistent | `verify_tts_protocols` |
