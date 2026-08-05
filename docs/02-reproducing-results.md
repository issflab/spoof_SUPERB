# 2. Reproducing the published results

**This is the primary goal of the repository.** Every number in the paper's
main tables is recomputed from published score files. You need no GPU, no model
checkpoints and no audio corpora -- only the score files.

## What a score file is

Every model-on-dataset run produces one plain-text file, four space-separated
columns:

```
LA_E_1234567 - spoof -3.4521
LA_E_7654321 - bonafide 2.1043
```

`{utt_id} - {key} {score}`. The second column is an unused placeholder. `score`
is the model's class-1 logit; higher means more bonafide. The EER of a file is
computed from these directly, so **anyone with the score files can recompute
every published number**, which is what this section does.

Fields are read from the **right**, not split on whitespace, because utt_ids
legitimately contain spaces -- MLAAD v10 has 39,000 rows with TTS system
directories like `Cartesia.ai (Sonic-3)`. Where that matters, a tab-separated
`.tsv` twin is written alongside.

## Step 1 -- get the score files

They are **not in this repo**: 234 files, 6.5 GB. What is in the repo is
`reference/manifest.json` (222 KB) -- every file's path, byte size, sha256, row
and class counts, EER, and a digest of its trial list.

```bash
export SPOOF_SUPERB_SCORES_URL=...   # see the note below
bin/fetch_scores.sh --list           # what the manifest offers; fetches nothing
bin/fetch_scores.sh                  # edit DATASET / MODEL at the top first
```

Files are fetched individually, so checking one model on one dataset costs a
megabyte rather than a gigabyte. Every download is checked against its sha256
and written atomically; a file already present and matching is skipped, so
re-running is cheap. Leave `DATASET` and `MODEL` empty to fetch everything,
which is what reproducing the full tables needs.

> **The archive URL is not published yet.** The score files are being uploaded
> to Hugging Face. Once they are, the URL is recorded in the manifest and
> `bin/fetch_scores.sh` needs no environment variable. Until then, set
> `SPOOF_SUPERB_SCORES_URL` yourself.

If you already have a score tree, skip the download and go straight to step 2.

## Step 2 -- point the repo at them

Edit `configs/paths.yaml`:

```yaml
scores_root: /data/ssl_anti_spoofing/spoof_superb_score_files
```

Confirm it took effect:

```bash
python -m spoof_superb.config | head -3
```

Nothing else in that file matters for reproduction.

## Step 3 -- the main table

```bash
bin/reproduce_main_results.sh
```

This recomputes every per-model, per-dataset EER behind the paper's main table
and then verifies it against `reference/analysis/`. It takes about 2m40s and
reads roughly 15 GB. No GPU, no checkpoints, no audio.

Expected output ends with:

```
=== VERDICTS ===
  IDENTICAL              1
```

The verification is graded on whether the paper's CLAIMS survive -- which model
is best in each column, the top five under Mean, the ordering of the columns by
mean EER -- with the per-cell deltas reported beside them as diagnostics. See
[verification](08-verification.md).

Set `MODE="compute"` to recompute without checking.

If this passes, your installation and your score files are good.

## Step 4 -- all three analyses

```bash
bin/analyze.sh
```

Runs main results, then acoustic degradation (Section 4.4.2), then TTS systems
(Sections 4.4.3 and 3.2.3), then level-2 verification over all six tables. The
latter two build the view they report over as they go; main results reads the
raw tree directly and has nothing to group by.

`WHICH` runs a subset, `VERIFY="no"` skips the check, `OUT_ROOT` moves the
output. See [analysis](09-analysis.md) for what each one produces.

To recompute the main table directly:

```bash
python -m spoof_superb.analysis.recompute_main_results --out_dir outputs/main_results
```

That writes `main_results.json` plus a printed table.

## What the reproducer actually does

It is the authority on what the main table reports, and two columns are not the
obvious file:

| Column | Source |
|---|---|
| 7 standard datasets | `{scores_root}/linear_head/linear_head_{dataset}_{ssl}.txt` |
| **MLAAD** | `linear_head_MLAAD_v10/` (1,040,006 rows) -- *not* the legacy `linear_head_Multilingual` (307,998). A different corpus scale entirely. |
| **ASVLD** | `linear_head_asvspoofLD` **pooled with** `asvld_rerun/Recompression`. Reading only the first file silently reproduces a stale column. |
| SpoofCeleb | the `linear_head_SpoofCeleb/` re-run (0 NaN), not the legacy files |

**Mean** is the arithmetic mean of the 10 per-dataset EERs. **Pooled** is the
raw scores concatenated across all 10 datasets with the EER taken once, with
**no normalisation** -- the sigmoid/z-score combined directories do *not*
reproduce the published table.

EER convention throughout: bonafide is the target class, higher score = more
bonafide.

## Expected warnings

A clean run still prints a "PROBLEMS" section. These are **known and expected**,
not a broken setup:

* `FBANK: MLAAD EER 52.414 >= 50%` and similar -- weak models genuinely score
  above chance on MLAAD.
* `Mockingjay: MISSING MLAAD v10 file` -- that model has no v10 score file, so
  its MLAAD/Mean/Pooled cells are legitimately empty.
* `TERA / Mockingjay-960h: ASVLD NaN fraction 23.497%` -- fp16 overflow in the
  original scoring of the masked-spectrogram front-ends. This is the defect the
  fp32 re-run exists to correct, and it is why `--amp` is off by default
  everywhere.

The gate fails only if the *set* of problems grows.

## Other published artefacts

```bash
# independent cross-check of the MLAAD column against the LaTeX source
python -m spoof_superb.analysis.verify_mlaad_column --tex access.tex
```

`verify_mlaad_column` checks three things: the number printed in the paper
against a fresh recomputation (<= 0.0005), the repo's EER estimator against an
independent sklearn/Brent implementation (<= 0.01 pp), and the full pool
against the balanced pool (<= 0.2 pp). The duplicate estimator is deliberate --
it exists so a bug in the repo's own EER code cannot be reproduced by its own
verifier.

The degradation matrix and the TTS figures are produced by `bin/analyze.sh`
(step 4). The older chain that read legacy view trees has been removed; see
[analysis](09-analysis.md).

## Next

To produce *new* score files from checkpoints, you need the corpora on disk:
[datasets and protocols](04-datasets.md), then [scoring](05-scoring.md).
