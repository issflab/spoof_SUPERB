# ASD-SUPERB

A SUPERB-style benchmark of self-supervised speech models for audio deepfake
detection. Given an audio signal, decide whether it is genuine (bona fide) or
spoofed — speech artificially generated or manipulated by text-to-speech or
voice conversion.

ASD-SUPERB extends [S3PRL](https://github.com/s3prl/s3prl), the SUPERB toolkit.

**Everything in the paper can be reproduced from the published score files — no
GPU, no checkpoints, no audio corpora required.** Start there.

---

## What you can do with this repo

Four things. Most people only ever want the first.

```
  0. install                conda env + configs/paths.yaml

  A. REPRODUCE              fetch score files ──▶ analyse ──▶ verify (level 2)
     the published results  ~6.5 GB, no GPU                   claims still hold?

  B. REBUILD                score ──▶ analyse ──▶ verify (levels 1 and 2)
     the score tree         GPU + corpora                     scores AND claims

  C. TRAIN                  train ──▶ then B
     new models             GPU + corpora
```

Nothing in A, B or C compares itself against anything. Verification is always a
separate step — see [why](docs/08-verification.md).

---

## 0. Install

```bash
git clone https://github.com/issflab/spoof_SUPERB.git
cd spoof_SUPERB

conda env create -f environment.yaml
conda activate spoof_SUPERB

$EDITOR configs/paths.yaml        # the only settings file
python -m spoof_superb.config     # check what it resolved to
```

Details: [Installation](docs/01-installation.md) ·
[Configuration](docs/03-configuration.md).

---

## A. Reproduce the published results

### A1. Get the score files

They are **not in this repo** — 234 files, 6.5 GB. What *is* in the repo is
`reference/manifest.json` (222 KB): every file's path, size, sha256, row and
class counts, EER, and a digest of its trial list. Files are fetched
individually, so checking one model on one dataset costs a megabyte rather than
a gigabyte, and every download is verified against its hash.

```bash
export SPOOF_SUPERB_SCORES_URL=...   # see the note below
bin/fetch_scores.sh --list           # what the manifest offers; fetches nothing
bin/fetch_scores.sh                  # edit DATASET / MODEL at the top first
```

Leave `DATASET` and `MODEL` empty in the script to fetch everything, which is
what reproducing the full tables needs.

> **The archive URL is not published yet.** The score files are being uploaded
> to Hugging Face; once they are, the URL will be recorded in the manifest and
> `bin/fetch_scores.sh` will need no environment variable. Until then, set
> `SPOOF_SUPERB_SCORES_URL` yourself.

Then point `scores_root` in `configs/paths.yaml` at wherever you put them.

### A2. Confirm the setup — the main table

```bash
bin/reproduce_main_results.sh
```

Recomputes the benchmark table (21 rows × 10 datasets + Mean) and checks it
against `reference/analysis/`. It reads score files only: no GPU, no
checkpoints, no audio. Expect:

```
=== VERDICTS ===
  IDENTICAL              1
```

If that passes, your installation and your score files are good.

### A3. All three analyses

```bash
bin/analyze.sh
```

Runs the three analyses in sequence, then level-2 verification over all six
tables:

| # | Analysis | Produces | Builds a view? |
|---|---|---|---|
| 1 | main results | the benchmark table | no — reads the raw tree |
| 2 | acoustic degradation (§4.4.2) | EER per condition + heatmaps | yes |
| 3 | TTS systems (§4.4.3, §3.2.3) | EER by system / architecture / mode / vocoder | yes |

Each analysis builds the grouping it reports over, so a number and the view
behind it cannot disagree. Set `VERIFY="no"` to skip the check, or `WHICH` to
run a subset.

Details: [Reproducing the published results](docs/02-reproducing-results.md) ·
[Analysis](docs/09-analysis.md).

---

## B. Rebuild the score tree

Needs the corpora on disk and a GPU. Read
[datasets and protocols](docs/04-datasets.md) first — this is where most of the
work is.

```bash
bin/score.sh          # one model, one evaluation set
bin/orchestrate.sh    # every model for a job, across GPUs, with resume + retry
```

Then run the analyses as in **A3**, and verify **both** levels:

```bash
bin/verify.sh         # LEVEL="all"
```

Level 1 asks *did the pipeline produce the same scores?* Level 2 asks *do the
same conclusions come out?* They fail independently, and both are worth
knowing: identical scores with a changed table means the analysis code moved;
drifting scores with an intact table means the finding is robust to the drift.

Level 1 is only meaningful here. In path A you downloaded the reference, so
verifying it against itself proves nothing.

Details: [Scoring](docs/05-scoring.md) ·
[Orchestration](docs/06-orchestration.md) ·
[Verification](docs/08-verification.md).

---

## C. Train new models

```bash
bin/train.sh            # SSL linear heads and the torch baselines
bin/train_lfcc_gmm.sh   # the LFCC-GMM baseline (CPU)
```

`main.py` is the underlying entry point. Then continue at **B**.

Details: [Training](docs/07-training.md).

---

## Documentation

| # | Document | Read it when |
|---|---|---|
| 1 | [Installation](docs/01-installation.md) | first, always |
| 2 | [Reproducing the published results](docs/02-reproducing-results.md) | you want the paper's tables back |
| 3 | [Configuration](docs/03-configuration.md) | your data is not where ours is |
| 4 | [Datasets and protocols](docs/04-datasets.md) | before scoring — what each corpus must look like on disk |
| 5 | [Scoring](docs/05-scoring.md) | you have a checkpoint and want score files |
| 6 | [Orchestration](docs/06-orchestration.md) | you want to score many models unattended |
| 7 | [Training](docs/07-training.md) | you want to train a new model |
| 8 | [Verification](docs/08-verification.md) | you rebuilt the tree or re-ran the analyses and want to check them |
| 9 | [Analysis](docs/09-analysis.md) | you want to regenerate a table or figure |
| 10 | [Tests](docs/10-testing.md) | you changed the code |
| 11 | [Troubleshooting](docs/11-troubleshooting.md) | something broke |

---

## The shell scripts

Each script in `bin/` opens with a settings block you edit, rather than
requiring a long command line:

```bash
$EDITOR bin/score.sh     # set MODEL, SSL_MODEL, DATASET, OUTPUT_FILE, ...
bin/score.sh
```

They print the command they run and pass extra arguments through, so
`bin/score.sh --limit 300` still works for a one-off.

| Script | Does |
|---|---|
| `bin/fetch_scores.sh` | download published score files, checked against the manifest |
| `bin/reproduce_main_results.sh` | recompute the main table and check it |
| `bin/analyze.sh` | run all three analyses, then verify them |
| `bin/verify.sh` | check a finished run against the reference, both levels |
| `bin/score.sh` | score one model on one evaluation set |
| `bin/orchestrate.sh` | score every model for a job, across GPUs |
| `bin/train.sh` | train an SSL or torch baseline model |
| `bin/train_lfcc_gmm.sh` | train the LFCC-GMM baseline (CPU) |

---

## Layout

```
configs/paths.yaml     your settings — the only file you need to edit
reference/             what a reproduction is checked against
├── manifest.json      every published score file: sha256, counts, EER, trial digest
└── analysis/          the six analysis tables, with provenance
bin/                   editable shell scripts, one per task
spoof_superb/
├── config.py          settings schema and loader (code, not settings)
├── core/              metrics.py (EER, t-DCF), scorefile.py (the 4-col format)
├── models/            aasist, aasist_raw, linear_head, lfcc_gmm
├── frontends/         lfcc, rawboost
├── data/              dataset classes, prep/ (MLAAD and M-AILABS preparation)
├── scoring/           ONE scoring entry point + dataset registry + back-ends
├── orchestration/     ONE job runner (GPU pool, resume, retry) + job specs
├── analysis/          the three analyses, their views, and the figures
├── verification/      the separate two-level check: scores, then analysis
├── train/             LFCC-GMM trainer
└── tools/             view builder, layout migration, reference publishing
tests/                 contract tests, all on synthetic fixtures
main.py                training entry point
```

Also in the repo: `humanpending.md` (open items needing a decision),
`CLEANUP.md` (what is still to be trimmed before release), `PLANNED_CHANGES.md`
and `REORG_PLAN.md` (the design record behind the current layout). These are
working documents, not user documentation.

---

<details>
<summary><b>Legacy command migration</b> — only if you used the pre-reorganisation flat scripts</summary>

| Before | Now |
|---|---|
| `python eval_baselines.py --list_datasets` | `bin/score.sh --list_datasets` |
| `python eval_baselines.py --model M --dataset D ...` | `bin/score.sh` (set `MODEL`, `DATASET`) |
| `python eval_asvld.py --condition C ...` | `bin/score.sh` (set `SOURCE="asvld"`, `ASVLD_CONDITION`) |
| `python eval_mlaad.py ...` | `bin/score.sh` (set `SOURCE="walk"`) |
| `python eval_mlaad.py --protocol_csv ...` | `bin/score.sh` (set `SOURCE="protocol_csv"`) |
| `python orchestrate_mlaad.py` | `bin/orchestrate.sh` (set `DATASETS="Multilingual"`) |
| `python orchestrate_mailabs.py` | `bin/orchestrate.sh` (set `DATASETS="MAILABS"`) |
| `python orchestrate_spoofceleb.py` | `bin/orchestrate.sh` (set `DATASETS="spoofceleb"`) |
| `python orchestrate_baselines.py` | `bin/orchestrate.sh` (set `JOB="baselines"`, `WORKERS=1`) |
| `python verify_mlaad.py` / `verify_spoofceleb.py` | `bin/verify.sh` |
| `python verify_asvld.py` | `python -m spoof_superb.verification.asvld_report` |
| `python verify_noise_rerun.py` | `python -m spoof_superb.verification.noise_rerun_gate` |
| `python scripts/<name>.py` | `python -m spoof_superb.analysis.<name>` |
| `from evaluation import compute_eer` | `from spoof_superb.core.metrics import compute_eer` |
| `.asvld_skip` containing `Filtering` | `--skip_conditions Filtering` (the default) |

</details>

---

## Citation

If you use this benchmark, please cite the accompanying paper.

## License

See [LICENSE](LICENSE).
