# ASD-SUPERB

A SUPERB-style benchmark of self-supervised speech models for audio deepfake
detection. Given an audio signal, decide whether it is genuine (bona fide) or
spoofed — speech artificially generated or manipulated by text-to-speech or
voice conversion.

ASD-SUPERB extends [S3PRL](https://github.com/s3prl/s3prl), the SUPERB toolkit.

**Everything in the paper can be reproduced from published score files — no
GPU, no checkpoints, no audio corpora required.** Start there.

---

## Quick start

```bash
# 1. install
conda env create -f environment.yaml
conda activate spoof_SUPERB

# 2. point the repo at your data (this is the only settings file)
$EDITOR configs/paths.yaml
python -m spoof_superb.config          # check what it resolved to

# 3. reproduce the paper's tables from the published score files
bin/reproduce_main_results.sh
```

If step 3 passes, your installation and your score files are good. It reads
score files only and takes about 2m40s.

Full instructions: [Installation](docs/01-installation.md) →
[Reproducing the published results](docs/02-reproducing-results.md).

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
| 8 | [Verification](docs/08-verification.md) | you rebuilt the tree or re-ran the analyses and want to check them against the reference |
| 9 | [Analysis](docs/09-analysis.md) | you want to regenerate a table or figure |
| 10 | [Tests](docs/10-testing.md) | you changed the code |
| 11 | [Troubleshooting](docs/11-troubleshooting.md) | something broke |

Also: `humanpending.md` (open items needing a decision) and `REORG_PLAN.md`
(the audit the current layout came from).

---

## Layout

```
configs/paths.yaml     your settings — the only file you need to edit
bin/                   editable shell scripts, one per task
spoof_superb/
├── config.py          settings schema and loader (code, not settings)
├── core/              metrics.py (EER, t-DCF), scorefile.py (the 4-col format)
├── models/            aasist, aasist_raw, linear_head, sls, lfcc_gmm
├── frontends/         lfcc, rawboost
├── data/              dataset classes, prep/ (MLAAD and M-AILABS preparation)
├── scoring/           ONE scoring entry point + dataset registry + back-ends
├── orchestration/     ONE job runner (GPU pool, resume, retry) + job specs
├── verification/      the separate two-level check: scores, then analysis
├── analysis/          EER/FAR tables, figures, protocol construction
├── train/             LFCC-GMM trainer
└── tools/             checkpoint selection helpers
tests/                 contract tests + the main-results numerical baseline
main.py                training entry point
```

## The shell scripts

Each script in `bin/` opens with a settings block you edit, rather than
requiring a long command line:

```bash
$EDITOR bin/score.sh     # set MODEL, SSL_MODEL, DATASET, OUTPUT_FILE, ...
bin/score.sh
```

They print the command they run, and pass any extra arguments through, so
`bin/score.sh --limit 300` still works for a one-off.

| Script | Does |
|---|---|
| `bin/reproduce_main_results.sh` | regenerate the paper's tables and check them |
| `bin/score.sh` | score one model on one evaluation set |
| `bin/orchestrate.sh` | score every model for a job, across GPUs |
| `bin/verify.sh` | check a finished run against the published reference, both levels |
| `bin/train.sh` | train an SSL or torch baseline model |
| `bin/train_lfcc_gmm.sh` | train the LFCC-GMM baseline (CPU) |

## Command migration

If you used the flat scripts, they merged into the entry points above.

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
| `python verify_mlaad.py --new N --ref R` | `bin/verify.sh` |
| `python verify_spoofceleb.py --new N --ref R` | `bin/verify.sh` |
| `python verify_asvld.py` | `python -m spoof_superb.verification.asvld_report` |
| `python verify_noise_rerun.py` | `python -m spoof_superb.verification.noise_rerun_gate` |
| `python scripts/<name>.py` | `python -m spoof_superb.analysis.<name>` |
| `from evaluation import compute_eer` | `from spoof_superb.core.metrics import compute_eer` |
| `.asvld_skip` containing `Filtering` | `--skip_conditions Filtering` (the default) |

## Citation

If you use this benchmark, please cite the accompanying paper.

## License

See [LICENSE](LICENSE).
