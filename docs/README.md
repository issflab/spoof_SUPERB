# Spoof-SUPERB documentation

Read these in order if you are new. Each one stands alone if you are not.

| # | Document | Read it when |
|---|---|---|
| 1 | [Installation](01-installation.md) | first, always |
| 2 | [Reproducing the published results](02-reproducing-results.md) | you want to confirm the setup works and regenerate the paper's tables |
| 3 | [Configuration](03-configuration.md) | your data is not where ours is |
| 4 | [Datasets and protocols](04-datasets.md) | before scoring from checkpoints -- what each corpus must look like on disk |
| 5 | [Scoring](05-scoring.md) | you have a checkpoint and want score files |
| 6 | [Orchestration](06-orchestration.md) | you want to score many models unattended |
| 7 | [Training](07-training.md) | you want to train a new SSL or baseline model |
| 8 | [Verification](08-verification.md) | you want to check new score files against the published ones |
| 9 | [Analysis: tables and figures](09-analysis.md) | you want to regenerate a table or figure |
| 10 | [Tests](10-testing.md) | you changed the code |
| 11 | [Troubleshooting](11-troubleshooting.md) | something broke |

## The 60-second version

```bash
conda env create -f environment.yaml && conda activate spoof_SUPERB
$EDITOR configs/paths.yaml        # point it at your data
bin/reproduce_main_results.sh           # regenerate the paper's tables and check them
```

If that last command passes, your installation and your score files are good.
It needs no GPU, no checkpoints and no audio -- only the published score files.

## How the pieces fit together

```
   train                score                orchestrate            verify
  ────────           ───────────            ─────────────         ──────────
  audio +            checkpoint +           many scorings         new scores
  protocol    ──▶    trial list      ──▶    across GPUs     ──▶   vs published
     │                    │                                            │
     ▼                    ▼                                            ▼
  checkpoint          score file                                    PASS/FAIL
                          │
                          ▼
                      analysis  ──▶  EER/FAR tables  ──▶  figures, paper tables
```

Reproducing the paper starts at the bottom of that diagram, not the top: the
score files are published, so you can regenerate every table without running a
single model.
