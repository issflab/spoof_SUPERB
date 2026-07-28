# 7. Training

Two kinds of model: the SSL linear heads (and the other torch architectures),
and the two non-SSL baselines.

Corpus and output paths come from `configs/paths.yaml`
(`database_path`, `protocols_path`, `save_dir`). By default that is
ASVspoof2019 LA train.

## SSL models and torch baselines

Edit the settings block at the top of `bin/train.sh`, then:

```bash
bin/train.sh
```

or directly:

```bash
python main.py --model_arch linear_head --ssl_model wavlm_large \
    --batch_size 64 --num_epochs 50
```

| Flag | Meaning |
|---|---|
| `--model_arch` | `aasist`, `sls`, `linear_head`, `aasist_raw`, `lfcc_gmm` |
| `--ssl_model` | s3prl upstream name; ignored by `aasist_raw` |
| `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, `--loss` | optimisation |
| `--micro_batch` | gradient accumulation; `0` disables |
| `--algo` | RawBoost augmentation variant; `0` = none |
| `--train_dataset` | tag recorded in the checkpoint name |
| `--comment` | free-text suffix on the checkpoint directory |
| `--seed` | default 1234 |

`--model_arch`, `--mode` and `--train_dataset` previously had no CLI flag and
could only be set through `SSL_MODEL_ARCH` / `SSL_MODE` / `SSL_DATASET`. The
environment variables still work; the flags override them.

### Where checkpoints go

```
{save_dir}/model_{loss}_{epochs}_{batch}_{arch}_{dataset}_{ssl}/
```

e.g. `model_weighted_CCE_50_64_linear_head_ASV19_wavlm_large/`. That naming is
what `linear_head_prefix` in `configs/paths.yaml` matches when the orchestrator
discovers trained models, so if you change the training hyper-parameters you
must update the prefix for the sweep to find your checkpoints.

The non-SSL baselines record `none` in the SSL slot rather than the unused
`--ssl_model` default, so their checkpoint paths do not claim an upstream they
never had.

### Gradient accumulation

`--micro_batch 16` with `--batch_size 64` runs four micro-batches per optimizer
step. This is behaviour-preserving: `tests/test_grad_accum.py` asserts that
micro-batch 16 × accumulation 4 produces the same weights as a single
un-accumulated pass at batch 64, and includes a guard proving the test is not
vacuous. Use it when a model does not fit at the recipe's batch size, so the
run is still comparable to the others.

### Logs

TensorBoard events go to `outputs/logs/{model_tag}/`.

```bash
tensorboard --logdir outputs/logs
```

## LFCC-GMM

This one does not train through `main.py`'s loop -- it is EM over two diagonal
GMMs, with no gradients, no DataLoader and no GPU.

```bash
bin/train_lfcc_gmm.sh
```

or:

```bash
python -m spoof_superb.train.lfcc_gmm --n_jobs 16 --ncomp 512
```

It writes `{baseline_models_root}/lfcc_gmm/{bonafide,spoof}/gmm_final.pkl`,
which is the **directory** you later pass to `--model_path` when scoring.

The LFCC front-end is a vendored port of a reference spafe-backed
implementation. `tests/test_lfcc_frontend.py` asserts it reproduces that
reference bit-for-bit on real ASVspoof2019 audio; if that test fails, every
LFCC-GMM score is computed on different features than the reference system and
is not comparable.

## AASIST checkpoint selection

AASIST training does not always end on its best epoch. To pick the checkpoint
to score with:

```bash
python -m spoof_superb.tools.select_aasist_ckpt --run_dir {baseline_models_root}/...
```

## After training

Score the new checkpoint: [scoring](05-scoring.md). For a whole sweep,
[orchestration](06-orchestration.md).
