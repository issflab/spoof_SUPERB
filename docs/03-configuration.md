# 3. Configuration

## There is one settings file

```
configs/paths.yaml       <-- edit this. This is your settings file.
spoof_superb/config.py   <-- code: the schema, the defaults, the loader.
```

`configs/paths.yaml` is loaded **automatically**. You do not need to set any
environment variable to make it work.

`spoof_superb/config.py` is not a second settings file. It declares what the
settings are called, what type they have, and what they fall back to if your
YAML omits them -- so a fresh clone still imports before you have configured
anything. You should not need to edit it.

Check what is actually in effect, and where it came from:

```bash
python -m spoof_superb.config
```

```
config file : /home/alhashim/ASD_SUPERB/spoof_SUPERB/configs/paths.yaml

  data_root                = /data/Data
  scores_root              = /data/ssl_anti_spoofing/asd_superb_score_files
  ...
```

Values overridden by an environment variable are marked `[env]`.

## Resolution order

Lowest priority first. Each layer overrides the one above it.

| | Layer | Use it for |
|---|---|---|
| 1 | dataclass defaults in `config.py` | fallbacks, so a fresh clone imports |
| 2 | **`configs/paths.yaml`** | **your normal setup -- edit this** |
| 3 | environment variables | a one-off override for a single command |
| 4 | CLI flags of the tool you run | a one-off override for a single run |

```bash
# permanent: edit configs/paths.yaml

# just this once:
SPOOF_SUPERB_SCORES_ROOT=/tmp/experiment bin/reproduce_table5.sh

# just this run:
bin/score.sh --cuda_device cuda:1
```

## Settings

| Key | What it points at | Default |
|---|---|---|
| `data_root` | root containing every audio corpus | `/data/Data` |
| `scores_root` | score files, read and written | `/data/ssl_anti_spoofing/asd_superb_score_files` |
| `models_root` | one directory per trained SSL linear head | `.../asd_superb_models/linear_head_models` |
| `linear_head_prefix` | checkpoint directory name prefix | `model_weighted_CCE_50_64_linear_head_ASV19_` |
| `baseline_models_root` | `aasist_raw` and `lfcc_gmm` checkpoints | `.../asd_superb_models/baselines` |
| `save_dir` | where new training runs write | `/data/ssl_anti_spoofing/asd_superb/` |
| `database_path` | training audio | ASVspoof2019 LA train |
| `protocols_path` | training protocols | ASVspoof2019 LA cm protocols |
| `train_protocol`, `dev_protocol` | protocol filenames | ASVspoof2019 train / dev |
| `reference_ssl` | the SSL model whose score file defines a trial list | `xls_r_300m` |
| `score_layout` | score-file directory layout: `v2` or `legacy` | `legacy` |
| `cuda_device` | default device | `cuda:0` |
| `python` | interpreter the orchestrator launches subprocesses with | the running one |

Derived values are computed, not configured: `reference_dir` is always
`{scores_root}/linear_head`, and `model_save_path` is
`{save_dir}/{model_name}`. Move the root and everything built on it follows.

## Score-file layout

`score_layout` decides where score files are written and read.

```
v2      {scores_root}/raw/{system}/{dataset}/{frontend}.txt
legacy  the pre-reorganisation paths
```

`v2` example:

```
raw/linear_head/mlaad_v10/xls_r_300m.txt
raw/linear_head/asvspoof_ld/tera.txt
raw/lfcc_gmm/in_the_wild/none.txt
```

Four properties it is built for:

* **Nothing is parsed.** `linear_head_Noise_Addition_wavlm_large_babble_10.txt`
  cannot be split by any regex, because model names contain underscores.
  Directory levels have no such problem.
* **One glob per question.** `raw/*/*/xls_r_300m.txt` is every score file for
  one upstream; `raw/linear_head/mlaad_v10/*.txt` is every model on one set.
* **Versions are in the name.** `mlaad_v10` and `mlaad_legacy` are different
  datasets rather than two directories you must know to tell apart.
* **No condition level.** ASVLD conditions are rows inside one file. The five
  condition protocols are mutually disjoint -- 2,065,873 rows, 2,065,873
  distinct utt_ids, zero collisions -- so the condition is already carried by
  the utt_id and a directory for it would add nothing. Pooling also forces a
  file to have one provenance instead of hiding a mixture.

The default is `legacy`, so an existing tree keeps working untouched. Set `v2`
in `configs/paths.yaml` for a new one. Every path comes from
`spoof_superb.core.scorepath.score_path`, so switching is a config change
rather than a refactor, and both layouts stay readable during a migration.

Dataset directory names are canonical (lowercase, version-bearing) and mapped
in `scorepath.DATASET_DIRS`; the registry keys stay as the CLI vocabulary, so
`--dataset wild` still selects `in_the_wild`.

## Environment variables

Every setting has one, for one-off overrides:

```
SPOOF_SUPERB_DATA_ROOT      SPOOF_SUPERB_SCORES_ROOT
SPOOF_SUPERB_MODELS_ROOT    SPOOF_SUPERB_BASELINE_MODELS_ROOT
SPOOF_SUPERB_PYTHON
SSL_DATABASE_PATH  SSL_PROTOCOLS_PATH  SSL_SAVE_DIR  SSL_MODEL_ARCH
SSL_MODE  SSL_MODEL_NAME  SSL_DATASET  SSL_PRETRAINED_CHECKPOINT
CUDA_DEVICE
```

One is different: **`SPOOF_SUPERB_CONFIG` holds no settings.** It points at a
*different* YAML file, for when you keep several -- one per machine, say:

```bash
SPOOF_SUPERB_CONFIG=configs/paths.gpu-server.yaml bin/orchestrate.sh
```

## Partial files are fine

Anything you leave out falls back to the default. A minimal config is legal:

```yaml
# configs/paths.yaml
data_root: /mnt/corpora
scores_root: /mnt/results/scores
```

A misspelled key is reported on load rather than silently ignored:

```
[config] ignoring unknown key 'scores_rooot' in configs/paths.yaml
```

## The shell scripts read the same file

`bin/_common.sh` queries the resolved config and exports `DATA_ROOT`,
`SCORES_ROOT`, `MODELS_ROOT` and friends into the calling script. The shell and
the Python code therefore cannot disagree about where your data is -- there is
one source of truth, and it is `configs/paths.yaml`.

## Should configs/paths.yaml be committed?

The file is tracked, holding the paths for the original host. If you are
working on a different machine, either edit it and keep the change local, or
leave it alone and keep your own file elsewhere:

```bash
cp configs/paths.yaml configs/paths.mine.yaml
export SPOOF_SUPERB_CONFIG=$PWD/configs/paths.mine.yaml
```

`configs/AASIST.conf` is unrelated -- it is the AASIST architecture
hyper-parameter file, read by the model, not by the path config.
