# ASD-SUPERB

* ASD-SUPERB includes a widely recognized Audio Spoof Detection downstream task that focuses on identifying whether an input audio signal is genuine (bona fide) or spoofed (fake). Spoofed audio refers to speech that has been artificially manipulated or generated using techniques such as text-to-speech (TTS) or voice conversion (VC).

* ASD-SUPERB serves as an extension of [S3PRL](https://github.com/s3prl/s3prl), a toolkit for SUPERB.

## Installation

Clone this repository to your workspace using the following command:
```ruby
git clone https://github.com/issflab/spoof_SUPERB.git
```

Create the conda environment from the provided YAML file:
```ruby
conda env create -f environment.yaml
```

Activate the environment:
```ruby
conda activate spoof_SUPERB
```

## Data preparation

1. Download the LA partition from [ASVSpoof 2019](https://datashare.ed.ac.uk/handle/10283/3336) dataset, and unzip it.
```ruby
mkdir -p ASVSpoofData_2019

cd ASVSpoofData_2019

wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip

unzip LA.zip
```

2. Check the ASVSpoofData_2019 structure. You should have the following folder and files in LA directory at minimum. Read the Readme.txt to understand the folder and file structure.

```
ASVSpoofData_2019/
├── LA/
    ├── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019_LA_dev
    ├── ASVspoof2019_LA_eval
    ├── ASVspoof2019_LA_eval
    ├── README.LA.txt
```

## Documentation

* **[docs/RUNNING.md](docs/RUNNING.md)** — how to run every component: training,
  scoring, orchestration, verification, analysis, data prep, tests, plus
  end-to-end recipes and troubleshooting.
* `REORG_PLAN.md` — the audit the current layout came from.
* `humanpending.md` — open items needing a human decision.

## Layout

```
spoof_superb/
├── config.py          all corpus / model / score roots; env + YAML overridable
├── core/              metrics.py (EER, t-DCF), scorefile.py (the 4-col format)
├── models/            aasist, aasist_raw, linear_head, sls, lfcc_gmm
├── frontends/         lfcc, rawboost
├── data/              datasets_ssl.py, prep/ (M-AILABS and MLAAD preparation)
├── scoring/           ONE scoring entry point + dataset registry + back-ends
├── orchestration/     ONE job runner (GPU pool, resume, retry) + job specs
├── verification/      shared statistics + per-dataset grade policies
├── analysis/          EER/FAR tables, figures, protocol construction
└── tools/             checkpoint selection helpers
bin/                   shell wrappers over the entry points
tests/                 contract tests + the Table 5 numerical baseline
main.py                training entry point
```

## Configuration

`spoof_superb/config.py` is the single source of truth for paths. Resolution
order, lowest priority first: dataclass defaults, then a YAML file pointed at by
`SPOOF_SUPERB_CONFIG`, then environment variables, then the CLI of whatever tool
you run. Importing it has no side effects.

```bash
export SPOOF_SUPERB_SCORES_ROOT=/somewhere/else/scores
export SPOOF_SUPERB_CONFIG=my_paths.yaml
```

## Training

By default a linear-head model is trained on ASVspoof 2019 LA. Use `-h` for the
full `--ssl_model` list.

```bash
bin/train.sh --batch_size 64 --num_epochs 50 --ssl_model wavlm_large
```

Architecture, mode and the training-set tag now have CLI flags
(`--model_arch`, `--mode`, `--train_dataset`); they were previously settable
only through the `SSL_MODEL_ARCH` / `SSL_MODE` / `SSL_DATASET` environment
variables, which still work.

## Scoring

One entry point covers every model and every benchmark set.

```bash
bin/score.sh --list_datasets

# a published benchmark column (trial list comes from the reference score file)
bin/score.sh --model linear_head --ssl_model xls_r_300m \
    --model_path .../swa.pth --dataset wild --output_file out.txt

# one ASVLD laundering condition
bin/score.sh --model linear_head --ssl_model xls_r_300m --model_path .../swa.pth \
    --source asvld --asvld_condition Noise_Addition --output_file out.txt

# MLAAD / M-AILABS (walk the corpus) and SpoofCeleb (protocol CSV)
bin/score.sh ... --source walk --walk_root /data/Data/MLAAD/fake --label spoof
bin/score.sh ... --source protocol_csv
```

Scoring runs **fp32 by default**. `--amp` enables fp16 autocast and is opt-in:
fp16 overflow is what put 384,157 NaN per model into the masked-spectrogram
front-ends. A CUDA device that is requested but unavailable is a hard error, not
a silent CPU fallback.

## Orchestration, verification, reproduction

```bash
bin/orchestrate.sh --job spoofceleb            # jobs: mlaad, mailabs, spoofceleb, baselines
bin/orchestrate.sh --job baselines --jobs 1    # sequential
bin/orchestrate.sh --job mlaad --list          # enumerate tasks, run nothing

bin/verify.sh --check spoofceleb --new out.txt --ref reference.txt

bin/reproduce_table5.sh                        # recompute Tables 5/6 and diff vs baseline
```

## Tests

```bash
pytest tests/ -q                # fast contract tests
RUN_TABLE5=1 pytest tests/ -q   # + the ~2m40s numerical regression gate
```

## Command migration

The flat scripts were merged into the entry points above. Old invocation on the
left, its replacement on the right.

| Before | Now |
|---|---|
| `python eval_baselines.py --list_datasets` | `bin/score.sh --list_datasets` |
| `python eval_baselines.py --model M --dataset D ...` | `bin/score.sh --model M --dataset D ...` |
| `python eval_asvld.py --condition C ...` | `bin/score.sh --model linear_head --source asvld --asvld_condition C ...` |
| `python eval_mlaad.py ...` | `bin/score.sh --model linear_head --source walk ...` |
| `python eval_mlaad.py --protocol_csv ...` | `bin/score.sh --model linear_head --source protocol_csv ...` |
| `python orchestrate_mlaad.py` | `bin/orchestrate.sh --job mlaad` |
| `python orchestrate_mailabs.py` | `bin/orchestrate.sh --job mailabs` |
| `python orchestrate_spoofceleb.py` | `bin/orchestrate.sh --job spoofceleb` |
| `python orchestrate_baselines.py` | `bin/orchestrate.sh --job baselines --jobs 1` |
| `python verify_mlaad.py --new N --ref R` | `bin/verify.sh --check mlaad --new N --ref R` |
| `python verify_spoofceleb.py --new N --ref R` | `bin/verify.sh --check spoofceleb --new N --ref R` |
| `python verify_asvld.py` | `python -m spoof_superb.verification.asvld_report` |
| `python verify_noise_rerun.py` | `python -m spoof_superb.verification.noise_rerun_gate` |
| `python scripts/<name>.py` | `python -m spoof_superb.analysis.<name>` |
| `from evaluation import compute_eer` | `from spoof_superb.core.metrics import compute_eer` |
| `.asvld_skip` file containing `Filtering` | `--skip_conditions Filtering` (the default) |
