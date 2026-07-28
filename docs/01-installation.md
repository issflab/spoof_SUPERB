# 1. Installation

## Requirements

* Linux, Python 3.10
* NVIDIA GPU for training and for scoring the SSL models. Reproducing the
  published tables and running the LFCC-GMM baseline need no GPU.
* ~50 GB for the score files if you want to reproduce the tables; the corpora
  themselves are considerably larger.

## Install

```bash
git clone https://github.com/issflab/spoof_SUPERB.git
cd spoof_SUPERB

conda env create -f environment.yaml
conda activate spoof_SUPERB
```

`environment.yaml` creates a Python 3.10 environment and installs
`requirements.txt` into it. To install into an existing environment instead:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**The `--extra-index-url` is required.** `requirements.txt` pins
`torch==2.7.1` built against CUDA 12.6. Without the index URL pip resolves a
different build, and with `cu130` wheels (torch >= 2.9)
`torch.cuda.is_available()` returns `False` on this host with *"The NVIDIA
driver on your system is too old"*. The pins in `requirements.txt` are the
verified-working set; the file's header explains each one.

## Verify the installation

```bash
python -c "import torch, s3prl, librosa; print(torch.__version__, torch.cuda.is_available())"
python -m spoof_superb.config      # prints the resolved settings
pytest tests/ -q                   # ~16 s, no GPU needed
```

Expected: `2.7.1+cu126 True` (or `False` if you have no GPU, which is fine for
reproduction), a settings dump, and a green test run.

If `pytest` reports `ModuleNotFoundError: No module named 'spoof_superb'`, you
are not in the repo root. See [troubleshooting](11-troubleshooting.md).

## A note on environments and reproducibility

`requirements.txt` does not currently pin `soxr`, which is the resampler
librosa uses. Two environments on the original host differed on it (1.0.0 vs
0.5.0.post1), and score files for datasets that get resampled to 16 kHz are
sensitive to that difference. If you are producing results that must be
comparable to ours, record your `soxr` version:

```bash
python -c "import soxr, librosa; print('soxr', soxr.__version__, 'librosa', librosa.__version__)"
```

This is tracked as item RP-1 in `humanpending.md`.

## Next

Go to [reproducing the published results](02-reproducing-results.md).
