# 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'spoof_superb'` | You are not in the repo root, or you ran an analysis script by file path. Use `bin/*.sh`, or `export PYTHONPATH=/path/to/spoof_SUPERB:$PYTHONPATH`, and invoke analysis modules with `-m`. |
| `torch.cuda.is_available()` is `False` on a GPU host | Wrong torch build. `requirements.txt` pins `torch==2.7.1`; install it with `--extra-index-url https://download.pytorch.org/whl/cu126`. cu130 wheels (torch ≥ 2.9) report *"NVIDIA driver too old"* on this host. |
| Scoring exits immediately with **rc=2** | CUDA requested but unavailable. Deliberate -- it refuses to fall back to CPU, because a CPU MLAAD run is ~25 h vs ~20 min. Fix the driver, or pass `--cuda_device cpu` if you really mean it. |
| Orchestrator says "existing output is complete" and skips | Resume. Set `FORCE="yes"` in `bin/orchestrate.sh`, or pass `--force`. |
| NaN in the scores | Almost certainly `--amp`. Re-run in fp32, which is the default. See the fp16 overflow note in [scoring](05-scoring.md). |
| An ASVLD condition produced no output | `Filtering` is in the default skip list. Pass `--skip_conditions` with no values to score it. |
| `numpy.genfromtxt` chokes on a score file | utt_ids contain spaces. Use the `.tsv` twin written beside it. |
| `[config] ignoring unknown key 'x'` | Typo in `configs/paths.yaml`. Run `python -m spoof_superb.config` to see what actually loaded. |
| Config changes have no effect | An environment variable is overriding your YAML. `python -m spoof_superb.config` marks those `[env]`. |
| A figure landed in an unexpected directory | `create_taxonomy` and `create_SSL_taxonomy` write CWD-relative. Run from the repo root. |
| Main-results gate fails after a refactor | Read the printed diff; it names the model, dataset and field. Usually a moved path or a changed parser. |
| `bin/*.sh` says `interpreter 'python' not found` | Activate the conda environment, or set `SPOOF_SUPERB_PYTHON=/path/to/python`. |
| Trial counts differ from the paper | You re-derived the trial list from a raw protocol instead of the published score file. Several published sets are subsets whose selection rule is not recorded. Use `--source benchmark`. |
| Famous Figures scores zero bonafide trials | The `/-/` → `Bonafide` path remap did not apply. Check the corpus layout in [datasets](04-datasets.md). |
| ASVspoof2021 scoring is far slower than other corpora, and the log fills with `PySoundFile failed. Trying audioread instead.` | `av` is not installed. libsndfile cannot decode 36-43% of the ASVspoof2021 FLAC files and librosa falls back to a subprocess per file. `pip install av==17.1.0`. See below. |

## ASVspoof2021 decodes ~40x slower without `av`

libsndfile 1.2.2 refuses a large minority of the ASVspoof2021 FLAC files:

| corpus | files libsndfile cannot read |
|---|---|
| `asvspoof2021_LA` | 43% |
| `asvspoof2021_DF` | 36% |
| `deepfake_eval_2024` | 9% |
| every other corpus | 0% |

The files are fine -- their headers are indistinguishable from the ones that
read, ffmpeg decodes them, and remuxing does not help. It is a decoder bug.

`librosa.load` masks it by falling back to audioread, which spawns a
subprocess per file:

```
soundfile, in process        0.8 ms/file
av (libav), in process       3.6 ms/file
librosa -> audioread        67   ms/file
```

Measured over a mixed 500-file ASVspoof2021-DF sample: **1.98 ms/file with
`av`, 28.65 ms/file without.**

FLAC is lossless, so all three decoders return bit-identical samples --
asserted in `tests/test_audio.py`, not assumed. Installing or removing `av`
therefore cannot move a single score; it only changes how long they take.
Score files produced before and after the change are directly comparable.

## Known open issues

Tracked in `humanpending.md`:

* **RP-1** — Two environments on the original host disagreed on `soxr`
  (1.0.0 vs 0.5.0.post1), librosa's resampler. Existing score files for
  datasets that get resampled to 16 kHz were produced by both. New runs use one
  interpreter (`cfg.python`) and are self-consistent, but this cannot be fixed
  retroactively by a refactor. Record your `soxr` version with any new results.
* **RP-2** — `create_combined_mlaad_meta.py` (English-only) still has the
  quoting bug that its `_all` sibling fixes: `ja/kokoro` silently loses 53 of
  1000 rows. Its output feeds two downstream scripts.
* **RP-3** — Confirm `Filtering` should remain in the default ASVLD skip list.
* **RP-4** — `compute_far_matrix` and `compute_eer_tts` duplicate their
  aggregation logic; FAR has no home in `core/metrics.py`.
* **RP-5** — The score directory holds ~19 GB of duplicated or regenerable
  views. Reorganisation deferred.

## Getting more detail

Every entry point supports `--help`:

```bash
python -m spoof_superb.scoring.driver --help
python -m spoof_superb.orchestration.driver --help
python -m spoof_superb.verification --help
python -m spoof_superb.config
```

The module docstrings carry the reasoning behind the non-obvious behaviour --
`spoof_superb/scoring/driver.py` and `spoof_superb/verification/verdicts.py` in
particular are worth reading before changing anything there.
