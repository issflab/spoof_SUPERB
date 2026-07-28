# Running Spoof-SUPERB

How to run each component. For the layout and the old-command → new-command
table, see the [README](../README.md).

Every component is a module under `spoof_superb/`, so all of them are invoked
the same way:

```bash
python -m spoof_superb.<component>.<entry> [flags]
```

The `bin/*.sh` wrappers are thin shells over exactly these commands; they set
`PYTHONPATH` so the invocation works from any directory. Use whichever you
prefer — they are interchangeable.

---

## 0. Setup

```bash
conda activate spoof_SUPERB
cd /home/alhashim/ASD_SUPERB/spoof_SUPERB
```

Running `python -m spoof_superb...` requires the repo root on the import path.
From the repo root that is automatic. From anywhere else, either use the
`bin/` wrappers or export it yourself:

```bash
export PYTHONPATH=/home/alhashim/ASD_SUPERB/spoof_SUPERB:$PYTHONPATH
```

**The analysis scripts must be run as modules, not by file path.**
`python spoof_superb/analysis/create_heatmap.py` will fail with
`ModuleNotFoundError`; `python -m spoof_superb.analysis.create_heatmap` works.

---

## 1. Configuration

`spoof_superb/config.py` holds every corpus, model and score root. Resolution
order, lowest priority first:

1. the dataclass defaults in `config.py`
2. a YAML file, if `SPOOF_SUPERB_CONFIG` points at one
3. environment variables
4. the CLI flags of whatever tool you run

Importing config has no side effects — it creates nothing.

| Setting | Env var | Default |
|---|---|---|
| corpus root | `SPOOF_SUPERB_DATA_ROOT` | `/data/Data` |
| score files | `SPOOF_SUPERB_SCORES_ROOT` | `/data/ssl_anti_spoofing/asd_superb_score_files` |
| linear-head checkpoints | `SPOOF_SUPERB_MODELS_ROOT` | `/data/ssl_anti_spoofing/asd_superb_models/linear_head_models` |
| baseline checkpoints | `SPOOF_SUPERB_BASELINE_MODELS_ROOT` | `/data/ssl_anti_spoofing/asd_superb_models/baselines` |
| training output | `SSL_SAVE_DIR` | `/data/ssl_anti_spoofing/asd_superb/` |
| training corpus | `SSL_DATABASE_PATH` | ASVspoof2019 LA train |
| protocols | `SSL_PROTOCOLS_PATH` | ASVspoof2019 LA cm protocols |
| architecture | `SSL_MODEL_ARCH` | `aasist` |
| device | `CUDA_DEVICE` | `cuda:0` |
| subprocess interpreter | `SPOOF_SUPERB_PYTHON` | the running interpreter |

A YAML config takes any of the field names:

```yaml
# my_paths.yaml
scores_root: /data/scratch/my_scores
data_root: /data/Data
reference_ssl: wavlm_large
```

```bash
SPOOF_SUPERB_CONFIG=my_paths.yaml python -m spoof_superb.scoring.driver --list_datasets
```

Check what is actually in effect:

```bash
python -c "from spoof_superb.config import cfg; print(cfg)"
```

---

## 2. Training — `main.py`

```bash
bin/train.sh --model_arch linear_head --ssl_model wavlm_large \
    --batch_size 64 --num_epochs 50
```

Key flags:

| Flag | Meaning |
|---|---|
| `--model_arch` | `aasist`, `sls`, `linear_head`, `aasist_raw`, `lfcc_gmm` |
| `--ssl_model` | s3prl upstream (ignored by `aasist_raw` / `lfcc_gmm`) |
| `--train_dataset` | tag recorded in the checkpoint name, e.g. `ASV19` |
| `--batch_size`, `--num_epochs`, `--lr`, `--loss` | optimisation |
| `--micro_batch` | gradient accumulation; `0` disables it |
| `--algo` | RawBoost augmentation variant |
| `--comment` | free-text suffix on the checkpoint directory |

Checkpoints land in `{save_dir}/model_{loss}_{epochs}_{batch}_{arch}_{dataset}_{ssl}/`.

`--model_arch`, `--mode` and `--train_dataset` used to be settable only through
`SSL_MODEL_ARCH` / `SSL_MODE` / `SSL_DATASET`. The env vars still work; the
flags override them.

**LFCC-GMM trains separately** — it is EM over two diagonal GMMs, with no
gradients and no GPU:

```bash
python -m spoof_superb.train.lfcc_gmm --n_jobs 16
```

---

## 3. Scoring — `spoof_superb.scoring.driver`

One entry point for every model and every benchmark set. It always produces the
canonical 4-column file `{utt_id} - {key} {score}`.

```bash
bin/score.sh --list_datasets       # the 10 published columns and their trial counts
```

Pick a **back-end** with `--model` and a **trial source** with `--source`.

### `--source benchmark` (default) — a published column

The trial list and labels come from the reference score file, which is how the
baselines stay comparable to the paper's numbers.

```bash
bin/score.sh --model linear_head --ssl_model xls_r_300m \
    --model_path $MODELS/model_weighted_CCE_50_64_linear_head_ASV19_xls_r_300m/swa.pth \
    --dataset wild --output_file out/wild_xls_r_300m.txt
```

Datasets: `eval_2019`, `asvspoof2021_LA`, `asvspoof2021_DF`, `asvspoof5`,
`deepfake_eval_2024`, `wild`, `Famous_Figures`, `spoofceleb`, `Multilingual`,
`asvspoofLD`.

### `--source asvld` — one laundering condition

```bash
bin/score.sh --model linear_head --ssl_model xls_r_300m --model_path .../swa.pth \
    --source asvld --asvld_condition Noise_Addition \
    --output_file out/noise_xls_r_300m.txt
```

Conditions: `Noise_Addition`, `Reverberation`, `Resampling`, `Recompression`,
`Filtering`. **`Filtering` is skipped by default** (`--skip_conditions`), which
preserves the behaviour of the old `.asvld_skip` sentinel file. Pass
`--skip_conditions` with no values to score it.

### `--source walk` — MLAAD and M-AILABS

Enumerates every wav under a root; one label for all of them.

```bash
# MLAAD fake (defaults: walk_root=$DATA/MLAAD/fake, label=spoof)
bin/score.sh --model linear_head --ssl_model xls_r_300m --model_path .../swa.pth \
    --source walk --output_file out/mlaad_xls_r_300m.txt

# M-AILABS bonafide
bin/score.sh ... --source walk --walk_root /data/Data/MAILabs --label bonafide \
    --output_file out/mailabs_xls_r_300m.txt
```

### `--source protocol_csv` — SpoofCeleb

Per-utterance labels from the protocol (`attack == a00` is bonafide).

```bash
bin/score.sh --model linear_head --ssl_model xls_r_300m --model_path .../swa.pth \
    --source protocol_csv --output_file out/spoofceleb_xls_r_300m.txt
```

### Back-ends

| `--model` | Checkpoint | Notes |
|---|---|---|
| `linear_head` | `swa.pth` | needs `--ssl_model`; batch default 32 |
| `aasist_raw` | `swa.pth` | no SSL upstream; batch default 64 |
| `lfcc_gmm` | GMM **directory** | CPU only; use `--n_jobs`, not `--batch_size` |

### Flags that matter

- `--amp` — fp16 autocast. **Off by default and should stay off.** fp16
  overflow is what wrote 384,157 NaN per model into the masked-spectrogram
  front-ends (tera, mockingjay, mockingjay_960hr, audio_albert_960hr).
- `--cuda_device` — if you request a CUDA device and CUDA is unavailable, the
  run **fails with rc=2** rather than silently falling back to CPU. A CPU run of
  MLAAD is ~25 h against ~20 min on an A100.
- `--restrict_to REF [--restrict_prefix P]` — score only the utt_ids present in
  an existing score file, in that file's order. Use this to reproduce or verify
  against a published subset.
- `--limit N` — score the first N trials only. Use for smoke tests.

A `.tsv` twin is written automatically when any utt_id contains a space (MLAAD
v10 has 39,000 such rows) because `numpy.genfromtxt` cannot parse those.

---

## 4. Orchestration — `spoof_superb.orchestration.driver`

Runs a whole job — every model on one set — across GPUs, with resume, retry and
a status file.

```bash
bin/orchestrate.sh --job spoofceleb            # all linear heads, pooled over GPUs
bin/orchestrate.sh --job mlaad --only xls_r_300m
bin/orchestrate.sh --job baselines --jobs 1    # sequential
bin/orchestrate.sh --job mlaad --list          # enumerate tasks, run nothing
```

| Job | What it scores | Verify policy |
|---|---|---|
| `mlaad` | every linear head on MLAAD v10 fake | `mlaad` |
| `mailabs` | every linear head on M-AILABS bonafide (staging dir) | `mlaad` |
| `spoofceleb` | every linear head on SpoofCeleb eval | `spoofceleb` |
| `baselines` | `aasist_raw` and `lfcc_gmm` on all 10 sets | none |

Behaviour worth knowing:

- **Resume is automatic.** A complete, NaN-free output is not recomputed; it is
  only re-verified. `--force` overrides.
- **GPUs are pinned by UUID**, not index — index-based `CUDA_VISIBLE_DEVICES`
  fails to initialise on this host once another process holds a device.
- **rc=2 is retried** in a fresh process. That return code is the scoring
  driver's "CUDA unavailable" guard, i.e. an environment fault rather than a
  model fault. Budgets differ per job (`spoofceleb` retries 6× over 3 h,
  `baselines` 3× over 1 h).
- `mlaad` and `mailabs` skip `byol_a_2048` and `mockingjay`; `spoofceleb`
  deliberately does not.
- Progress lands in `{out_dir}/run_status.json`, a summary table in
  `{out_dir}/SUMMARY.txt`, per-task logs in `{out_dir}/logs/`.

`--jobs 1` reproduces the old sequential `orchestrate_baselines.py`.

---

## 5. Verification

Three different kinds of check. They are separate on purpose.

### Score file vs reference — `spoof_superb.verification.driver`

```bash
bin/verify.sh --check spoofceleb --new out.txt --ref reference.txt
```

Exit 0 = pass, 1 = fail. The policies are **not** interchangeable:

| `--check` | Verdict | NaN tolerance in your output |
|---|---|---|
| `mlaad`, `mailabs` | Pearson ≥ 0.99 **and** Spearman ≥ 0.99 **and** sign@0 ≥ 0.999 | up to 1% |
| `spoofceleb` | Spearman ≥ 0.99 alone | zero |

Bit-exact reproduction is not achievable — the references were produced in a
different environment, which introduces a near-constant logit offset with
r > 0.99. That offset is irrelevant to EER, which is rank-based, so every policy
asks for detection-equivalence rather than absolute agreement. SpoofCeleb drops
the Pearson requirement because tail outliers once dragged it to 0.92 on models
whose Spearman was 0.996.

A reference that is itself >50% NaN reports `REF_UNUSABLE` and exits 0 — the
reference is the broken side and must not fail your run.

### ASVLD report (descriptive, no pass/fail)

```bash
python -m spoof_superb.verification.asvld_report
```

### fp32 noise re-run promotion gate

```bash
python -m spoof_superb.verification.noise_rerun_gate            # verify only
python -m spoof_superb.verification.noise_rerun_gate --promote  # verify, then swap
```

`--promote` **moves directories** on all-pass. Run without it first.

### Protocol and composition checks (in `analysis/`)

```bash
python -m spoof_superb.analysis.verify_tts_protocols --master M.csv --lookup L.csv
python -m spoof_superb.analysis.verify_and_split_condition_scores --help
python -m spoof_superb.analysis.check_condition_composition --help
python -m spoof_superb.analysis.verify_mlaad_column --tex access.tex
```

`verify_mlaad_column` cross-checks the paper's MLAAD column three ways:
transcription against `access.tex` (≤ 0.0005), the repo's EER estimator against
an independent sklearn/Brent one (≤ 0.01 pp), and the full pool against the
balanced pool (≤ 0.2 pp). **The duplicate estimator is deliberate** — it exists
so a bug in `compute_det_curve` cannot be reproduced by its own verifier. Do not
"deduplicate" it.

---

## 6. Analysis — tables and figures

These consume score files and produce CSVs, then figures from those CSVs. Run
them in dependency order.

### The paper's main table

```bash
python -m spoof_superb.analysis.recompute_table5_mlaad_v10 --out_dir out/table5
```

Recomputes Tables 5 and 6 and writes `table5_mlaad_v10.json`. It carries its own
internal reproduction gate over the untouched published cells. This is the
authority for what Table 5 reports.

### Acoustic degradation (Section 5.2)

```bash
python -m spoof_superb.analysis.compute_eer_matrix \
    --baseline_dir $SCORES/Baseline_by_Hashim \
    --augmented_dir $SCORES/scores_by_acoustic_degradation \
    --output_csv out/eer_matrix.csv

python -m spoof_superb.analysis.create_heatmap --csv out/eer_matrix.csv --out_dir out/figures
```

### TTS diversity

```bash
# score trees first
python -m spoof_superb.analysis.organize_tts_scores --help
python -m spoof_superb.analysis.build_mlaad_dir_map        # -> mlaad_v10_dir_to_system.csv
python -m spoof_superb.analysis.organize_mlaad_scores

# normalisation (needed by the EER-vs-pooled-bonafide path)
python -m spoof_superb.analysis.apply_zscore_and_pool --linear_head_dir ... --tts_dir ... --out_base ...

# metrics
python -m spoof_superb.analysis.compute_eer_tts   --norm_dir ... --tts_dir ... --out_dir out/tts_eer
python -m spoof_superb.analysis.compute_far_matrix --tts_dir ... --combined_dir ... --out_dir out/tts_far
python -m spoof_superb.analysis.create_mlaad_tts_eer_heatmaps --out-dir out/mlaad_tts

# figures
python -m spoof_superb.analysis.create_tts_eer_heatmaps --eer_dir out/tts_eer --out_dir out/figures
python -m spoof_superb.analysis.create_tts_heatmaps     --far_dir out/tts_far --out_dir out/figures
python -m spoof_superb.analysis.create_mlaad_group_ranked_figures
python -m spoof_superb.analysis.create_mlaad_tts_system_ranked_figure
```

### Ad-hoc EER of a score file or directory

```bash
python -m spoof_superb.analysis.evaluate_score_file --score_filepath out.txt --score_file_has_keys
python -m spoof_superb.analysis.evaluate_score_directory --input_dir DIR --output_dir OUT
```

### Other

```bash
python -m spoof_superb.analysis.layer_weight_analysis          # SSL layer weights (no CLI)
python -m spoof_superb.analysis.plot_score_distributions --help
python -m spoof_superb.analysis.analyze_benchmark_distribution --help
python -m spoof_superb.analysis.create_taxonomy                # figure, no CLI
python -m spoof_superb.analysis.create_SSL_taxonomy            # figure, no CLI
```

Six modules have **no CLI** and read hardcoded constants:
`layer_weight_analysis`, `create_mlaad_group_ranked_figures`,
`create_mlaad_tts_system_ranked_figure`, `create_taxonomy`, `create_SSL_taxonomy`.
Edit the constants at the top of the file to retarget them.

`create_taxonomy` and `create_SSL_taxonomy` write to a **CWD-relative**
`outputs/figures/` path — run them from the repo root or they land somewhere
unexpected.

---

## 7. Data preparation

```bash
python -m spoof_superb.data.prep.append_mailabs --dry-run   # then without --dry-run
python -m spoof_superb.data.prep.balance_mailabs --dry-run
python -m spoof_superb.data.prep.make_tsv_mlaad
python -m spoof_superb.data.prep.report_mlaad
```

`append_mailabs` folds the staged M-AILABS bonafide scores into each MLAAD v10
score file. It is a **separate, guarded step** from the `mailabs` orchestration
job precisely so a crash mid-run cannot leave a half-appended file. Always
`--dry-run` first.

Checkpoint selection helper:

```bash
python -m spoof_superb.tools.select_aasist_ckpt --run_dir ...
```

---

## 8. Tests and the numerical gate

```bash
pytest tests/ -q                  # ~16 s, 28 contract tests
RUN_TABLE5=1 pytest tests/ -q     # + the numerical regression gate (~2m40s)
bin/reproduce_table5.sh           # same gate, standalone
```

The gate re-runs `recompute_table5_mlaad_v10` and diffs every per-model,
per-dataset EER against `tests/baseline_table5.json` at **zero tolerance**.
These are deterministic recomputations over fixed score files, not
re-inference, so any drift at all means something moved that should not have.

Run it after any change that touches scoring, metrics, or the score-file format.

If the gate legitimately needs a new baseline (you re-scored something on
purpose), regenerate it deliberately:

```bash
python -m spoof_superb.analysis.recompute_table5_mlaad_v10 --out_dir /tmp/t5
cp /tmp/t5/table5_mlaad_v10.json tests/baseline_table5.json
```

and say why in the commit message.

---

## 9. End-to-end recipes

**Score one new SSL model on every published column**

```bash
CKPT=$MODELS/model_weighted_CCE_50_64_linear_head_ASV19_mynewssl/swa.pth
for ds in eval_2019 asvspoof2021_LA asvspoof2021_DF asvspoof5 deepfake_eval_2024 \
          wild Famous_Figures spoofceleb Multilingual asvspoofLD; do
    bin/score.sh --model linear_head --ssl_model mynewssl --model_path $CKPT \
        --dataset $ds --output_file out/linear_head_${ds}_mynewssl.txt
done
```

**Re-run SpoofCeleb for all models, verify, then check the paper table**

```bash
bin/orchestrate.sh --job spoofceleb
cat $SCORES/linear_head_SpoofCeleb/SUMMARY.txt
bin/reproduce_table5.sh
```

**Smoke-test a change without a GPU**

`lfcc_gmm` is CPU-only, so it exercises the whole trial-list → resolve → score →
write path cheaply:

```bash
bin/score.sh --model lfcc_gmm \
    --model_path $BASELINE_MODELS/lfcc_gmm \
    --dataset deepfake_eval_2024 --limit 300 --n_jobs 8 \
    --output_file /tmp/smoke.txt
```

---

## 10. Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'spoof_superb'` | not at the repo root and `PYTHONPATH` unset — use `bin/*.sh`, or export it |
| Scoring exits **rc=2** immediately | CUDA requested but unavailable. Deliberate: it refuses to fall back to CPU |
| Orchestrator says "existing output is complete" | resume kicked in; pass `--force` |
| NaN in scores | almost certainly `--amp`. Re-run in fp32 (the default) |
| An ASVLD condition produced no output | `Filtering` is in the default `--skip_conditions` |
| `numpy.genfromtxt` chokes on a score file | utt_ids contain spaces — use the `.tsv` twin written beside it |
| Analysis script writes figures to an odd path | `create_taxonomy` / `create_SSL_taxonomy` use CWD-relative output; run from the repo root |
| Table 5 gate fails after a refactor | read the diff it prints; a path or a parser moved |

**Known environment hazard.** Two conda environments on this host disagree on
`soxr` (1.0.0 vs 0.5.0.post1), which is librosa's resampler. Score files
produced under different interpreters for datasets that get resampled to 16 kHz
are not strictly comparable. Everything now runs under `cfg.python` (the running
interpreter) so new runs are self-consistent, but existing files predate that.
See `humanpending.md` item RP-1.
