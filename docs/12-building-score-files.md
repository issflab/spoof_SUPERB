# 12. Building the score files from scratch

A runbook for rebuilding `scores_root` yourself, rather than inheriting it.

This is **option A**: rebuild the published benchmark, then extend it. The
published reference score files are an *input*, not an output — see
[why](#why-the-reference-files-are-an-input) at the bottom.

> **Status.** Steps 0–3 are runnable today. Steps 4–6 need code that does not
> exist yet; each is marked and says what is missing. Read the whole page
> before starting so you know where the road ends.

---

## Step 0 — Environment and config

```bash
conda activate spoof_SUPERB
cd /home/alhashim/ASD_SUPERB/spoof_SUPERB

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m spoof_superb.config
pytest tests/ -q
```

Decide where the new tree lives and point the config at it. Keep the old tree
untouched so you can compare:

```yaml
# configs/paths.yaml
scores_root: /data/ssl_anti_spoofing/asd_superb_scores_v2
```

```bash
python -m spoof_superb.config | grep scores_root     # confirm
```

**Checkpoint:** `scores_root` is the new path, and `models_root` /
`data_root` point at real directories.

---

## Step 1 — Seed the reference score files

Seven of the ten columns get their trial list from a published score file, so
those files must exist before anything can be scored against them.

```bash
OLD=/data/ssl_anti_spoofing/asd_superb_score_files
NEW=$(python -c "from spoof_superb.config import cfg; print(cfg.scores_root)")

mkdir -p "$NEW/linear_head"
for ds in eval_2019 asvspoof2021_LA asvspoof2021_DF asvspoof5 \
          deepfake_eval_2024 wild Famous_Figures spoofceleb; do
    cp "$OLD/linear_head/linear_head_${ds}_xls_r_300m.txt" "$NEW/linear_head/"
done

# the two pooled columns
cp "$OLD/linear_head/linear_head_asvspoofLD_xls_r_300m.txt" "$NEW/linear_head/"
mkdir -p "$NEW/asvld_rerun/Recompression" "$NEW/linear_head_MLAAD_v10"
cp "$OLD/asvld_rerun/Recompression/linear_head_Recompression_xls_r_300m.txt" \
   "$NEW/asvld_rerun/Recompression/"
cp "$OLD/linear_head_MLAAD_v10/linear_head_MLAAD_v10_xls_r_300m.txt" \
   "$NEW/linear_head_MLAAD_v10/"
```

**Checkpoint:**

```bash
bin/score.sh --list_datasets
```

All ten rows must show a non-zero trial count. Expected:

```
eval_2019            71237      asvspoof5           680774
asvspoof2021_LA     181566      deepfake_eval_2024    1976
asvspoof2021_DF     152955      wild                 31779
Famous_Figures      346471      spoofceleb           91130
Multilingual       1040006      asvspoofLD         1634931
```

If a count is 0, that file did not copy.

---

## Step 2 — Score one model, one dataset

Do not start a sweep. Prove the path end to end on the cheapest thing first.

Edit `bin/score.sh`:

```bash
MODEL="lfcc_gmm"                     # CPU only, no GPU needed
MODEL_PATH="$BASELINE_MODELS_ROOT/lfcc_gmm"
SOURCE="benchmark"
DATASET="deepfake_eval_2024"         # smallest set: 1,976 trials
OUTPUT_FILE="$REPO/outputs/scores/smoke.txt"
```

```bash
bin/score.sh --limit 300
```

**Checkpoint:** it prints `scores saved -> ...` and an inline EER. Then confirm
your output matches the existing published file on the same trials:

```bash
bin/verify.sh --check mlaad \
    --new outputs/scores/smoke.txt \
    --ref $OLD/baselines/lfcc_gmm/lfcc_gmm_deepfake_eval_2024.txt
```

Expect `-> PASS` with `r=1.0000`. If it fails here, stop: something is wrong
with paths or the environment, not with the sweep.

---

## Step 3 — Segment Deepfake-Eval 2024

The segmentation is our artifact, not part of the released corpus, so it is
regenerated from the two things that are: `audio-data/` and
`audio-metadata-publish.csv`. Nothing under `Deepfake_Eval_2024/data/` is read
or written.

```bash
python -m spoof_superb.data.prep.segment_deepfake_eval --dry-run
```

Expect: `1980 (1167 bonafide, 813 spoof)`.

```bash
python -m spoof_superb.data.prep.segment_deepfake_eval --limit 20 --jobs 8   # smoke
python -m spoof_superb.data.prep.segment_deepfake_eval --jobs 16             # full
```

Writes:

```
/data/Data/Deepfake_Eval_2024/segmented/wav/{stem}_seg{N}.wav
/data/Data/Deepfake_Eval_2024/segmented/protocol.txt
```

4 s segments (= the models' 64,600-sample crop), 16 kHz mono PCM, flat, no
train/test split. Trailing fragments under 1 s are discarded.

**Checkpoint:**

```bash
wc -l /data/Data/Deepfake_Eval_2024/segmented/protocol.txt
ls /data/Data/Deepfake_Eval_2024/segmented/wav | wc -l          # = protocol rows - 1
awk -F'\t' 'NR>1{print $3}' /data/Data/Deepfake_Eval_2024/segmented/protocol.txt \
    | sort | uniq -c                                            # bonafide / spoof split
```

Two things worth knowing:

* **wav, not mp3.** 91% of the sources are already mp3, and codec compression
  is a condition this benchmark measures. Re-encoding mp3 → mp3 would inject
  the artifact under study into the clean condition. `--format mp3` exists if
  you want it anyway.
* **All 1,980 recordings are used**, not 1,976. The 4 files the published run
  dropped have a `.dat` extension but are really MP4 containers; librosa cannot
  open them, ffmpeg can. They yield 122 segments.

---

## Step 4 — Score the segmented set ⚠️ needs code

**Missing:** the segmented set is not in the dataset registry, so
`--dataset deepfake_eval_2024_segmented` does not exist yet.

What has to be added to `spoof_superb/scoring/datasets.py`:

1. a trial source that reads the tab-separated `segmented/protocol.txt`
   (`segment_id`, `source_file`, `label`, `start_s`, `duration_s`)
2. a resolver mapping `segment_id` → `{root}/segmented/wav/{segment_id}`
3. a `DATASETS` entry so it is selectable like any other column

This is small — the four existing trial sources are the template.

---

## Step 5 — Full ASV21 DF and Famous Figures ⚠️ needs code

The published columns are subsets whose selection rule was never recorded:

| Column | Published | Full protocol |
|---|---|---|
| ASV21 DF | 152,955 | 611,829 (`trial_metadata.txt`) |
| Famous Figures | 346,471 | 348,135 (`protocol.txt`, minus header) |

**Missing:** there is no tool that turns a raw protocol into a trial list, and
no dataset entry for the full sets.

**These columns will no longer match the paper.** Scoring 611,829 instead of
152,955 trials is a different measurement. Plan for it: keep the published
columns alongside the full ones rather than overwriting, so the two are
comparable and you can say what changed.

---

## Step 6 — Sweep, verify, rebuild the tables ⚠️ partly needs code

Once 4 and 5 exist:

```bash
bin/orchestrate.sh --job mlaad --list       # always --list first
bin/orchestrate.sh --job mlaad
bin/orchestrate.sh --job spoofceleb
bin/orchestrate.sh --job baselines
```

The three protocol-driven sets (`asvspoofLD`, `Multilingual`, `spoofceleb`)
plus M-AILABS need no reference files at all. The other seven do.

**Missing:** there is no orchestrator job covering the seven reference-driven
columns for the SSL models — only `baselines` sweeps them, and only for the two
non-SSL models.

Then:

```bash
python -m spoof_superb.data.prep.append_mailabs --dry-run   # merge M-AILABS into MLAAD
python -m spoof_superb.data.prep.append_mailabs
bin/reproduce_table5.sh
```

`bin/reproduce_table5.sh` compares against `tests/baseline_table5.json`. Once
you have deliberately changed the ASV21 DF and Famous Figures columns it will
fail, correctly. Re-capture the baseline only when the new numbers are the
intended ones, and say why in the commit:

```bash
python -m spoof_superb.analysis.recompute_table5_mlaad_v10 --out_dir /tmp/t5
cp /tmp/t5/table5_mlaad_v10.json tests/baseline_table5.json
```

---

## Why the reference files are an input

Seven columns take their trial list from a published score file rather than
from a raw protocol, because the published sets are subsets and the selection
rule is not recorded anywhere — ASV21 DF is ~25% of the protocol, stratified
across all three phases and all nine codec conditions, with no recorded seed.

Re-deriving those lists from raw protocols gives a *different benchmark*: the
numbers would be internally consistent but not comparable to the paper. So for
a faithful rebuild, the reference files are part of the benchmark definition,
like a protocol file. Step 5 is the deliberate decision to depart from that.

## Order summary

| Step | Needs | Runnable |
|---|---|---|
| 0 Environment and config | — | yes |
| 1 Seed reference files | old tree | yes |
| 2 Smoke test one model | step 1 | yes |
| 3 Segment Deepfake-Eval | ffmpeg | yes |
| 4 Score segmented set | registry entry | **no** |
| 5 Full ASV21 DF / FF | trial-list tool | **no** |
| 6 Sweep and rebuild tables | steps 4, 5 | partly |
| 7 Generate `MANIFEST.csv` | a finished tree | later |

The manifest comes last. It is derived from the tree by walking it, never a
prerequisite.
