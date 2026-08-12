# 5. Scoring from checkpoints

One entry point covers every model and every evaluation set. It produces the
canonical 4-column score file described in
[reproducing results](02-reproducing-results.md).

Before this: make sure the corpora are laid out as
[datasets and protocols](04-datasets.md) describes, and that
`configs/paths.yaml` points at them.

## The easy way

Edit the settings block at the top of `bin/score.sh`, then:

```bash
bin/score.sh
```

The script prints the exact command it runs, so you can see what the settings
map to. Anything you add on the command line is appended:

```bash
bin/score.sh --limit 300        # smoke test on the first 300 trials
bin/score.sh --list_datasets    # the 10 published sets and their trial counts
```

## The direct way

```bash
python -m spoof_superb.scoring.driver --model linear_head --ssl_model xls_r_300m \
    --model_path $MODELS/model_weighted_CCE_50_64_linear_head_ASV19_xls_r_300m/swa.pth \
    --dataset wild --output_file outputs/scores/wild_xls_r_300m.txt
```

## Choose a back-end

| `--model` | Checkpoint | Notes |
|---|---|---|
| `linear_head` | `swa.pth` | the SSL models; needs `--ssl_model`; batch default 32 |
| `aasist_raw` | `swa.pth` | non-SSL baseline, no upstream; batch default 64 |
| `lfcc_gmm` | a **directory** of GMMs | CPU only; use `--n_jobs`, not `--batch_size` |

`lfcc_gmm` scores the full utterance with no 4-second crop:
`GaussianMixture.score` returns the mean per-frame log-likelihood, so the LLR is
already length-normalised. The crop is an AASIST-side choice.

## The dataset decides its own trial source

`--dataset` is the single input. It determines the trial list, the audio root
and where the output file is placed:

```bash
bin/score.sh          # DATASET="spoofceleb" is enough
```

```bash
bin/score.sh --list_datasets
```

```
spoofceleb             source=protocol_csv (from the corpus/protocol)
Multilingual           source=walk         (from the corpus/protocol)
MAILABS                source=walk         (from the corpus/protocol)
asvspoofLD             source=asvld        (from the corpus/protocol)
wild                   source=benchmark    trials=31779 ref=linear_head_wild_...
```

Datasets with their own protocol are scored from it. The seven that show
`source=benchmark` have no corpus-derived trial list yet and read the published
score file instead.

Setting the dataset and the protocol separately used to be possible, and they
could disagree silently: `--dataset wild --source protocol_csv` scored
SpoofCeleb audio and filed it under `in_the_wild`. The dataset now carries its
own parameters, so that state cannot be constructed.

`--source` and the per-source flags remain as overrides for unusual runs, and
an explicit flag always wins:

```bash
bin/score.sh --protocol_csv /path/to/my_subset.csv
```

## Trial sources in detail

### `--source benchmark` (default) -- a published column

```bash
bin/score.sh    # SOURCE="benchmark", DATASET="wild"
```

Sets: `eval_2019`, `asvspoof2021_LA`, `asvspoof2021_DF`, `asvspoof5`,
`deepfake_eval_2024`, `wild`, `Famous_Figures`, `spoofceleb`, `Multilingual`,
`asvspoofLD`.

The trial list and the ground-truth labels are read from the published
reference score file for `reference_ssl`, so your output lines up row-for-row
with the SSL models' and can go in the same table. Use `--reference_file` to
point at a specific file instead.

### `--source asvld` -- one laundering condition

```bash
bin/score.sh    # SOURCE="asvld", ASVLD_CONDITION="Noise_Addition"
```

Conditions: `Noise_Addition`, `Reverberation`, `Resampling`, `Recompression`,
`Filtering`.

**`Filtering` is skipped by default.** It is excluded from the published ASVLD
column, and this preserves the behaviour of the old untracked `.asvld_skip`
sentinel file -- now visible as `--skip_conditions`. To score it:

```bash
bin/score.sh --skip_conditions          # empty list: skip nothing
```

### `--source walk` -- MLAAD and M-AILABS

Enumerates every `.wav` under a root and applies one label to all of them.

```bash
# MLAAD fake  (defaults: WALK_ROOT={data_root}/MLAAD/fake, WALK_LABEL=spoof)
bin/score.sh    # SOURCE="walk"

# M-AILABS bonafide
bin/score.sh    # SOURCE="walk", WALK_ROOT=".../MAILabs", WALK_LABEL="bonafide"
```

### `--source protocol_csv` -- SpoofCeleb

```bash
bin/score.sh    # SOURCE="protocol_csv"
```

Labels are per-utterance from the CSV, so `--label` does not apply.

## Flags that matter

**`--amp` — leave it off.** fp16 autocast is opt-in and should stay that way.
fp16 overflow is what wrote 384,157 NaN per model into the masked-spectrogram
front-ends (`tera`, `mockingjay`, `mockingjay_960hr`, `audio_albert_960hr`) --
53.93% of an ASVLD file. Those NaN were read as an EER near chance and produced
a wrong claim in an earlier draft. The default is fp32.

**`--cuda_device`** — if you request a CUDA device and CUDA is not available,
the run **fails with rc=2**. It does not fall back to CPU. That is deliberate: a
CPU run of MLAAD takes ~25 hours against ~20 minutes on an A100, and a silent
fallback is easy not to notice until the next morning.

**`--restrict_to REF [--restrict_prefix P]`** — score only the utt_ids present
in an existing score file, in that file's order. This is how you reproduce or
verify against a published subset.

**`--limit N`** — score the first N trials only. Use it for smoke tests.

## What a run prints

```
[wild] reference /data/.../linear_head_wild_xls_r_300m.txt
  31779 trials (11816 bonafide)
  scoring 31779 utterances
  model loaded (317431106 params) <- .../swa.pth  amp=False
  scores saved -> outputs/scores/wild_xls_r_300m.txt  (31779 lines)
  EER = 7.4210 %
```

The inline EER is a diagnostic computed from the file just written; it is
compared against nothing. To check a finished tree against the published
reference, see [verification](08-verification.md).

## Missing and undecodable audio

Two separate stages, because they are different failures:

1. Before scoring, ids whose file is not on disk are dropped and counted, so
   the DataLoader cannot die mid-run.
2. During scoring, a file that exists but does not decode is dropped and
   counted. One bad file must not kill a multi-hour run.

Both are reported as `[WARN]` lines with counts. A run that produces any
non-finite score aborts rather than writing it.

## Output

Written atomically (`.part`, then rename), so an interrupted run never leaves a
truncated file that looks complete.

A tab-separated `.tsv` twin is written automatically when any utt_id contains a
space, because `numpy.genfromtxt` -- and therefore `calculate_EER` -- cannot
parse those rows. MLAAD v10 needs it; most sets do not.

## Building score files from scratch

If `scores_root` is empty, most columns cannot be scored yet: seven of the ten
take their trial list from a published score file. Those files are an **input**
to a rebuild, not an output -- see
[where trial lists come from](04-datasets.md#where-trial-lists-come-from).

Only `asvspoofLD`, `Multilingual` and `spoofceleb` (plus M-AILABS) can be built
with no prior score files, because their trial lists come from a protocol or
from the corpus itself.

**Step 1 -- seed the reference files.**

```bash
OLD=/data/ssl_anti_spoofing/asd_superb_score_files
NEW=$(python -c "from spoof_superb.config import cfg; print(cfg.scores_root)")

mkdir -p "$NEW/linear_head"
for ds in eval_2019 asvspoof2021_LA asvspoof2021_DF asvspoof5 \
          deepfake_eval_2024 wild Famous_Figures spoofceleb; do
    cp "$OLD/linear_head/linear_head_${ds}_xls_r_300m.txt" "$NEW/linear_head/"
done

cp "$OLD/linear_head/linear_head_asvspoofLD_xls_r_300m.txt" "$NEW/linear_head/"
mkdir -p "$NEW/asvld_rerun/Recompression" "$NEW/linear_head_MLAAD_v10"
cp "$OLD/asvld_rerun/Recompression/linear_head_Recompression_xls_r_300m.txt" \
   "$NEW/asvld_rerun/Recompression/"
cp "$OLD/linear_head_MLAAD_v10/linear_head_MLAAD_v10_xls_r_300m.txt" \
   "$NEW/linear_head_MLAAD_v10/"
```

Check: `bin/score.sh --list_datasets` must show a non-zero trial count on all
ten rows.

| | | | |
|---|---|---|---|
| eval_2019 | 71,237 | asvspoof5 | 680,774 |
| asvspoof2021_LA | 181,566 | deepfake_eval_2024 | 1,976 |
| asvspoof2021_DF | 152,955 | wild | 31,779 |
| Famous_Figures | 346,471 | spoofceleb | 91,130 |
| Multilingual | 1,040,006 | asvspoofLD | 1,634,931 |

A count of 0 means that file did not copy.

**Step 2 -- prove the path on one cheap run** before starting any sweep. Use
`lfcc_gmm` on `deepfake_eval_2024`, which needs no GPU and takes about a
minute:

```bash
bin/score.sh --limit 300          # MODEL="lfcc_gmm", DATASET="deepfake_eval_2024"
python -m spoof_superb.verification scores \
    --datasets DFEval24 --models lfcc_gmm
```

Expect `IDENTICAL` or `EQUIVALENT`. A failure here is paths or environment, not
the sweep -- fix it before going further.

**Step 3 -- sweep.** See [orchestration](06-orchestration.md). Then merge
M-AILABS into MLAAD, analyse the finished tree, and verify BOTH levels -- a
tree you built yourself is the only case where level 1 says anything:

```bash
python -m spoof_superb.data.prep.append_mailabs --dry-run
python -m spoof_superb.data.prep.append_mailabs
bin/analyze.sh                  # all three analyses + level-2 verification
bin/verify.sh                   # LEVEL="all": adds level 1, the score files
```

A `MANIFEST.csv` over the finished tree, if you want one, is generated last by
walking it. It is never a prerequisite.

## Scoring many models

Do not loop over `bin/score.sh` by hand for a full sweep. See
[orchestration](06-orchestration.md).
