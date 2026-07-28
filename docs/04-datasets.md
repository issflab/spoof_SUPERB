# 4. Datasets and protocols

Read this before scoring from checkpoints. It documents **what each corpus is
assumed to look like on disk**, and the protocol files we had to construct
because the corpus did not ship one in a usable form.

Everything below is relative to `data_root` in `configs/paths.yaml`
(default `/data/Data`). The path rules are implemented as the `resolve`
functions in `spoof_superb/scoring/datasets.py`; that file is the ground truth
if this document and the code ever disagree.

## Assumed directory layout

```
{data_root}/
├── ASVSpoofData_2019/train/LA/
│   ├── ASVspoof2019_LA_train/flac/         training audio
│   ├── ASVspoof2019_LA_dev/flac/
│   ├── ASVspoof2019_LA_eval/flac/          eval_2019 audio
│   └── ASVspoof2019_LA_cm_protocols/       train / dev / eval protocols
├── ASVSpoof2021_complete/
│   ├── LA/ASVspoof2021_LA_eval/flac/
│   └── DF/ASVspoof2021_DF_eval/flac/
├── ASVSpoof5/
│   ├── No_Laundering_eval/flac/
│   └── protocols/ASVspoof5.eval.track_1.tsv
├── ASVSpoofLaunderedDatabase/ASVspoofLD/
│   ├── protocols/ASVspoofLauneredDatabase_{condition}.txt
│   ├── Noise_Addition/flac/
│   ├── Reverberation/flac/
│   ├── Resampling/flac/
│   ├── Recompression/flac/
│   └── Filtering/flac/
├── ds_wild/release_in_the_wild/            In-the-Wild, .wav
├── Deepfake_Eval_2024/audio-data/          mixed .mp3 / .m4a / .mp4 / .wav
├── famousfigures/{Speaker}/{Source}/*.wav
├── MLAAD/fake/{lang}/{tts_system}/*.wav
├── MAILabs/                                bonafide counterpart to MLAAD
└── SpoofCeleb/
    ├── flac/evaluation/
    └── metadata/evaluation.csv
```

## How a utt_id becomes a file path

| Dataset | utt_id form | Resolves to |
|---|---|---|
| `eval_2019` | carries `.flac` | `ASVSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac/{utt}` |
| `asvspoof2021_LA` | bare id | `ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/flac/{utt}.flac` |
| `asvspoof2021_DF` | bare id | `ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/flac/{utt}.flac` |
| `asvspoof5` | bare id | `ASVSpoof5/No_Laundering_eval/flac/{utt}.flac` |
| `wild` | bare id | `ds_wild/release_in_the_wild/{utt}.wav` |
| `deepfake_eval_2024` | id with `.wav` | matched **by stem** against `Deepfake_Eval_2024/audio-data/` |
| `Famous_Figures` | absolute path | rewritten, see below |
| `spoofceleb` | relative path with `.flac` | `SpoofCeleb/flac/evaluation/{utt}` |
| `Multilingual` (MLAAD) | path relative to `data_root` | `{data_root}/{utt}` |
| `asvspoofLD` | bare id | condition looked up in the protocols, then `ASVspoofLD/{cond}/flac/{utt}.flac` |

Three of these need explanation.

**Deepfake-Eval 2024** — the score files write every id with a `.wav`
extension, but on disk the files are `.mp3`, `.m4a`, `.mp4` or `.wav`. The
resolver builds a stem → real-path index once and matches on the stem.

**Famous Figures** — needs two rewrites, both verified against the full
reference file:

1. Reference ids are absolute paths under an NFS mount
   (`/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/`) that no longer
   exists; the same tree lives under `{data_root}/famousfigures`.
2. Bonafide rows carry the protocol's empty `Source` field as the literal
   directory `-`, but on disk they live under `Bonafide`. All 49,945 `/-/` rows
   are `key=bonafide` and all 49,945 resolve after the remap. Without it the
   dataset scores zero bonafide trials and its EER is undefined.

**ASVLD** — audio is split by laundering condition, but the pooled reference
score file carries no condition column. The resolver parses all five protocols
once into a utt_id → condition index. Note the upstream filename misspelling,
`ASVspoofLauneredDatabase_{condition}.txt`, which is reproduced deliberately.

## Where trial lists come from

Four mechanisms, selected with `--source`:

| `--source` | Trial list | Label |
|---|---|---|
| `benchmark` | the published reference score file | copied from that file |
| `asvld` | the ASVLD protocol for one condition | protocol column 4 |
| `walk` | every `.wav` under a directory | one constant, via `--label` |
| `protocol_csv` | a `file,speaker,attack` CSV | per-utterance: `a00` = bonafide |

`benchmark` is the default and the one that matters for comparability. The
trial list comes from an existing published score file rather than being
re-derived from a raw protocol, because several published sets are subsets
whose selection rule is not recorded anywhere. Verified against the corpora:

| Dataset | Published trial list | Full protocol on disk | Gap |
|---|---|---|---|
| ASV21 DF | 152,955 | 611,829 (`trial_metadata.txt`, matches the `.trl` file and the flac count) | ~25% subsample, stratified across all three phases (eval 133,464 / progress 14,820 / hidden 4,671) and all 9 codec conditions. The sampling rule and seed are not recorded. |
| Famous Figures | 346,471 | 348,135 (`protocol.txt`, excluding its header) | 1,664 rows absent: 1,344 spoof, 320 bonafide. No single attribute explains them. |
| Deepfake-Eval 2024 | 1,976 | 1,980 (`audio-metadata-publish.csv`, and 1,980 files on disk) | 4 files. The published set is a strict **subset** of the metadata -- 2 Fake, 2 Real, mixed train/test, most plausibly dropped as undecodable at scoring time. |

Comparing on Famous Figures requires normalising the path prefix first: the
published ids are absolute paths under the retired NFS mount, and the protocol
records `/data/Data/...`. The bonafide `-` vs `Bonafide` directory remap
accounts for 50,266 of the apparent difference before the real 1,664 remains.

Re-deriving these lists would
silently score a different trial set and produce EERs that cannot go in the same
table.

## Protocols we had to construct

Two corpora did not ship a protocol we could use.

### MLAAD

MLAAD ships one `meta.csv` per language directory and no combined protocol.
Two combiners exist:

```bash
# all 54 languages -> combined_meta_all.txt, with a `language` column
python -m spoof_superb.analysis.create_combined_mlaad_meta_all

# English only -> fake/en/combined_meta.txt, 3 columns
python -m spoof_superb.analysis.create_combined_mlaad_meta
```

The `_all` variant sets `csv.field_size_limit` and `QUOTE_NONE` because default
quoting silently drops rows -- `ja/kokoro` loses 53 of 1000. It asserts a total
of 456,000 rows so a silent loss cannot recur.

> **The English-only variant still has that bug.** Its output feeds
> `organize_tts_scores.py` and `verify_tts_protocols.py`. Tracked as RP-2 in
> `humanpending.md`; fix it deliberately, with a before/after comparison,
> because it changes downstream results.

Directory names in MLAAD v10 are not the canonical TTS system names, so a
mapping is built separately:

```bash
python -m spoof_superb.analysis.build_mlaad_dir_map   # -> mlaad_v10_dir_to_system.csv
```

### M-AILABS

M-AILABS has no protocol at all. It is the **bonafide** counterpart to MLAAD's
spoof audio, so the trial list is simply every wav under the corpus with a
constant label:

```bash
bin/score.sh    # with SOURCE="walk", WALK_ROOT=".../MAILabs", WALK_LABEL="bonafide"
```

The walk skips macOS AppleDouble sidecars (`._name.wav`) -- 176-245 byte
metadata stubs that are not audio. M-AILABS carries 6 of them and they are
absent from the reference score files.

### Scoring MLAAD and M-AILABS: separately, then combine

They are scored as two separate runs and merged afterwards, **not** combined
into one protocol first:

```bash
# 1. score MLAAD fake  (label=spoof)      -> linear_head_MLAAD_v10/
# 2. score M-AILABS    (label=bonafide)   -> linear_head_MLAAD_v10/mailabs/  (staging)
# 3. merge
python -m spoof_superb.data.prep.append_mailabs --dry-run
python -m spoof_superb.data.prep.append_mailabs
```

The M-AILABS run writes to a **staging directory**, and appending is a separate
guarded step, specifically so a crash mid-run cannot leave a half-appended
MLAAD score file that looks complete. Always `--dry-run` first.

The published MLAAD column samples ~26.4% of M-AILABS to balance the spoof
count; `balance_mailabs.py` produces that balanced variant.

### SpoofCeleb

SpoofCeleb ships a usable protocol (`metadata/evaluation.csv`,
columns `file,speaker,attack`), so no construction is needed. `attack == a00`
is bonafide, everything else is spoof. It is natively 16 kHz, so no resampling
happens and agreement with the reference is near-exact.

## Verifying your protocols

```bash
python -m spoof_superb.analysis.verify_tts_protocols --master M.csv --lookup L.csv
python -m spoof_superb.analysis.check_condition_composition --help
```

`verify_tts_protocols` checks referential integrity: every lookup key must
actually occur in the stated column of the raw protocol it claims to come from.

## Next

[Scoring](05-scoring.md).
