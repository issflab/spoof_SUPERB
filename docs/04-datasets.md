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
├── Deepfake_Eval_2024/
│   ├── audio-data/                         mixed .mp3 / .m4a / .mp4 / .wav
│   ├── audio-metadata-publish.csv
│   └── segmented/                          built by us, see below
│       ├── wav/
│       └── protocol.txt
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
| `asvspoofLD` | bare id | condition looked up, then `ASVspoofLD/{cond}/flac/{utt}.flac` |

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

**ASVLD** — audio is split by laundering condition (`ASVspoofLD/{cond}/flac/`),
but a utt_id does not say which condition it belongs to. The resolver parses all
five condition protocols once into a 2,065,873-entry utt_id → condition index.
Note the upstream filename misspelling,
`ASVspoofLauneredDatabase_{condition}.txt`, reproduced deliberately.

The combined protocol we now build **does** carry a `condition` column, so for a
protocol-driven run that index is redundant work -- `trials_from_protocol` reads
only `utt_col` and `label_col` and discards the rest. Harmless but wasteful;
tracked as P10 in `internal/PLANNED_CHANGES.md` rather than changed under a running sweep.

## Where trial lists come from

**Every dataset reads its trial list from a protocol file.** All twelve, with no
exceptions and no per-dataset special cases. This is the whole of the default
behaviour; `--source` exists only as an override for one-off work.

### The protocol path for every dataset

Declared once, in `PROTOCOL_SPECS` in `spoof_superb/scoring/datasets.py`. Every
path is built from `cfg.data_root`, so a corpus that moves is a one-line config
change and nothing else:

| Dataset | Protocol, relative to `data_root` | Ships with the corpus? |
|---|---|---|
| `eval_2019` | `ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt` | yes |
| `asvspoof2021_LA` | `ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/trial_metadata.txt` | yes |
| `asvspoof2021_DF` | `ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/trial_metadata.txt` | yes |
| `asvspoof5` | `ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv` | yes (named `.tsv`, actually space-delimited) |
| `wild` | `ds_wild/protocols/meta.csv` | yes |
| `spoofceleb` | `SpoofCeleb/metadata/evaluation.csv` | yes |
| `Famous_Figures` | `famousfigures/protocol.txt` | yes |
| `deepfake_eval_2024` | `Deepfake_Eval_2024/audio-metadata-publish.csv` | yes |
| `Multilingual` (MLAAD) | `MLAAD/combined_meta_all.txt` | **built** |
| `MAILABS` | `MAILabs/protocol.txt` | **built** |
| `asvspoofLD` | `ASVSpoofLaunderedDatabase/ASVspoofLD/protocol.txt` | **built** |
| `deepfake_eval_2024_segmented` | `Deepfake_Eval_2024/segmented/protocol.txt` | **built** |

The four **built** ones each record their builder in the registry, and the
orchestrator prints that command if the protocol is missing rather than failing
200 tasks into a sweep:

```
[orchestrate] 1 dataset(s) have no protocol on disk; their tasks will fail:
    MAILABS  build it with: python -m spoof_superb.data.prep.build_protocols mailabs
```

How each protocol is *parsed* is declared beside its path -- `delimiter`,
`utt_col`, `label_col`, `label_const`, `bonafide_when`, `strip_ext`, `add_ext`,
`rel_to`. So the per-corpus differences are data, not twelve reader functions.
`delimiter=None` splits on any whitespace, which several "tsv" files need.

Label vocabularies differ too (`Real`/`Fake`, `bona-fide`, `bonafide`,
`genuine`) and are normalised through `LABEL_ALIASES`. An unrecognised label
raises rather than being silently treated as spoof: In-the-Wild writes
`bona-fide`, and a downstream filter on `== "bonafide"` would once have dropped
all 19,963 of its bonafide trials from the EER.

### `--source`, and why it is not what you want

| `--source` | Trial list | Label |
|---|---|---|
| `protocol` | the dataset's protocol file | from the protocol |
| `benchmark` | an existing published score file | copied from that file |
| `asvld` | one ASVLD condition protocol | protocol column 4 |
| `walk` | every `.wav` under a directory | one constant, via `--label` |
| `protocol_csv` | a `file,speaker,attack` CSV | `a00` = bonafide |

`protocol` is the default for all twelve. The others remain for ad-hoc work; in
particular **`benchmark` should not be used to build a score tree.** Taking the
trial list from a previously produced score file means a new tree can only ever
reproduce the old one's coverage, and the old coverage is demonstrably partial:

| Dataset | Published trial list | Full protocol on disk | Gap |
|---|---|---|---|
| ASV21 DF | 152,955 | 611,829 (`trial_metadata.txt`, matches the `.trl` file and the flac count) | ~25% subsample, stratified across all three phases (eval 133,464 / progress 14,820 / hidden 4,671) and all 9 codec conditions. The sampling rule and seed are not recorded anywhere. |
| Famous Figures | 346,471 | 348,135 (`protocol.txt`, excluding its header) | 1,664 rows absent: 1,344 spoof, 320 bonafide. No single attribute explains them. |
| Deepfake-Eval 2024 | 1,976 | 1,980 (`audio-metadata-publish.csv`, and 1,980 files on disk) | 4 files, a strict **subset** -- 2 Fake, 2 Real, mixed train/test. They carry a `.dat` extension but are MP4 containers: librosa cannot open them, ffmpeg can. |

Those three columns were reference-driven until RP-7 closed, which is why they
were short. Reading the full protocol means **a fresh run legitimately produces
more rows than the published file**. That is reported by verification as a
coverage line, not treated as a failure -- see [verification](08-verification.md).

Comparing on Famous Figures needs the path prefix normalised first: published
ids are absolute paths under the retired NFS mount while the protocol records
`{data_root}/...`, and the bonafide `-` vs `Bonafide` remap accounts for 50,266
of the apparent difference before the real 1,664 remains.

## Protocols and preparation we had to build

Four of the twelve did not ship something usable: MLAAD has no combined
protocol, M-AILABS has none at all, ASVLD ships five disjoint ones, and
Deepfake-Eval needs segmenting before most of its audio is reachable.
SpoofCeleb is included for contrast -- it ships a protocol that works as-is.

Each writes its output beside its corpus and records its builder in
`PROTOCOL_SPECS`, so a missing protocol names its own fix.

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

> **The English-only variant still has that bug.** Its remaining consumer is
> `verify_tts_protocols.py` -- `organize_tts_scores.py` was part of the legacy
> TTS chain and has been deleted. Tracked as RP-2 in `internal/humanpending.md`; fix it
> deliberately, with a before/after comparison, because it changes downstream
> results.

Directory names in MLAAD v10 are not the canonical TTS system names, so a
mapping is built separately:

```bash
python -m spoof_superb.analysis.build_mlaad_dir_map   # -> mlaad_v10_dir_to_system.csv
```

### M-AILABS

M-AILABS has no protocol at all. It is the **bonafide** counterpart to MLAAD's
spoof audio, so every wav under the corpus is one trial with a constant label.
That used to be a directory walk at scoring time; it is now written down once:

```bash
python -m spoof_superb.data.prep.build_protocols mailabs --dry-run
python -m spoof_superb.data.prep.build_protocols mailabs
```

Writing it down matters more than it looks. A directory walk has no record: two
runs months apart can enumerate different sets -- a re-download, an added file,
a filesystem that orders differently -- and nothing in the output says so. A
protocol file is an artifact you can diff. 584,006 rows, all bonafide, ids
relative to `data_root` so they pool with MLAAD's without rewriting.

The builder skips macOS AppleDouble sidecars (`._name.wav`) -- 176-245 byte
metadata stubs that are not audio. M-AILABS carries 6 and they are absent from
the published score files.

### Scoring MLAAD and M-AILABS: separately, then combine

Each is single-class, so **neither yields an EER on its own** -- MLAAD is
456,000 spoof with no bonafide, M-AILABS is 584,006 bonafide with no spoof.
Pooled they are 1,040,006, which is exactly the published MLAAD row including
both class counts. The MLAAD column is the pool, and until the pooling runs
there is no MLAAD number.

The orchestrator scores both and marks them `ok`, because each protocol was read
in full and contains no NaN -- it does not know the column is not yet a column.
**This is tracked as P8 in `internal/PLANNED_CHANGES.md` and is open.** The tooling below
is the pre-reorganisation path and predates the v2 score layout, so check it
before relying on it:

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

### Deepfake-Eval 2024 (segmented)

Deepfake-Eval ships whole recordings of wildly varying length, from seconds to
minutes. The models read a fixed 4.0375 s window (`CROP` = 64,600 samples at
16 kHz), so a long recording contributes exactly one scored window and the rest
of its audio is never seen. Segmenting first turns each recording into several
trials and makes the whole corpus reachable.

The segmentation is **our artifact** -- it does not ship with the dataset -- so
it is regenerated from the two things that do:

```bash
python -m spoof_superb.data.prep.segment_deepfake_eval --dry-run   # expect 1,980
python -m spoof_superb.data.prep.segment_deepfake_eval --limit 20 --jobs 8
python -m spoof_superb.data.prep.segment_deepfake_eval --jobs 16
```

Writes:

```
{data_root}/Deepfake_Eval_2024/segmented/wav/{stem}_seg{N}.wav
{data_root}/Deepfake_Eval_2024/segmented/protocol.txt
```

4 s segments, 16 kHz mono PCM, flat -- no train/test split. Trailing fragments
under 1 s are discarded. The protocol is tab-separated:
`segment_id, source_file, label, start_s, duration_s`.

Check it:

```bash
wc -l {data_root}/Deepfake_Eval_2024/segmented/protocol.txt
ls {data_root}/Deepfake_Eval_2024/segmented/wav | wc -l      # = protocol rows - 1
```

Two decisions worth knowing:

* **wav, not mp3.** 91% of the sources are already mp3, and codec compression
  is one of the degradation conditions this benchmark measures. Re-encoding
  mp3 to mp3 would inject the artifact under study into the clean condition.
  `--format mp3` exists if you want the smaller files anyway.
* **All 1,980 recordings are used, not 1,976.** The four the published run
  dropped carry a `.dat` extension but are really MP4 containers: librosa
  cannot open them, ffmpeg can. They yield 122 segments.

Once built, it is a dataset like any other:

```bash
bin/score.sh    # DATASET="deepfake_eval_2024_segmented"
```

**The segmented set is what a default sweep scores.** It is in
`DEFAULT_DATASETS`; the unsegmented `deepfake_eval_2024` is not, though it stays
scoreable by name and the two write to different paths, so both can sit on disk:

```bash
bin/orchestrate.sh                                 # segmented
bin/orchestrate.sh --datasets deepfake_eval_2024   # the published column
```

It is still marked non-benchmark -- a derived set, not a *published* column -- so
`--source benchmark` is refused for it and it is absent from the release
manifest.

**The two are not comparable, and this is the one thing to get right.** The
published column is 1,976 trials, one 4 s window per recording; segmented is
56,481, every window of every recording. A three-minute file contributes ~45
trials instead of one, so the segmented EER weights long recordings far more
heavily. It answers a different question rather than correcting the old answer,
and any write-up has to say which column it reports.

The pre-existing `Deepfake_Eval_2024/data/` tree is a separate local artifact
with its own train/test and duration splits. This script never reads or writes
it, and the benchmark does not use it.

### SpoofCeleb

SpoofCeleb ships a usable protocol (`metadata/evaluation.csv`,
columns `file,speaker,attack`), so no construction is needed. `attack == a00`
is bonafide, everything else is spoof. It is natively 16 kHz, so no resampling
happens and agreement with the reference is near-exact.

## Verifying your protocols

```bash
python -m spoof_superb.analysis.verify_tts_protocols --master M.csv --lookup L.csv
```

`verify_tts_protocols` checks referential integrity: every lookup key must
actually occur in the stated column of the raw protocol it claims to come from.

## Next

[Scoring](05-scoring.md).
