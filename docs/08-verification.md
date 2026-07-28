# 8. Verification

Three different kinds of check. They are deliberately separate.

## 1. A new score file against its published reference

```bash
bin/verify.sh                 # edit CHECK / NEW_FILE / REF_FILE at the top
```

or:

```bash
python -m spoof_superb.verification.driver \
    --check spoofceleb --new out.txt --ref reference.txt
```

Exit 0 = pass, 1 = fail. The orchestrator runs this automatically after each
task in a job that declares a policy.

### The policies are not interchangeable

| `--check` | Verdict | NaN tolerance in *your* output |
|---|---|---|
| `mlaad`, `mailabs` | Pearson ≥ 0.99 **and** Spearman ≥ 0.99 **and** sign@0 ≥ 0.999 | up to 1% |
| `spoofceleb` | Spearman ≥ 0.99 alone | none at all |

Bit-exact reproduction of a reference is not achievable. The published files
were produced in a different environment -- different librosa / soxr / torch /
CUDA -- which introduces a near-constant logit offset (~0.33 for `xls_r_300m`)
with r > 0.99. That offset is irrelevant to EER, which is rank-based. So every
policy asks for *detection-equivalence*, not absolute agreement.

SpoofCeleb drops the Pearson requirement because on the MLAAD run a handful of
tail outliers dragged Pearson to 0.92 on models whose Spearman was 0.996 --
failing a Pearson-gated check for no detection-relevant reason. It tolerates no
NaN because SpoofCeleb is natively 16 kHz, no resampling happens, and agreement
should be near-exact.

If you are tempted to merge these two behind a flag: don't. That erases the
reasoning above.

### Reading the output

```
[verify] new=91130 ref=91130 shared=91130 both=91130 r=1.0000 spearman=1.0000
         sign@0=99.9934% offset=-0.001±0.008 maxΔ=0.200 -> PASS
```

`REF_UNUSABLE` means the *reference* is more than 50% NaN -- the broken side is
theirs, your output is finite, and it exits 0. It is a report, not a failure.

## 1b. Against the shipped reference pack (no large download)

The full score files are ~6 GB, so they are not in the repo. What *is* in the
repo is a few-MB pack that answers the same questions:

```
trials/published/{dataset}.tsv.gz        the trial list the benchmark used
reference/summary.json                   per (dataset, model): counts, EER, sha256
reference/subsample/{dataset}/{model}.tsv.gz   2,000 reference scores
```

```bash
python -m spoof_superb.verification.driver --check spoofceleb --pack \
    --dataset spoofceleb --model xls_r_300m --new my_scores.txt
```

```
[coverage] spoofceleb/xls_r_300m  scored 91130  published 91130  overlap 91130
           published-not-scored 0  scored-not-published 0
[ranking ] on 2000 subsample rows: ... spearman=1.0000 ... -> PASS
[expected] published EER = 1.2340% over 91130 rows
```

Three separate answers: **coverage** (did you score the right trials?),
**ranking** (do your scores order them the same way?), and **expected** (is
your headline EER where it should be?).

Grading ranking on 2,000 rows is not a compromise: a 2,000-row subsample
reproduces the full-file Spearman to within about 3e-6, against a 0.99 pass
threshold.

**Coverage is reported, never enforced.** Scoring the full protocol where the
published column used a subset is a deliberate, legitimate difference; the
point is that it appears as a line of output rather than silently changing what
gets compared.

Regenerate the pack from a full score tree with:

```bash
python -m spoof_superb.tools.build_reference_pack --dry-run
python -m spoof_superb.tools.build_reference_pack
```

## 2. The fp32 ASVLD noise re-run promotion gate

```bash
python -m spoof_superb.verification.noise_rerun_gate            # verify only
python -m spoof_superb.verification.noise_rerun_gate --promote  # verify, then swap
```

This gates replacing archived score files with the fp32 re-run. Five contracts:
same utterance sequence and order, same labels, no NaN, Pearson ≥ 0.9998 on
utterances the archive scored finitely, and |ΔEER| ≤ 0.15 pp for models that
had no archived NaN. Models *with* archived NaN are exempt by construction --
correcting them is the point, and their EER moves by 5.8 to 8.0 pp.

**`--promote` moves directories.** Run without it first and read the table.

## 3. Descriptive and structural checks

```bash
# ASVLD rerun vs reference, per (model, condition). No pass/fail, just a table.
python -m spoof_superb.verification.asvld_report

# every utt_id in a condition file is attributable to exactly one source
python -m spoof_superb.analysis.verify_and_split_condition_scores --help

# exact expected row counts per condition
python -m spoof_superb.analysis.check_condition_composition --help

# the TTS protocol CSVs actually match the raw dataset protocols
python -m spoof_superb.analysis.verify_tts_protocols --master M.csv --lookup L.csv

# the paper's MLAAD column, three ways
python -m spoof_superb.analysis.verify_mlaad_column --tex access.tex
```

`verify_mlaad_column` checks the number printed in the paper against a fresh
recomputation (≤ 0.0005), the repo's EER estimator against an independent
sklearn/Brent one (≤ 0.01 pp), and the full pool against the balanced pool
(≤ 0.2 pp).

**Its duplicate EER implementation is deliberate.** It exists so that a bug in
`core/metrics.py::compute_det_curve` cannot be reproduced by the very code
verifying it. Do not "deduplicate" it.

## Which one do I want?

| Situation | Use |
|---|---|
| I re-scored a model and want to know if it matches | `bin/verify.sh` |
| I re-ran ASVLD noise in fp32 and want to replace the archive | `noise_rerun_gate` |
| I changed code and want to know if the paper's numbers moved | [tests](10-testing.md) |
| I want to know if my protocol CSVs are self-consistent | `verify_tts_protocols` |
