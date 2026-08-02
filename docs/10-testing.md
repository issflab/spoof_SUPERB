# 10. Tests

```bash
pytest tests/ -q                  # ~16 s, no GPU, no corpora
RUN_MAIN_RESULTS=1 pytest tests/ -q     # + the numerical gate (~2m40s, reads ~15 GB)
```

Run the first after any code change. Run the second after anything touching
scoring, metrics, or the score-file format.

## What is covered

| File | Contracts |
|---|---|
| `test_config.py` | configuration layering: the YAML loads automatically, env beats YAML, partial files fall back, import has no side effects |
| `test_aasist_raw.py` | the AASIST baseline: parameter count matches the published model, forward shape, node counts, gradients flow, sinc filterbank stays fixed, no SSL upstream crept in |
| `test_grad_accum.py` | gradient accumulation is behaviour-preserving, plus a guard proving that claim is not vacuous |
| `test_lfcc_frontend.py` | the vendored LFCC front-end reproduces the reference spafe implementation bit-for-bit on real audio |
| `test_scoring_driver.py` | the parsers and writer behind the merged scoring driver: right-peeled fields, pooled column order, atomic writes, the `.tsv` twin, walk filtering, per-utterance labels, restrict semantics, fp32 default |
| `test_verification.py` | the two grade policies genuinely differ, and an unusable reference is not our failure |
| `test_main_results_regression.py` | every published EER is unchanged (opt-in) |

`test_lfcc_frontend.py` skips rather than fails when its external dependencies
(a second interpreter with `spafe`, the reference checkout, ASVspoof2019 audio)
are absent. Unevaluatable is not the same as broken.

## The numerical gate

`tests/baseline_main_results_table.json` is the output of the main-results reproducer captured
on a known-good tree. `test_main_results_regression.py` re-runs the reproducer and
diffs every per-model, per-dataset EER, row count and NaN fraction against it at
**zero tolerance**.

Zero is right because this recomputes over fixed score files rather than
re-inferring, so it is deterministic -- confirmed by two consecutive runs
producing byte-identical output. Any drift at all means something moved that
should not have.

It is opt-in via `RUN_MAIN_RESULTS=1` because a run reads ~15 GB and takes ~2m40s.

### If the gate fails

Read the diff it prints; it names the model, dataset and field that moved. The
usual causes are a path that now resolves somewhere else, or a parser that
changed.

### Regenerating the baseline

Only when you have deliberately re-scored something and the new numbers are the
intended ones:

```bash
python -m spoof_superb.analysis.recompute_main_results --out_dir /tmp/t5
cp /tmp/t5/main_results.json tests/baseline_main_results_table.json
```

Say why in the commit message. Silently refreshing this file defeats its
purpose.

## Writing new tests

Each test should name the contract it protects and fail precisely when that
contract is violated. The existing files put the contract list in the module
docstring with an identifier (`C1`, `A2`, `S3`, `V4`, `G1`) referenced from the
test names -- follow that, because it makes a failure self-explaining.
