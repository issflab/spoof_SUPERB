# 10. Tests

```bash
pytest tests/ -q                  # ~25 s, no GPU, no corpora
```

Run this after any code change. The tests use synthetic fixtures throughout and
touch no score files.

**Numbers are not checked here.** Whether the pipeline still produces the
published EERs is a question about DATA, not code, and it is answered by
`python -m spoof_superb.verification all` against `reference/` -- a refreshable
published artefact rather than a capture pinned to a tree. See
[verification](08-verification.md).

## What is covered

| File | Contracts |
|---|---|
| `test_config.py` | configuration layering: the YAML loads automatically, env beats YAML, partial files fall back, import has no side effects |
| `test_aasist_raw.py` | the AASIST baseline: parameter count matches the published model, forward shape, node counts, gradients flow, sinc filterbank stays fixed, no SSL upstream crept in |
| `test_grad_accum.py` | gradient accumulation is behaviour-preserving, plus a guard proving that claim is not vacuous |
| `test_lfcc_frontend.py` | the vendored LFCC front-end reproduces the reference spafe implementation bit-for-bit on real audio |
| `test_scoring_driver.py` | the parsers and writer behind the merged scoring driver: right-peeled fields, pooled column order, atomic writes, the `.tsv` twin, walk filtering, per-utterance labels, restrict semantics, fp32 default |
| `test_verification_levels.py` | each verdict on both ladders means exactly one thing, and the boundaries are where the reasoning says |
| `test_seeding.py` | a seed reproduces all three RNGs, and the config shapes that kept two seeding functions alive stay distinguishable |
| `test_compare_trees.py` | the ad-hoc tree comparison distinguishes WHY two cells differ |
| `test_views.py` | analysis views partition losslessly and compose as the paper's tables specify |

`test_lfcc_frontend.py` skips rather than fails when its external dependencies
(a second interpreter with `spafe`, the reference checkout, ASVspoof2019 audio)
are absent. Unevaluatable is not the same as broken.

## The retired numerical gate

`test_main_results_regression.py` used to re-run the main-results reproducer
against the LEGACY tree and diff every EER against
`tests/baseline_main_results_table.json` at zero tolerance. It has been removed.

It was answering the right question in the wrong place. The reference was a
capture of one tree at one commit, unrefreshable without contradicting its own
purpose, and it pinned the reproducer to a tree that is no longer authoritative.
`python -m spoof_superb.verification` asks the same question -- did the numbers
move -- against a reference that is a published artefact anyone can rebuild, and
it grades on whether the paper's claims survive rather than on a tolerance.

The one live part of its fixture -- the display-name to slug mapping -- was
extracted to `spoof_superb/scoring/paper_roster.json`, which is source data that
production code reads, not a test fixture. The other 97% was per-model EERs
measured on the legacy tree; they are stale and superseded by `reference/`.

## Writing new tests

Each test should name the contract it protects and fail precisely when that
contract is violated. The existing files put the contract list in the module
docstring with an identifier (`C1`, `A2`, `S3`, `V4`, `G1`) referenced from the
test names -- follow that, because it makes a failure self-explaining.
