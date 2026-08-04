"""Verification: the separate step that compares a finished run to the reference.

Nothing in scoring, orchestration or analysis compares itself against anything.
Comparison lives here, behind one entry point, at two independent levels:

    python -m spoof_superb.verification all

    level 1  score files      candidate tree  vs  reference/manifest.json
                                              or  a reference score tree
    level 2  analysis tables  candidate CSVs  vs  reference/analysis/

Module map:

    verdicts.py   the two ladders and the thresholds, with the measurements
                  that set them. Read this first -- it is where the design is.
    cells.py      per-(dataset, model) measurement. Measures, never grades.
    scores.py     level 1: the sweep and its grading
    analysis.py   level 2: structure, cells, and the paper's claims
    report.py     Markdown for a human, JSON for a script
    __main__.py   the CLI

    noise_rerun_gate.py   the fp32 ASVLD promotion gate (unrelated, current)
    asvld_report.py       descriptive ASVLD table (unrelated, current)

`driver.py`, `policies.py` and `stats.py` were removed here. They graded one
fresh file against one legacy file, with thresholds tuned per corpus to the
legacy environment's constant logit offset, and were called from inside the
scoring sweep. `verdicts.py` replaces them with a ladder that is
dataset-independent: it grades on the EER over identical trials -- the quantity
a reproduction claim is about -- and reports the correlation statistics beside
it rather than thresholding them per corpus.
"""
