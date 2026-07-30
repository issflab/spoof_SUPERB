"""Which SSL upstreams the paper reports, and where that list comes from.

24 trained linear heads exist on disk. The paper's main results table -- Table 6,
``\\label{tab:results_main}`` in access.tex -- prints 19 of them, plus the two
non-SSL reference systems. Scoring the other five spends tasks on columns nobody
reads, and on the two biggest corpora that is hours per model.

WHY THE LIST IS EXPLICIT HERE
-----------------------------
The first version of this module derived the roster from
``tests/baseline_table5.json`` on the argument that a hand-kept list would drift
from the paper. That argument was right about the risk and wrong about the
remedy: the baseline had **already** drifted. It carries 21 rows, two of which
the paper does not print --

    FBANK        has mean/pooled, not printed
    Mockingjay   no MLAAD cell, not printed

-- and nothing in the JSON distinguishes them from the 19 that are printed, so
no filter over that file could have recovered the real roster.

So the membership below is stated once, from the paper, and
``tests/test_paper_models.py`` reconciles it against ``access.tex`` whenever the
paper repo is checked out beside this one. Drift is now detected rather than
assumed impossible. That is the only honest arrangement when the authority for
"what the paper reports" lives in a different repository.

Display names, not slugs, because those are what the table prints and what a
human can check against it by eye. The name -> slug mapping still comes from
the regression baseline, so slugs cannot disagree with the gate.

``paper_only`` is a default, never a restriction: ``--models`` names any
upstream explicitly, in the paper or not.
"""

import functools
import json
import os

from spoof_superb import REPO_ROOT

__all__ = ["paper_models", "paper_table_rows", "is_paper_model",
           "non_paper_models", "TABLE5_BASELINE", "PAPER_TABLE_ROWS"]

TABLE5_BASELINE = os.path.join(str(REPO_ROOT), "tests", "baseline_table5.json")

#: The SSL rows of the paper's main results table, in printed order. The two
#: non-SSL reference systems (LFCC-GMM, AASIST) are excluded: they are not
#: upstreams and are scored by --systems, not --models.
PAPER_TABLE_ROWS = (
    "APC",
    "VQ-APC",
    "NPC",
    "Mockingjay-960h",
    "TERA",
    "DeCoAR 2.0",
    "wav2vec",
    "wav2vec 2.0 Base",
    "wav2vec 2.0 Large",
    "HuBERT Base",
    "HuBERT Large",
    "MR-HuBERT",
    "XLS-R",
    "UniSpeech-SAT",
    "Data2Vec",
    "WAVLABLM",
    "WavLM Large",
    "SSAST",
    "MAE-AST-FRAME",
)


def paper_table_rows():
    return PAPER_TABLE_ROWS


@functools.lru_cache(maxsize=None)
def _slug_by_display(path=None):
    """Table-row display name -> model slug, read from the regression baseline."""
    path = path or TABLE5_BASELINE
    try:
        with open(path) as f:
            results = json.load(f)["results"]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"cannot map the paper's model roster to slugs: {path} is missing. "
            f"Pass explicit --models, or restore the Table 5 baseline.")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable Table 5 baseline: {exc}")
    return {name: row["slug"] for name, row in results.items() if row.get("slug")}


@functools.lru_cache(maxsize=None)
def paper_models(path=None):
    """frozenset of the SSL slugs the paper's results table reports.

    Raises if a printed row has no slug in the baseline: that means the two have
    diverged, and guessing would either drop a reported model or score an
    unreported one.
    """
    mapping = _slug_by_display(path)
    missing = [n for n in PAPER_TABLE_ROWS if n not in mapping]
    if missing:
        raise ValueError(
            f"these paper table rows have no slug in {path or TABLE5_BASELINE}: "
            f"{missing}. PAPER_TABLE_ROWS and the baseline have diverged.")
    return frozenset(mapping[n] for n in PAPER_TABLE_ROWS)


def is_paper_model(ssl, path=None):
    return ssl in paper_models(path)


def non_paper_models(available, path=None):
    """The slugs in ``available`` that the paper's results table does not report."""
    return sorted(set(available) - paper_models(path))
