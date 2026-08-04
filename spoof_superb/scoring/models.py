"""Which SSL upstreams the paper reports, and where that list comes from.

24 trained linear heads exist on disk. The paper's main results table --
``\\label{tab:results_main}`` in access.tex -- prints 19 of them, plus the two
non-SSL reference systems. Scoring the other five spends tasks on columns nobody
reads, and on the two biggest corpora that is hours per model.

WHY THE LIST IS EXPLICIT HERE
-----------------------------
The first version of this module derived the roster from the regression
baseline on the argument that a hand-kept list would drift from the paper. That
argument was right about the risk and wrong about the remedy: the baseline had
**already** drifted. ``paper_roster.json`` carries 21 rows, two of which the
paper does not print --

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

__all__ = ["paper_models", "paper_table_rows", "paper_table_order",
           "display_by_slug", "family_separator_rows", "PAPER_TABLE_FAMILIES",
           "is_paper_model",
           "non_paper_models", "PAPER_ROSTER", "PAPER_TABLE_ROWS"]

#: Display name -> score-file slug, the only record of which score file produced
#: which printed row. The two cannot be derived from each other: slugs are s3prl
#: upstream names, display names are what the table prints, and no rule maps
#: between them.
#:
#: This used to be read out of `tests/baseline_main_results_table.json`, the
#: fixture of the regression gate retired in P15 -- so production code depended
#: on a test fixture, and 97% of that file was per-model EERs measured on the
#: legacy tree, stale and superseded by `reference/`. Only the mapping was live,
#: so only the mapping was kept. Git history has the rest.
PAPER_ROSTER = os.path.join(str(REPO_ROOT), "spoof_superb", "scoring",
                            "paper_roster.json")

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


#: Pretraining family of each row, in table order. The paper groups its results
#: by these and every figure rules between them, so the boundaries are stated
#: once here rather than as row indices in each plotting script -- an index is
#: only correct for one roster, and silently mis-rules any other.
#:
#: The counts are the paper's own: "the eleven discriminative models average
#: 25.68%, the two spectrogram-based models 28.77%, and the six earlier
#: generative models 33.83%".
PAPER_TABLE_FAMILIES = {
    "APC": "generative", "VQ-APC": "generative", "NPC": "generative",
    "Mockingjay-960h": "generative", "TERA": "generative",
    "DeCoAR 2.0": "generative",
    "wav2vec": "discriminative", "wav2vec 2.0 Base": "discriminative",
    "wav2vec 2.0 Large": "discriminative", "HuBERT Base": "discriminative",
    "HuBERT Large": "discriminative", "MR-HuBERT": "discriminative",
    "XLS-R": "discriminative", "UniSpeech-SAT": "discriminative",
    "Data2Vec": "discriminative", "WAVLABLM": "discriminative",
    "WavLM Large": "discriminative",
    "SSAST": "spectrogram", "MAE-AST-FRAME": "spectrogram",
}


def paper_table_rows():
    return PAPER_TABLE_ROWS


@functools.lru_cache(maxsize=None)
def display_by_slug(path=None):
    """Model slug -> the display name the paper's table prints."""
    return {slug: name for name, slug in _slug_by_display(path).items()}


def paper_table_order(slugs, path=None):
    """`slugs` reordered as the paper's table prints them, unknown ones last.

    Figures must not invent their own model order: the paper's tables and every
    figure read top-to-bottom in the same sequence, and a reader comparing them
    is entitled to assume row 3 is the same model in both.
    """
    by_slug = display_by_slug(path)
    rank = {name: i for i, name in enumerate(PAPER_TABLE_ROWS)}
    return sorted(slugs, key=lambda s: rank.get(by_slug.get(s, s), len(rank)))


def family_separator_rows(display_names):
    """Row indices where the pretraining family changes, for the rows given.

    Derived from the rows actually plotted, so a figure over a subset of the
    roster rules in the right places instead of at hardcoded indices.
    """
    return [i for i in range(1, len(display_names))
            if PAPER_TABLE_FAMILIES.get(display_names[i])
            != PAPER_TABLE_FAMILIES.get(display_names[i - 1])]


@functools.lru_cache(maxsize=None)
def _slug_by_display(path=None):
    """Table-row display name -> model slug. See PAPER_ROSTER."""
    path = path or PAPER_ROSTER
    try:
        with open(path) as f:
            roster = json.load(f)["roster"]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"cannot map the paper's model roster to slugs: {path} is missing. "
            f"Pass explicit --models, or restore paper_roster.json.")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable paper roster: {exc}")
    return dict(roster)


@functools.lru_cache(maxsize=None)
def paper_models(path=None):
    """frozenset of the SSL slugs the paper's results table reports.

    Raises if a printed row has no slug in the roster file: that means the two
    have diverged, and guessing would either drop a reported model or score an
    unreported one.
    """
    mapping = _slug_by_display(path)
    missing = [n for n in PAPER_TABLE_ROWS if n not in mapping]
    if missing:
        raise ValueError(
            f"these paper table rows have no slug in {path or PAPER_ROSTER}: "
            f"{missing}. PAPER_TABLE_ROWS and the roster file have diverged.")
    return frozenset(mapping[n] for n in PAPER_TABLE_ROWS)


def is_paper_model(ssl, path=None):
    return ssl in paper_models(path)


def non_paper_models(available, path=None):
    """The slugs in ``available`` that the paper's results table does not report."""
    return sorted(set(available) - paper_models(path))
