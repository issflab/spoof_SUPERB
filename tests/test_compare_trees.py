"""
test_compare_trees.py
---------------------
Contracts for tools.compare_trees, which decides whether a freshly built score
tree reproduces a published one. Its verdicts are what a provenance claim rests
on, so each one has to mean exactly what it says.

  C0  The verdict distinguishes WHY two cells differ, not just how much:
      different trials (coverage) and different scores are separate findings,
      and only the second can indict the pipeline.
  C1  "reproduces" is graded on the EER over the SHARED trials. Scores may move
      substantially without moving the EER -- EER depends on the ordering of the
      two class distributions, not the values -- and the reported metric is what
      a reproduction claim is about.
  C2  A shared ABSOLUTE prefix is stripped, because one tree wrote its mount
      point into every utt_id. Relative ids are never touched, so a genuine
      coverage difference cannot be normalised into a false match.
  C3  Component rewrites are opt-in and never inferred. Famous Figures names the
      bonafide directory `-` in one tree and `Bonafide` in the other; asserting
      that those are the same utterances is the caller's claim to make.
  C4  Label disagreement outranks every other verdict. If the two trees disagree
      about ground truth, no EER comparison between them means anything.

Run:  pytest tests/test_compare_trees.py
"""

import numpy as np
import pytest

from spoof_superb.tools.compare_trees import (
    REPRODUCE_TOL_PP, rewrite_components, strip_absolute_prefix, verdict,
)


def cell(**kw):
    """A comparison row with the neutral defaults an 'identical' cell has."""
    base = dict(n_common=10, n_only_a=0, n_only_b=0, label_mismatch=0,
                frac_exact=1.0, eer_a_common=5.0, eer_b_common=5.0)
    base.update(kw)
    return base


# --- C0/C1: what the verdicts mean -----------------------------------------

def test_c0_same_trials_and_bit_identical_scores_is_identical():
    assert verdict(cell()) == "identical"


def test_c0_bit_identical_scores_with_extra_trials_is_coverage_only():
    assert verdict(cell(n_only_b=5)) == "coverage only"


def test_c1_moved_scores_that_hold_the_eer_reproduce():
    # The load-bearing case: scores differ everywhere, the metric does not.
    assert verdict(cell(frac_exact=0.0, eer_b_common=5.0 + REPRODUCE_TOL_PP / 2)) \
        == "reproduces"


def test_c1_moved_scores_that_move_the_eer_do_not_reproduce():
    assert verdict(cell(frac_exact=0.0, eer_b_common=5.0 + REPRODUCE_TOL_PP * 10)) \
        == "SCORES DIFFER"


def test_c0_coverage_and_score_differences_are_reported_separately():
    assert verdict(cell(frac_exact=0.0, n_only_b=5,
                        eer_b_common=5.0 + REPRODUCE_TOL_PP * 10)) \
        == "SCORES DIFFER + coverage"
    assert verdict(cell(frac_exact=0.0, n_only_b=5)) == "reproduces + coverage"


def test_c0_no_shared_trials_is_not_a_score_finding():
    # DFEval24: one tree scores whole recordings, the other every 4 s window.
    # Nothing about the pipeline follows from that.
    assert verdict(cell(n_common=0)) == "no overlap"


def test_c1_a_single_class_intersection_is_not_comparable():
    assert verdict(cell(frac_exact=0.0, eer_a_common=None)) == "not comparable"


# --- C4: labels outrank everything -----------------------------------------

def test_c4_label_disagreement_outranks_agreeing_eers():
    assert verdict(cell(label_mismatch=1)) == "LABELS DIFFER"


# --- C2: absolute-prefix normalisation --------------------------------------

def test_c2_shared_absolute_prefix_is_stripped():
    utts = np.asarray(["/mnt/corpus/A/x.wav", "/mnt/corpus/B/y.wav"], dtype=object)
    got, prefix = strip_absolute_prefix(utts)
    assert list(got) == ["A/x.wav", "B/y.wav"]
    assert prefix == "/mnt/corpus/"


def test_c2_relative_ids_are_never_touched():
    utts = np.asarray(["A/x.wav", "A/y.wav"], dtype=object)
    got, prefix = strip_absolute_prefix(utts)
    assert list(got) == ["A/x.wav", "A/y.wav"] and prefix == ""


def test_c2_only_whole_directory_components_are_removed():
    # The common string prefix is "/mnt/corpus/spk_1", but cutting mid-component
    # would corrupt both ids. Only up to the last separator may go.
    utts = np.asarray(["/mnt/corpus/spk_1/x.wav", "/mnt/corpus/spk_12/y.wav"],
                      dtype=object)
    got, prefix = strip_absolute_prefix(utts)
    assert prefix == "/mnt/corpus/"
    assert list(got) == ["spk_1/x.wav", "spk_12/y.wav"]


def test_c2_empty_input_is_handled():
    got, prefix = strip_absolute_prefix(np.asarray([], dtype=object))
    assert len(got) == 0 and prefix == ""


# --- C3: component rewrites are asserted, not inferred ----------------------

def test_c3_no_rules_means_no_change():
    utts = np.asarray(["A/-/x.wav"], dtype=object)
    assert list(rewrite_components(utts, {})) == ["A/-/x.wav"]


def test_c3_rewrites_whole_components_only():
    utts = np.asarray(["A/-/x-y.wav", "A/COZY/z.wav"], dtype=object)
    got = rewrite_components(utts, {"-": "Bonafide"})
    # The '-' directory is renamed; the '-' inside the filename is not.
    assert list(got) == ["A/Bonafide/x-y.wav", "A/COZY/z.wav"]
