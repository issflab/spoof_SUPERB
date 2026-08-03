"""
test_verification_levels.py
---------------------------
Contracts for the two-level verification step. The verdict IS the product here:
a reproducer reads one word and decides whether they have a problem, so each
word has to mean exactly one thing and the boundaries between them have to be
where the reasoning says they are.

Level 1 -- score files
  L1  Bit-identical output is IDENTICAL and says so without needing a metric.
  L2  Scores that all moved but held the EER are EQUIVALENT. This is the
      TARGET outcome on different hardware, not a grudging pass -- a check that
      demanded bit-exactness would fail every honest reproduction.
  L3  Rank-preserved scores that still moved the EER are SENSITIVE, not a
      failure. Measured case: r = 0.99999 on max |d| = 0.043 moving the EER
      4.15 pp, because a near-chance model sits where the DET curve is flat.
  L4  Scores that disagree in RANK and move the EER are SCORES_DIFFER -- the
      only score verdict that indicts the pipeline.
  L5  Label disagreement outranks every score statistic. If the two runs
      disagree about ground truth, no comparison between them means anything.
  L6  Different trial sets are COVERAGE_DIFFERS, never a score finding.
  L7  Manifest mode proves trial-set identity by DIGEST, not by row count --
      two different 71,237-trial sets are still different trial sets.
  L8  Manifest mode cannot claim SENSITIVE, because rank agreement is not in
      the manifest. It must say so instead of guessing.

Level 2 -- analysis tables
  L9  Cells drifting within tolerance while every claim survives is EQUIVALENT.
  L10 A changed best-in-column is CONCLUSIONS_DIFFER even when every cell moved
      less than a claim-preserving run did. Magnitude is not the grade.
  L11 A changed column ordering is CONCLUSIONS_DIFFER: those orderings ARE the
      sentences in 4.4.2 and 4.4.3.
  L12 A missing model is STRUCTURE_DIFFERS, not a small number.
  L13 The paper's own `*` emphasis markers are compared directly, so the
      published bolding is checked as published.
  L14 Degradation sign flips against the Baseline are caught even when the
      cells barely moved -- "this condition hurts" is the claim.

Run:  pytest tests/test_verification_levels.py
"""

import numpy as np
import pytest

from spoof_superb.verification.analysis import compare_table, load_table
from spoof_superb.verification.scores import (grade_manifest_cell,
                                              grade_tree_cell, utt_digest)
from spoof_superb.verification.verdicts import (EER_TOL_PP, IS_FAILURE,
                                                RANK_TOL, SCORE_LADDER, rank,
                                                worst)


# ===========================================================================
# Level 1
# ===========================================================================

def tree_cell(**kw):
    """A measured cell with the neutral defaults a perfect reproduction has."""
    base = dict(status="ok", n_a=100, n_b=100, n_common=100, n_only_a=0,
                n_only_b=0, nan_a=0, nan_b=0, label_mismatch=0, frac_exact=1.0,
                spearman=1.0, corr=1.0, eer_a_common=5.0, eer_b_common=5.0)
    base.update(kw)
    return base


def test_l1_bit_identical_output_is_identical():
    v, why = grade_tree_cell(tree_cell())
    assert v == "IDENTICAL"
    assert "bit-identical" in why


def test_l2_moved_scores_that_hold_the_eer_are_equivalent():
    """The target outcome on different hardware, not a grudging pass."""
    v, _ = grade_tree_cell(tree_cell(frac_exact=0.0,
                                     eer_b_common=5.0 + EER_TOL_PP / 2))
    assert v == "EQUIVALENT"
    assert v not in IS_FAILURE


def test_l3_rank_preserved_scores_that_move_the_eer_are_sensitive():
    """The measured flat-DET case: r=0.99999, max|d|=0.043, EER moves 4.15 pp."""
    v, why = grade_tree_cell(tree_cell(frac_exact=0.0, spearman=0.99999,
                                       eer_b_common=5.0 + 4.15))
    assert v == "SENSITIVE"
    assert v not in IS_FAILURE, (
        "a flat operating point is a caveat on the metric, not a defect in the "
        "run; failing it trains people to ignore the check"
    )
    assert "flat" in why


def test_l4_rank_disagreement_plus_a_moved_eer_indicts_the_pipeline():
    v, _ = grade_tree_cell(tree_cell(frac_exact=0.0, spearman=RANK_TOL - 0.05,
                                     eer_b_common=5.0 + 4.15))
    assert v == "SCORES_DIFFER"
    assert v in IS_FAILURE


def test_l5_label_disagreement_outranks_agreeing_eers():
    """Identical EERs must not launder a protocol disagreement into a pass."""
    v, _ = grade_tree_cell(tree_cell(label_mismatch=3))
    assert v == "LABELS_DIFFER"
    assert rank("LABELS_DIFFER") > rank("SCORES_DIFFER"), (
        "comparing scores across disagreeing ground truth answers the wrong "
        "question confidently"
    )


def test_l6_different_trial_sets_are_never_a_score_finding():
    for kw in ({"n_only_a": 7}, {"n_only_b": 7}, {"n_common": 0}):
        v, _ = grade_tree_cell(tree_cell(frac_exact=0.9, **kw))
        assert v == "COVERAGE_DIFFERS", kw


def test_l6b_candidate_nan_beats_every_agreement_statistic():
    v, _ = grade_tree_cell(tree_cell(nan_b=50, frac_exact=0.0))
    assert v == "CANDIDATE_INVALID"


def test_worst_picks_the_most_severe_verdict():
    assert worst(["IDENTICAL", "SENSITIVE", "LABELS_DIFFER"]) == "LABELS_DIFFER"
    assert worst([]) is None
    assert worst(["EQUIVALENT", "IDENTICAL"]) == "EQUIVALENT"


# --- manifest mode ---------------------------------------------------------

def manifest_pair(ref_over=None, cand_over=None):
    ref = {"n_rows": 100, "n_nonfinite": 0, "eer_percent": 5.0,
           "utt_digest": "abc", "sha256": ["deadbeef"]}
    cand = dict(ref)
    ref.update(ref_over or {})
    cand.update(cand_over or {})
    return ref, cand


def test_l7_same_row_count_with_a_different_trial_set_is_coverage():
    """Row counts prove nothing; the digest is what proves the trials match."""
    ref, cand = manifest_pair(cand_over={"utt_digest": "zzz",
                                         "sha256": ["cafe"]})
    assert ref["n_rows"] == cand["n_rows"]
    v, why = grade_manifest_cell(ref, cand)
    assert v == "COVERAGE_DIFFERS"
    assert "trial set" in why


def test_l7b_utt_digest_ignores_row_order():
    """Two runs that scored the same utterances in a different order match."""
    a = np.asarray(["u3", "u1", "u2"], dtype=object)
    b = np.asarray(["u1", "u2", "u3"], dtype=object)
    assert utt_digest(a) == utt_digest(b)
    assert utt_digest(a) != utt_digest(np.asarray(["u1", "u2"], dtype=object))


def test_l8_manifest_mode_never_claims_sensitive():
    """It has no rank information, so it must name the limit, not guess."""
    ref, cand = manifest_pair(cand_over={"eer_percent": 9.15,
                                         "sha256": ["cafe"]})
    v, why = grade_manifest_cell(ref, cand)
    assert v == "SCORES_DIFFER"
    assert "--ref-root" in why, "the report must name the command that decides"
    assert "SENSITIVE" in why


def test_l8b_identical_sha256_short_circuits():
    ref, cand = manifest_pair()
    v, why = grade_manifest_cell(ref, cand)
    assert v == "IDENTICAL" and "byte-identical" in why


def test_missing_candidate_is_a_failure_not_a_skip():
    v, _ = grade_manifest_cell({"n_rows": 1}, None)
    assert v == "MISSING" and v in IS_FAILURE


# ===========================================================================
# Level 2
# ===========================================================================

REF_CSV = """Model,ColA,ColB,Mean
alpha,10.000,20.000,15.000
beta,12.000,22.000,17.000
gamma,14.000,24.000,19.000
delta,16.000,26.000,21.000
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return load_table(p)


def test_l9_drifting_cells_with_intact_claims_are_equivalent(tmp_path):
    ref = write(tmp_path, "ref.csv", REF_CSV)
    cand = write(tmp_path, "cand.csv", REF_CSV.replace("10.000", "10.030")
                 .replace("22.000", "22.040"))
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert v == "EQUIVALENT"
    assert rep["cells"]["n_over_tol"] > 0, "the fixture must actually drift"
    assert not rep["broken_claims"]


def test_l10_a_changed_best_in_column_fails_however_small_the_move(tmp_path):
    """Magnitude is not the grade. 0.5 pp that reorders beats 3 pp that does not."""
    ref = write(tmp_path, "ref.csv", REF_CSV)
    # alpha 10.000 -> 12.500 puts beta (12.000) first in ColA.
    cand = write(tmp_path, "cand.csv", REF_CSV.replace("alpha,10.000",
                                                       "alpha,12.500"))
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert v == "CONCLUSIONS_DIFFER"
    assert any("best on ColA" in b for b in rep["broken_claims"])
    assert v in IS_FAILURE


def test_l11_a_changed_column_ordering_is_a_changed_sentence(tmp_path):
    """'which condition hurts most' is the finding, not a presentation detail."""
    ref = write(tmp_path, "ref.csv", REF_CSV)
    swapped = "\n".join(
        [REF_CSV.splitlines()[0]]
        + [",".join([r.split(",")[0], r.split(",")[2], r.split(",")[1],
                     r.split(",")[3]]) for r in REF_CSV.splitlines()[1:]])
    cand = write(tmp_path, "cand.csv", swapped + "\n")
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert v == "CONCLUSIONS_DIFFER"
    assert any("ordering of columns" in b for b in rep["broken_claims"])


def test_l12_a_missing_model_is_structural_not_numeric(tmp_path):
    ref = write(tmp_path, "ref.csv", REF_CSV)
    cand = write(tmp_path, "cand.csv",
                 "\n".join(l for l in REF_CSV.splitlines()
                           if not l.startswith("delta")) + "\n")
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert v == "STRUCTURE_DIFFERS"
    assert rep["rows_reference_only"] == ["delta"]


def test_l12b_a_withheld_cell_on_one_side_is_structural(tmp_path):
    """'TODO' against a number is an absent measurement, not a delta of zero."""
    ref = write(tmp_path, "ref.csv", REF_CSV)
    cand = write(tmp_path, "cand.csv", REF_CSV.replace("beta,12.000",
                                                       "beta,TODO"))
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert v == "STRUCTURE_DIFFERS"
    assert rep["cells"]["n_one_sided"] == 1


def test_l13_emphasis_markers_are_compared_as_published(tmp_path):
    """The `*` encodes the caption's bolding rule -- check it, don't restate it."""
    marked = REF_CSV.replace("alpha,10.000", "alpha,10.000*")
    ref = write(tmp_path, "ref.csv", marked)
    cand = write(tmp_path, "cand.csv", REF_CSV.replace("beta,12.000",
                                                       "beta,12.000*"))
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert rep["claims"]["emphasis_markers"]["same"] is False
    assert any("emphasis marker" in b for b in rep["broken_claims"])
    assert v == "CONCLUSIONS_DIFFER"

    # And the marker must not have been parsed into the number.
    assert ref.values[("alpha", "ColA")] == pytest.approx(10.0)


DEG_CSV = """Model,Baseline,Codec
alpha,10.000,10.100
beta,12.000,12.100
gamma,14.000,14.100
"""


def test_l14_a_degradation_sign_flip_is_caught_though_the_cells_barely_moved(tmp_path):
    """'this condition hurts' reversing is a finding, not a rounding artefact."""
    ref = write(tmp_path, "ref.csv", DEG_CSV)
    cand = write(tmp_path, "cand.csv", DEG_CSV.replace("alpha,10.000,10.100",
                                                       "alpha,10.000,9.900"))
    v, rep = compare_table(ref, cand, reference_col="Baseline")
    assert rep["claims"]["degradation_sign_flips"] == [
        {"model": "alpha", "column": "Codec"}]
    assert v == "CONCLUSIONS_DIFFER"
    assert rep["cells"]["max_delta_pp"] < 0.3, (
        "the fixture must stay small, or it proves nothing about magnitude"
    )


def test_identical_tables_are_identical(tmp_path):
    ref = write(tmp_path, "ref.csv", REF_CSV)
    cand = write(tmp_path, "cand.csv", REF_CSV)
    v, rep = compare_table(ref, cand, mean_col="Mean")
    assert v == "IDENTICAL"
    assert rep["cells"]["max_delta_pp"] == 0.0


def test_excluded_rows_do_not_compete_for_best_in_column(tmp_path):
    """The caption bolds the best SSL MODEL; the non-SSL rows are the baseline."""
    csv = REF_CSV + "LFCC-GMM,1.000,1.000,1.000\n"
    ref = write(tmp_path, "ref.csv", csv)
    cand = write(tmp_path, "cand.csv", csv)
    _v, rep = compare_table(ref, cand, exclude=("LFCC-GMM",), mean_col="Mean")
    assert rep["claims"]["best_per_column"]["ColA"]["reference"] == "alpha"


def test_the_ladders_are_ordered_worst_last():
    """`rank` is an index, so the order of the ladder is the contract."""
    assert rank("IDENTICAL") < rank("EQUIVALENT") < rank("SENSITIVE")
    assert rank("SENSITIVE") < rank("SCORES_DIFFER") < rank("LABELS_DIFFER")
    assert SCORE_LADDER[0] == "IDENTICAL"
    assert SCORE_LADDER[-1] == "ERROR"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
