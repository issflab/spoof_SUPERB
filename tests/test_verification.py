"""
test_verification.py
--------------------
Contracts for the verify_mlaad.py / verify_spoofceleb.py split into shared
statistics plus separate grade policies.

The whole point of the split is that the two policies are NOT the same, so the
tests assert the difference rather than the sameness:

  V1  Both policies read the 4-column format by peeling from the RIGHT, so
      utt_ids containing spaces compare correctly.
  V2  Identical inputs produce identical statistics regardless of policy --
      only the verdict is policy-dependent.
  V3  SpoofCeleb passes on rank agreement alone. A constant offset destroys
      Pearson but not Spearman, and must still PASS: EER is rank-based.
  V4  MLAAD FAILS that same input, because it additionally requires Pearson.
      This is the difference that would have been erased by merging them.
  V5  Any NaN in our own output fails SpoofCeleb; MLAAD tolerates up to 1%.
  V6  A reference that is >50% NaN is REF_UNUSABLE, not a failure -- the
      reference is the broken side and must not fail our run (rc 0).
  V7  No shared utt_ids is a failure, not a silent pass.

Run:  pytest tests/test_verification.py
"""

import pytest

from spoof_superb.verification.policies import FAIL, PASS, REF_UNUSABLE, grade
from spoof_superb.verification.stats import compare, load_scores


def _write(path, rows):
    path.write_text("".join(f"{u} - {k} {s}\n" for u, k, s in rows))
    return str(path)


def _pair(tmp_path, new_rows, ref_rows):
    return compare(_write(tmp_path / "new.txt", new_rows),
                   _write(tmp_path / "ref.txt", ref_rows))


def test_v1_utt_ids_with_spaces_are_parsed(tmp_path):
    spacey = "MLAAD/fake/en/OpenAI TTS-1 HD/utt 1.wav"
    path = _write(tmp_path / "s.txt", [(spacey, "spoof", -1.5)])
    assert load_scores(path) == {spacey: -1.5}


def test_v2_statistics_are_policy_independent(tmp_path):
    rows_new = [(f"u{i}", "spoof", i * 0.5) for i in range(20)]
    rows_ref = [(f"u{i}", "spoof", i * 0.5) for i in range(20)]
    c = _pair(tmp_path, rows_new, rows_ref)
    assert c.n_shared == 20 and c.n_both_finite == 20
    assert c.pearson == pytest.approx(1.0)
    assert c.spearman == pytest.approx(1.0)
    assert grade("spoofceleb", c).status == PASS
    assert grade("mlaad", c).status == PASS


def test_v3_v4_rank_preserving_but_value_distorting_input_splits_the_policies(tmp_path):
    """A monotone but strongly non-linear remap: Spearman 1.0, Pearson well below 0.99."""
    ref = [(f"u{i}", "spoof", float(i)) for i in range(1, 40)]
    new = [(f"u{i}", "spoof", float(i) ** 4) for i in range(1, 40)]
    c = _pair(tmp_path, new, ref)

    assert c.spearman == pytest.approx(1.0), "rank order was not preserved by the fixture"
    assert c.pearson < 0.99, "fixture does not actually stress Pearson"

    assert grade("spoofceleb", c).status == PASS, (
        "SpoofCeleb must pass on rank agreement alone -- EER is rank-based"
    )
    assert grade("mlaad", c).status == FAIL, (
        "MLAAD additionally requires Pearson; merging the two policies would "
        "have silently relaxed this"
    )


def test_v5_nan_tolerance_differs_between_policies(tmp_path):
    # 1 NaN in 200 rows = 0.5%: under MLAAD's 1% tolerance, over SpoofCeleb's 0%.
    ref = [(f"u{i}", "spoof", float(i)) for i in range(200)]
    new = [(f"u{i}", "spoof", float(i)) for i in range(199)] + [("u199", "spoof", "nan")]
    c = _pair(tmp_path, new, ref)
    assert c.new_nan == 1

    assert grade("spoofceleb", c).status == FAIL, "SpoofCeleb tolerates no NaN"
    assert grade("mlaad", c).status == PASS, "MLAAD tolerates up to 1% NaN"


def test_v6_unusable_reference_is_not_our_failure(tmp_path):
    ref = [(f"u{i}", "spoof", "nan") for i in range(80)] + \
          [(f"u{i}", "spoof", float(i)) for i in range(80, 100)]
    new = [(f"u{i}", "spoof", float(i)) for i in range(100)]
    c = _pair(tmp_path, new, ref)

    for policy in ("mlaad", "spoofceleb"):
        v = grade(policy, c)
        assert v.status == REF_UNUSABLE, policy
        assert v.returncode == 0, (
            f"{policy}: a broken reference must not fail our run -- our output is finite"
        )


def test_v7_no_overlap_fails(tmp_path):
    c = _pair(tmp_path, [("a", "spoof", 1.0)], [("b", "spoof", 1.0)])
    assert c.n_shared == 0
    for policy in ("mlaad", "spoofceleb"):
        assert grade(policy, c).returncode == 1, policy


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
