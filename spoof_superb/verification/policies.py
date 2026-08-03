"""SUPERSEDED -- per-dataset grade policies for the legacy migration check.

Replaced by `verification.verdicts`, whose ladder is dataset-independent: it
grades on the EER over identical trials, which is the quantity a reproduction
claim is about, and reports the correlation statistics beside it rather than
thresholding them per corpus. Kept because `tests/test_verification.py` pins
the reasoning below, which is worth not losing.

----------------------------------------------------------------------------
Grade policies: what counts as a pass, per dataset.

These are NOT interchangeable, and collapsing them into one flag-driven check
would erase the reasoning behind each threshold. The differences below are
deliberate and were each arrived at from an observed failure.

Bit-exact reproduction of a reference is not achievable: the references were
produced in a different environment (different librosa/soxr/torch/CUDA), which
introduces a near-constant logit offset (~0.33 for xls_r_300m) with r > 0.99.
That offset is irrelevant to EER, which is rank-based. So every policy asks for
detection-equivalence, not absolute agreement.
"""

from dataclasses import dataclass
from typing import Callable

from spoof_superb.verification.stats import Comparison

__all__ = ["Verdict", "POLICIES", "grade"]

PASS, FAIL, REF_UNUSABLE, NO_OVERLAP = "PASS", "FAIL", "REF_UNUSABLE", "NO_OVERLAP"


@dataclass
class Verdict:
    status: str
    reason: str = ""

    @property
    def returncode(self):
        # REF_UNUSABLE is not our failure: our output is finite and valid, the
        # reference is the broken side, so it must not fail the run.
        return 0 if self.status in (PASS, REF_UNUSABLE) else 1


def _common_guards(c: Comparison, new_nan_tolerance: float) -> Verdict:
    """Checks every policy shares, in the order they must be applied."""
    if c.n_shared == 0:
        return Verdict(NO_OVERLAP, "no shared utt_ids")
    if c.new_nan > new_nan_tolerance * c.n_shared:
        return Verdict(FAIL, f"our output has NaN/inf ({c.new_nan})")
    if c.n_both_finite < 0.5 * c.n_shared:
        # e.g. the spectrogram-upstream references saved with 98% NaN.
        return Verdict(REF_UNUSABLE,
                       f"reference is >50% NaN ({c.ref_nan}); our output is finite")
    return None


def grade_mlaad(c: Comparison, r_min=0.99, sign_min=0.999) -> Verdict:
    """MLAAD / M-AILABS: agreement on value, rank AND sign.

    A 1% NaN tolerance in our own output, unlike SpoofCeleb below.
    """
    guard = _common_guards(c, new_nan_tolerance=0.01)
    if guard:
        return guard
    ok = (c.pearson >= r_min and c.spearman >= r_min and c.sign_agree >= sign_min)
    return Verdict(PASS if ok else FAIL,
                   f"r>={r_min}, spearman>={r_min}, sign@0>={sign_min}")


def grade_spoofceleb(c: Comparison, spearman_min=0.99) -> Verdict:
    """SpoofCeleb: rank correlation alone decides.

    Pearson is reported as a diagnostic only. On the MLAAD run a handful of
    tail outliers dragged Pearson to 0.92 on models whose Spearman was 0.996,
    which would have failed a Pearson-gated check for no detection-relevant
    reason. EER is rank-based, so Spearman is the contract.

    SpoofCeleb audio is natively 16 kHz, so no resampling happens and agreement
    is near-exact; that is why ANY NaN in our output is a failure here, where
    MLAAD tolerates 1%.
    """
    guard = _common_guards(c, new_nan_tolerance=0.0)
    if guard:
        return guard
    return Verdict(PASS if c.spearman >= spearman_min else FAIL,
                   f"spearman>={spearman_min}")


POLICIES: dict = {
    "mlaad": grade_mlaad,
    "mailabs": grade_mlaad,
    "spoofceleb": grade_spoofceleb,
}


def grade(policy: str, c: Comparison) -> Verdict:
    fn: Callable[[Comparison], Verdict] = POLICIES[policy]
    return fn(c)
