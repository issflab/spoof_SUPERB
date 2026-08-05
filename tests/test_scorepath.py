"""
test_scorepath.py
-----------------
Contracts for core.scorepath, the single decision about where a score file
lives. There is one layout now; the `legacy` and `v2` conventions were retired
on 2026-08-05, so the tests that pinned them are gone with them.

  P0  `linear_head` is raw/linear_head/{dataset}/{frontend}.txt; the non-SSL
      systems are raw/non_ssl/{dataset}/{system}.txt. Two non-SSL systems on one
      dataset therefore have two DISTINCT filenames -- the defect v3 existed to
      fix, since the previous convention gave both `none.txt` and left every
      baseline score file on disk indistinguishable once out of its directory.
  P2  Dataset directory names are canonical and version-bearing. `Multilingual`
      lands in `mlaad_v10`, because "which MLAAD is this?" is the question that
      put a wrong column in an earlier draft.
  P3  No path component needs parsing to recover. Splitting on the separator
      returns the exact inputs -- including frontends whose names contain
      underscores, which is what defeated the old flat filenames.
  P4  There is no condition level. Every ASVLD condition for one (system,
      frontend) resolves to ONE file, because the condition is carried by the
      utt_id and the five condition protocols are disjoint.
  P6  An unknown dataset raises, rather than silently inventing a path that
      would scatter score files somewhere nobody looks.
  P7  No upstream is invented for the non-SSL systems: the frontend argument
      cannot reach their path at all, whatever it is set to.
  P8  A benchmark column reads its own registry key, except DFEval24 -- see
      `column_key`, and `test_paper_models.py::test_d5` for why that one differs.

Run:  pytest tests/test_scorepath.py
"""

import os

import pytest

from spoof_superb.core.scorepath import (
    COLUMN_KEYS,
    DATASET_DIRS,
    NON_SSL_SYSTEMS,
    canonical_dataset,
    column_key,
    score_path,
)

ROOT = "/tmp/scores"


def path(system, dataset, frontend="none"):
    return score_path(system, dataset, frontend, scores_root=ROOT)


def test_p0_linear_head_shape():
    p = path("linear_head", "wild", "xls_r_300m")
    assert p == os.path.join(ROOT, "raw", "linear_head", "in_the_wild",
                             "xls_r_300m.txt")
    rel = os.path.relpath(p, ROOT).split(os.sep)
    assert len(rel) == 4, f"expected raw/system/dataset/frontend.txt, got {rel}"


def test_p0_non_ssl_filenames_are_distinct():
    """The defect this layout exists to fix: both used to be `none.txt`."""
    names = {s: os.path.basename(path(s, "Famous_Figures"))
             for s in NON_SSL_SYSTEMS}
    assert names == {"lfcc_gmm": "lfcc_gmm.txt", "aasist_raw": "aasist_raw.txt"}
    assert len(set(names.values())) == len(NON_SSL_SYSTEMS)


def test_p0_non_ssl_share_one_directory_per_dataset():
    dirs = {os.path.dirname(path(s, "Multilingual")) for s in NON_SSL_SYSTEMS}
    assert len(dirs) == 1
    assert dirs.pop() == os.path.join(ROOT, "raw", "non_ssl", "mlaad_v10")


def test_p2_dataset_names_are_canonical_and_versioned():
    assert canonical_dataset("Multilingual") == "mlaad_v10"
    assert canonical_dataset("eval_2019") == "asvspoof2019_la_eval"
    assert canonical_dataset("wild") == "in_the_wild"
    for key, name in DATASET_DIRS.items():
        assert name == name.lower(), f"{key} -> {name} is not lowercase"
        assert " " not in name and "-" not in name, f"{key} -> {name}"


@pytest.mark.parametrize("frontend", [
    "xls_r_300m",
    "wavlm_large",
    "multires_hubert_multilingual_large600k",   # underscores everywhere
    "audio_albert_960hr",
    "none",
])
def test_p3_inputs_are_recoverable_without_parsing(frontend):
    """The old flat names could not do this: model names contain underscores."""
    rel = os.path.relpath(path("linear_head", "Multilingual", frontend),
                          ROOT).split(os.sep)
    assert rel[0] == "raw"
    assert rel[1] == "linear_head"
    assert rel[2] == "mlaad_v10"
    assert os.path.splitext(rel[3])[0] == frontend


@pytest.mark.parametrize("dataset", sorted(DATASET_DIRS))
def test_p3_every_dataset_resolves_without_parsing(dataset):
    """Every real dataset, both non-SSL systems: four components, nothing parsed."""
    for system in NON_SSL_SYSTEMS:
        rel = os.path.relpath(path(system, dataset), ROOT).split(os.sep)
        assert len(rel) == 4, rel
        assert rel[:2] == ["raw", "non_ssl"]
        assert rel[2] == canonical_dataset(dataset)
        assert os.path.splitext(rel[3])[0] == system


def test_p4_no_condition_level_for_asvld():
    """All ASVLD conditions pool into one file per (system, frontend)."""
    p = path("linear_head", "asvspoofLD", "tera")
    assert p == os.path.join(ROOT, "raw", "linear_head", "asvspoof_ld",
                             "tera.txt")
    assert "Noise_Addition" not in p and "condition" not in p


def test_p6_unknown_dataset_raises():
    with pytest.raises(KeyError):
        path("linear_head", "not_a_dataset", "x")


def test_p7_no_upstream_is_invented_for_the_non_ssl_systems():
    """The frontend argument cannot leak into their path, whatever it is."""
    for bogus in ("none", "xls_r_300m", "", "wavlm_large"):
        assert path("lfcc_gmm", "wild", bogus) == os.path.join(
            ROOT, "raw", "non_ssl", "in_the_wild", "lfcc_gmm.txt"), bogus


def test_p8_only_dfeval_reads_a_different_key():
    assert column_key("deepfake_eval_2024") == "deepfake_eval_2024_segmented"
    assert set(COLUMN_KEYS) == {"deepfake_eval_2024"}, (
        "a second remapped column needs its own justification -- see the note "
        "on COLUMN_KEYS")
    for other in ("wild", "Multilingual", "asvspoofLD", "spoofceleb"):
        assert column_key(other) == other


def test_the_retired_layouts_are_really_gone():
    """`--layout` was on thirteen commands; none of it should be reachable."""
    import spoof_superb.core.scorepath as sp

    for gone in ("LAYOUTS", "layout_key", "DATASET_KEY_BY_LAYOUT",
                 "_LEGACY_LINEAR_HEAD"):
        assert not hasattr(sp, gone), f"{gone} is back"

    import inspect
    assert "layout" not in inspect.signature(score_path).parameters


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
