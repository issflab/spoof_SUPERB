"""
test_scorepath.py
-----------------
Contracts for core.scorepath, the single decision about where a score file
lives.

  P1  v2 is raw/{system}/{dataset}/{frontend}.txt -- four levels, no more.
  P2  Dataset directory names are canonical and version-bearing. `Multilingual`
      lands in `mlaad_v10`, because "which MLAAD is this?" is the question that
      put a wrong column in an earlier draft.
  P3  No path component needs parsing to recover. Given a v2 path, splitting on
      the separator returns the exact inputs -- including frontends whose names
      contain underscores, which is what defeated the old flat filenames.
  P4  There is no condition level. Every ASVLD condition for one (system,
      frontend) resolves to ONE file, because the condition is carried by the
      utt_id and the five condition protocols are disjoint.
  P5  legacy reproduces the pre-reorg write paths exactly, so an existing tree
      keeps working during a migration.
  P6  An unknown dataset or layout raises, rather than silently inventing a
      path that would scatter score files somewhere nobody looks.
  P7  The baselines record 'none' as their frontend -- they take no upstream,
      and a path claiming one would be false.

Run:  pytest tests/test_scorepath.py
"""

import os

import pytest

from spoof_superb.core.scorepath import (
    DATASET_DIRS,
    LAYOUTS,
    canonical_dataset,
    score_path,
)

ROOT = "/tmp/scores"


def v2(system, dataset, frontend):
    return score_path(system, dataset, frontend, scores_root=ROOT, layout="v2")


def test_p1_v2_shape():
    p = v2("linear_head", "wild", "xls_r_300m")
    assert p == os.path.join(ROOT, "raw", "linear_head", "in_the_wild",
                             "xls_r_300m.txt")
    rel = os.path.relpath(p, ROOT).split(os.sep)
    assert len(rel) == 4, f"expected raw/system/dataset/frontend.txt, got {rel}"


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
    p = v2("linear_head", "Multilingual", frontend)
    rel = os.path.relpath(p, ROOT).split(os.sep)
    assert rel[0] == "raw"
    assert rel[1] == "linear_head"
    assert rel[2] == "mlaad_v10"
    assert os.path.splitext(rel[3])[0] == frontend


def test_p4_no_condition_level_for_asvld():
    """All ASVLD conditions pool into one file per (system, frontend)."""
    p = v2("linear_head", "asvspoofLD", "tera")
    assert p == os.path.join(ROOT, "raw", "linear_head", "asvspoof_ld", "tera.txt")
    assert "Noise_Addition" not in p and "condition" not in p


def test_p5_legacy_reproduces_the_old_write_paths():
    cases = {
        ("linear_head", "Multilingual", "xls_r_300m"):
            "linear_head_MLAAD_v10/linear_head_MLAAD_v10_xls_r_300m.txt",
        ("linear_head", "MAILABS", "apc"):
            "linear_head_MLAAD_v10/mailabs/linear_head_MAILABS_apc.txt",
        ("linear_head", "spoofceleb", "wavlm_large"):
            "linear_head_SpoofCeleb/linear_head_SpoofCeleb_wavlm_large.txt",
        ("lfcc_gmm", "wild", "none"):
            "baselines/lfcc_gmm/lfcc_gmm_wild.txt",
        ("aasist_raw", "eval_2019", "none"):
            "baselines/aasist_raw/aasist_raw_eval_2019.txt",
    }
    for (system, dataset, frontend), expected in cases.items():
        got = score_path(system, dataset, frontend, scores_root=ROOT, layout="legacy")
        assert got == os.path.join(ROOT, expected), f"{system}/{dataset}"


def test_p6_unknown_dataset_and_layout_raise():
    with pytest.raises(KeyError):
        v2("linear_head", "not_a_dataset", "x")
    with pytest.raises(ValueError):
        score_path("linear_head", "wild", "x", scores_root=ROOT, layout="v3")
    # legacy has no convention for linear_head on the reference-driven columns
    with pytest.raises(KeyError):
        score_path("linear_head", "wild", "x", scores_root=ROOT, layout="legacy")


def test_p7_baselines_declare_no_frontend():
    p = v2("lfcc_gmm", "wild", "none")
    assert p.endswith(os.path.join("lfcc_gmm", "in_the_wild", "none.txt"))


def test_layouts_are_the_documented_set():
    assert set(LAYOUTS) == {"v2", "legacy"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
