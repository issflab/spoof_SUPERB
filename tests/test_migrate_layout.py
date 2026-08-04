"""
test_migrate_layout.py
----------------------
Contracts for the v2 -> v3 score-tree migration.

  M1  Only the non-SSL systems move. linear_head files are not touched, because
      v3 does not change where they live.
  M2  Destinations come from score_path(layout="v3"), so the two non-SSL systems
      land side by side in one dataset directory with distinct filenames --
      the defect that motivated v3.
  M3  Dry run is the default and writes nothing. A score file can be 15+ minutes
      of scoring; the tool must not move one because a flag was forgotten.
  M4  Copy is verified by digest before any source is removed, and sources
      survive unless --delete-source is given.
  M5  A destination that exists and DIFFERS aborts rather than overwriting,
      unless --force. A destination already holding a byte-identical copy is
      not a clash: it is the normal state after a copy-only pass, and the
      recommended workflow is copy, inspect, then delete sources later.
  M6  Migration is idempotent: a second run finds nothing to do.
  M7  An unrecognised dataset directory raises instead of being guessed at.

Run:  pytest tests/test_migrate_layout.py
"""

import io
import os

import pytest

from spoof_superb.core.scorepath import score_path
from spoof_superb.tools.migrate_layout import classify, migrate, plan

BODY = {
    ("lfcc_gmm", "famous_figures"): "a - bonafide 1.0\n",
    ("aasist_raw", "famous_figures"): "b - spoof -1.0\n",
    ("lfcc_gmm", "mlaad_v10"): "c - spoof -2.0\n",
}
HEAD = ("linear_head", "in_the_wild", "xls_r_300m.txt", "d - bonafide 3.0\n")


@pytest.fixture
def tree(tmp_path):
    for (system, dsdir), body in BODY.items():
        d = tmp_path / "raw" / system / dsdir
        d.mkdir(parents=True)
        (d / "none.txt").write_text(body)
    system, dsdir, name, body = HEAD
    d = tmp_path / "raw" / system / dsdir
    d.mkdir(parents=True)
    (d / name).write_text(body)
    return tmp_path


def run(root, **kw):
    buf = io.StringIO()
    rc = migrate(str(root), out=buf, **kw)
    return rc, buf.getvalue()


def test_m1_linear_head_is_not_touched(tree):
    system, dsdir, name, body = HEAD
    kept = tree / "raw" / system / dsdir / name
    # Compare path components, not substrings: pytest's tmp_path embeds the
    # test's own name, which contains "linear_head".
    moved_systems = {os.path.relpath(src, tree / "raw").split(os.sep)[0]
                     for src, _, _ in plan(str(tree))}
    assert system not in moved_systems
    assert moved_systems == {"lfcc_gmm", "aasist_raw"}
    run(tree, apply=True, delete_source=True)
    assert kept.read_text() == body


def test_m2_destinations_match_score_path_and_are_distinct(tree):
    run(tree, apply=True, delete_source=True)
    ff = tree / "raw" / "non_ssl" / "famous_figures"
    assert sorted(p.name for p in ff.iterdir()) == ["aasist_raw.txt", "lfcc_gmm.txt"]
    for system in ("lfcc_gmm", "aasist_raw"):
        expected = score_path(system, "Famous_Figures", scores_root=str(tree),
                              layout="v3")
        assert os.path.isfile(expected)
    assert (ff / "lfcc_gmm.txt").read_text() == BODY[("lfcc_gmm", "famous_figures")]
    assert (ff / "aasist_raw.txt").read_text() == BODY[("aasist_raw", "famous_figures")]


def test_m3_dry_run_is_the_default_and_writes_nothing(tree):
    rc, out = run(tree)
    assert rc == 0 and "dry run" in out
    assert not (tree / "raw" / "non_ssl").exists()
    assert (tree / "raw" / "lfcc_gmm" / "famous_figures" / "none.txt").exists()


def test_m4_sources_survive_without_delete_source(tree):
    rc, _ = run(tree, apply=True)
    assert rc == 0
    src = tree / "raw" / "lfcc_gmm" / "famous_figures" / "none.txt"
    dst = tree / "raw" / "non_ssl" / "famous_figures" / "lfcc_gmm.txt"
    assert src.exists() and dst.exists()
    assert src.read_text() == dst.read_text()


def test_m5_identical_destination_is_not_a_clash(tree):
    """The two-phase workflow: copy first, delete sources on a later run."""
    rc, _ = run(tree, apply=True)                     # phase 1: copy only
    assert rc == 0
    assert {s for *_, s in classify(str(tree))} == {"copied"}

    rc, out = run(tree, apply=True, delete_source=True)   # phase 2: delete
    assert rc == 0, out
    assert "removed 3 source(s)" in out
    assert not (tree / "raw" / "lfcc_gmm").exists()
    dst = tree / "raw" / "non_ssl" / "famous_figures" / "lfcc_gmm.txt"
    assert dst.read_text() == BODY[("lfcc_gmm", "famous_figures")]


def test_m5_differing_destination_aborts_unless_forced(tree):
    run(tree, apply=True, delete_source=True)
    # a second, different v2 file arrives at an already-migrated destination
    src = tree / "raw" / "lfcc_gmm" / "famous_figures"
    src.mkdir(parents=True)
    (src / "none.txt").write_text("DIFFERENT\n")
    dst = tree / "raw" / "non_ssl" / "famous_figures" / "lfcc_gmm.txt"

    rc, out = run(tree, apply=True)
    assert rc == 2 and "ABORT" in out
    assert dst.read_text() == BODY[("lfcc_gmm", "famous_figures")], "clobbered"

    rc, _ = run(tree, apply=True, force=True)
    assert rc == 0 and dst.read_text() == "DIFFERENT\n"


def test_m6_migration_is_idempotent(tree):
    run(tree, apply=True, delete_source=True)
    rc, out = run(tree, apply=True, delete_source=True)
    assert rc == 0 and "nothing to do" in out


def test_m7_unknown_dataset_directory_raises(tree):
    (tree / "raw" / "lfcc_gmm" / "not_a_dataset").mkdir(parents=True)
    with pytest.raises(KeyError, match="not a known dataset"):
        plan(str(tree))
