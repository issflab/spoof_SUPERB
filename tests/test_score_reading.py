"""
test_score_reading.py
---------------------
Contracts for the three pieces that let the analysis layer read any score tree:
`core.scorefile.read_scored`, `core.scorepath.mlaad_pool_paths`, and
`core.scorepath.available_frontends`.

  R0  read_scored accepts all three on-disk shapes -- canonical 4-column
      space-separated, its 4-column tab-separated twin, and the 3-column
      tab-separated legacy MLAAD file with a header. All three exist in the
      trees this repo must read, and the extension does not identify which:
      the legacy and v3 .tsv files share it and differ in both separator count
      and header.
  R1  Space-separated fields are peeled from the RIGHT. utt_ids legitimately
      contain spaces, and a left split silently returns the wrong utt_id and
      reads "-" as the label for every affected row.
  R2  Several paths concatenate in order, because a benchmark column the paper
      defines as a pool of corpora is assembled that way.
  R3  The MLAAD column resolves to ONE file under legacy and TWO under v2/v3,
      and both describe the same pool. This is the whole of P8: v3 stores MLAAD
      spoof and M-AILABS bonafide separately, and pooling them at read time is
      what makes the column formable.
  R4  available_frontends inverts the layout's own naming rule rather than
      guessing at it, so a frontend whose name contains the separator is
      recovered exactly.
  R5  Enumerating a directory that does not exist returns [], because "nothing
      was scored" is a normal state to report and not an error.

Run:  pytest tests/test_score_reading.py
"""

import os

import pytest

from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import available_frontends, mlaad_pool_paths

ROOT = "/scores"


def write(tmp_path, name, text):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return str(p)


# --- R0/R1: the three shapes, and right-peeling ----------------------------

def test_r0_canonical_space_separated(tmp_path):
    p = write(tmp_path, "a.txt", "u1 - bonafide 0.5\nu2 - spoof -1.25\n")
    utt, lab, sc = read_scored(p)
    assert list(utt) == ["u1", "u2"]
    assert list(lab) == ["bonafide", "spoof"]
    assert list(sc) == [0.5, -1.25]


def test_r1_utt_ids_with_spaces_survive_a_right_split(tmp_path):
    # The real case: MLAAD vendor directories such as "Cartesia.ai (Sonic-3)".
    p = write(tmp_path, "a.txt",
              "MLAAD/fake/en/OpenAI TTS-1 HD/x.wav - spoof -3.0\n")
    utt, lab, sc = read_scored(p)
    assert list(utt) == ["MLAAD/fake/en/OpenAI TTS-1 HD/x.wav"]
    assert list(lab) == ["spoof"]          # not "-"
    assert list(sc) == [-3.0]


def test_r0_four_column_tsv_twin(tmp_path):
    p = write(tmp_path, "a.tsv", "u 1\t-\tspoof\t2.5\n")
    utt, lab, sc = read_scored(p)
    assert list(utt) == ["u 1"] and list(lab) == ["spoof"] and list(sc) == [2.5]


def test_r0_three_column_legacy_tsv_with_header(tmp_path):
    p = write(tmp_path, "a.tsv", "utt_id\tlabel\tscore\nu1\tbonafide\t0.25\n")
    utt, lab, sc = read_scored(p)
    assert list(utt) == ["u1"] and list(lab) == ["bonafide"] and list(sc) == [0.25]


def test_r0_extension_does_not_decide_the_shape(tmp_path):
    # A .tsv holding the 4-column twin and a .tsv holding the 3-column legacy
    # file must both read correctly: the separator count decides, not the name.
    four = write(tmp_path, "four.tsv", "u\t-\tspoof\t1.0\n")
    three = write(tmp_path, "three.tsv", "utt_id\tlabel\tscore\nu\tspoof\t1.0\n")
    assert read_scored(four)[1].tolist() == read_scored(three)[1].tolist() == ["spoof"]


def test_r0_blank_and_malformed_lines_are_skipped(tmp_path):
    p = write(tmp_path, "a.txt", "u1 - spoof 1.0\n\ngarbage\nu2 - spoof 2.0\n")
    assert read_scored(p)[0].tolist() == ["u1", "u2"]


# --- R2: pooling ------------------------------------------------------------

def test_r2_multiple_paths_concatenate_in_order(tmp_path):
    a = write(tmp_path, "a.txt", "u1 - spoof 1.0\n")
    b = write(tmp_path, "b.txt", "u2 - bonafide 2.0\n")
    utt, lab, sc = read_scored([a, b])
    assert list(utt) == ["u1", "u2"]
    assert list(lab) == ["spoof", "bonafide"]


# --- R3: the MLAAD pool, which is P8 ---------------------------------------

def test_r3_legacy_mlaad_is_one_prepooled_file():
    paths = mlaad_pool_paths("apc", scores_root=ROOT, layout="legacy")
    assert paths == [os.path.join(
        ROOT, "linear_head_MLAAD_v10", "tsv", "linear_head_MLAAD_v10_apc.tsv")]


def test_r3_v3_mlaad_is_two_single_class_corpora_pooled_at_read_time():
    paths = mlaad_pool_paths("apc", scores_root=ROOT, layout="v3")
    assert paths == [
        os.path.join(ROOT, "raw", "linear_head", "mlaad_v10", "apc.tsv"),
        os.path.join(ROOT, "raw", "linear_head", "mailabs", "apc.txt"),
    ]


def test_r3_mlaad_spoof_half_is_read_from_the_tsv_twin():
    # utt_ids contain spaces, so the space-separated .txt cannot be parsed by a
    # whitespace splitter; M-AILABS ids never do and have no twin.
    spoof, bona = mlaad_pool_paths("apc", scores_root=ROOT, layout="v3")
    assert spoof.endswith(".tsv") and bona.endswith(".txt")


def test_r3_non_ssl_systems_have_an_mlaad_pool_under_v3():
    paths = mlaad_pool_paths("ignored", scores_root=ROOT, layout="v3",
                             system="lfcc_gmm")
    assert paths == [
        os.path.join(ROOT, "raw", "non_ssl", "mlaad_v10", "lfcc_gmm.tsv"),
        os.path.join(ROOT, "raw", "non_ssl", "mailabs", "lfcc_gmm.txt"),
    ]


def test_r3_legacy_refuses_a_non_ssl_mlaad_pool():
    # Those systems were scored on MLAAD only after the reorganisation. Naming
    # a path that was never written would be worse than refusing.
    with pytest.raises(KeyError):
        mlaad_pool_paths("x", scores_root=ROOT, layout="legacy",
                         system="lfcc_gmm")


def test_r3_unknown_layout_raises():
    with pytest.raises(ValueError):
        mlaad_pool_paths("apc", scores_root=ROOT, layout="v9")


# --- R4/R5: enumeration -----------------------------------------------------

def test_r4_v3_enumeration_returns_file_stems_sorted(tmp_path):
    d = tmp_path / "raw" / "linear_head" / "asvspoof5"
    d.mkdir(parents=True)
    for n in ("xls_r_300m.txt", "apc.txt", "notes.md"):
        (d / n).write_text("")
    got = available_frontends("linear_head", "asvspoof5",
                              scores_root=str(tmp_path), layout="v3")
    assert got == ["apc", "xls_r_300m"]


def test_r4_v3_enumeration_recovers_names_containing_underscores(tmp_path):
    d = tmp_path / "raw" / "linear_head" / "asvspoof5"
    d.mkdir(parents=True)
    (d / "multires_hubert_multilingual_large600k.txt").write_text("")
    got = available_frontends("linear_head", "asvspoof5",
                              scores_root=str(tmp_path), layout="v3")
    assert got == ["multires_hubert_multilingual_large600k"]


def test_r4_legacy_enumeration_inverts_the_naming_template(tmp_path):
    d = tmp_path / "linear_head_MLAAD_v10"
    d.mkdir(parents=True)
    (d / "linear_head_MLAAD_v10_hubert_large_ll60k.txt").write_text("")
    got = available_frontends("linear_head", "Multilingual",
                              scores_root=str(tmp_path), layout="legacy")
    assert got == ["hubert_large_ll60k"]


def test_r4_legacy_tsv_enumeration_swaps_the_extension_correctly(tmp_path):
    # Regression: the swap was done with os.path.splitext on a bare '.txt',
    # which returns ('.txt', '') because a leading dot reads as a hidden-file
    # NAME. The suffix became '.txt.tsv' and every model silently vanished.
    d = tmp_path / "linear_head_MLAAD_v10" / "tsv"
    d.mkdir(parents=True)
    (d / "linear_head_MLAAD_v10_apc.tsv").write_text("")
    got = available_frontends("linear_head", "Multilingual",
                              scores_root=str(tmp_path), layout="legacy",
                              ext=".tsv")
    assert got == ["apc"]


def test_r4_v3_non_ssl_and_linear_head_do_not_leak_into_each_other(tmp_path):
    d = tmp_path / "raw" / "non_ssl" / "asvspoof5"
    d.mkdir(parents=True)
    for n in ("lfcc_gmm.txt", "aasist_raw.txt"):
        (d / n).write_text("")
    assert available_frontends("lfcc_gmm", "asvspoof5",
                               scores_root=str(tmp_path), layout="v3") == [
        "aasist_raw", "lfcc_gmm"]


def test_r5_missing_directory_enumerates_empty(tmp_path):
    assert available_frontends("linear_head", "asvspoof5",
                               scores_root=str(tmp_path), layout="v3") == []
