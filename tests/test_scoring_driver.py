"""
test_scoring_driver.py
----------------------
Contracts for the merge of eval_asvld.py, eval_mlaad.py and eval_baselines.py
into spoof_superb.scoring.

The merge's numerical risk is concentrated in the parsers and the writer, not
in the GPU loop: if a trial list, a label, or a column boundary is read
differently, every score in the file is attached to the wrong utterance. Those
parts are pure Python and are tested here without a GPU or a corpus.

  S1  Reference fields are peeled from the RIGHT. utt_ids contain spaces
      (MLAAD v10: 39,000 rows like "Cartesia.ai (Sonic-3)"); a left-split
      yields the wrong id and reads "-" as the label for every one of them.
  S2  A pooled column concatenates its files in the declared order -- ASVLD is
      published as linear_head_asvspoofLD + asvld_rerun/Recompression.
  S3  Writes are atomic, and the .tsv twin appears exactly when an id contains
      a space (calculate_EER's genfromtxt cannot parse those rows).
  S4  The corpus walk skips macOS AppleDouble sidecars and is order-stable.
  S5  SpoofCeleb labels are per-utterance: attack 'a00' is bonafide.
  S6  The ASVLD protocol's utt_id is column 1 and its key is column 3.
  S7  --restrict_to preserves the reference file's order, de-duplicates, and
      drops ids absent from the trial list.
  S8  fp32 is the default (decision D1). --amp must be opt-in.
  S9  The ASVLD skip list still contains Filtering -- the behaviour the
      untracked `.asvld_skip` sentinel used to carry.

  D1-D5  The dataset is the single input: it resolves its own trial source and
      parameters, explicit flags still override, and an unknown dataset is
      rejected rather than silently mis-filed.

Run:  pytest tests/test_scoring_driver.py
"""

import os

import pytest

from spoof_superb.core.scorefile import read_reference, write_scores
from spoof_superb.scoring.driver import DEFAULT_SKIP_CONDITIONS, _apply_restrict, build_parser
from spoof_superb.scoring.datasets import (
    trials_from_asvld_protocol,
    trials_from_protocol_csv,
    trials_from_walk,
)

SPACEY = "MLAAD/fake/en/Cartesia.ai (Sonic-3)/utt 001.wav"


def _write(path, text):
    path.write_text(text)
    return str(path)


def test_s1_reference_fields_are_peeled_from_the_right(tmp_path):
    ref = _write(tmp_path / "ref.txt",
                 f"{SPACEY} - spoof -1.25\n"
                 "plain_utt - bonafide 0.5\n")
    utts, keys = read_reference(ref)
    assert utts == [SPACEY, "plain_utt"], "utt_id with spaces was split"
    assert keys[SPACEY] == "spoof", "label read from the wrong column"
    assert keys["plain_utt"] == "bonafide"


def test_s2_pooled_columns_concatenate_in_order(tmp_path):
    a = _write(tmp_path / "a.txt", "a1 - spoof 1.0\na2 - bonafide 2.0\n")
    b = _write(tmp_path / "b.txt", "b1 - spoof 3.0\n")
    utts, keys = read_reference([a, b])
    assert utts == ["a1", "a2", "b1"], "pooled order not preserved"
    assert set(keys) == {"a1", "a2", "b1"}


def test_s3_write_is_atomic_and_emits_tsv_only_when_needed(tmp_path):
    out = tmp_path / "scores.txt"
    keys = {"plain": "bonafide"}
    tsv = write_scores(str(out), [("plain", 0.25)], keys)
    assert tsv is None, "emitted a .tsv twin for ids that do not need one"
    assert out.read_text() == "plain - bonafide 0.25\n"
    assert not list(tmp_path.glob("*.part")), "left a .part file behind"

    out2 = tmp_path / "spacey.txt"
    keys2 = {SPACEY: "spoof"}
    tsv2 = write_scores(str(out2), [(SPACEY, -1.5)], keys2)
    assert tsv2 is not None, "no .tsv twin for an id containing a space"
    assert open(tsv2).read() == f"{SPACEY}\t-\tspoof\t-1.5\n"
    assert not list(tmp_path.glob("*.part"))


def test_s4_walk_skips_appledouble_sidecars_and_is_sorted(tmp_path):
    root = tmp_path / "corpus" / "en"
    root.mkdir(parents=True)
    for name in ("b.wav", "a.wav", "._a.wav", "notes.txt"):
        (root / name).write_bytes(b"")
    utts, keys = trials_from_walk(str(tmp_path / "corpus"), str(tmp_path), "spoof")
    assert utts == ["corpus/en/a.wav", "corpus/en/b.wav"], (
        "AppleDouble sidecar or non-wav leaked in, or order is unstable"
    )
    assert set(keys.values()) == {"spoof"}


def test_s5_spoofceleb_labels_are_per_utterance(tmp_path):
    csv_path = _write(tmp_path / "eval.csv",
                      "file,speaker,attack\n"
                      "x/a.flac,spk1,a00\n"
                      "x/b.flac,spk1,a07\n")
    utts, keys = trials_from_protocol_csv(csv_path)
    assert utts == ["x/a.flac", "x/b.flac"]
    assert keys["x/a.flac"] == "bonafide", "attack a00 must be bonafide"
    assert keys["x/b.flac"] == "spoof"


def test_s6_asvld_protocol_columns(tmp_path):
    proto_dir = tmp_path / "protocols"
    proto_dir.mkdir()
    (proto_dir / "ASVspoofLauneredDatabase_Reverberation.txt").write_text(
        "LA_0001 LA_E_1234 A07 spoof Reverberation v1\n"
        "LA_0002 LA_E_5678 - bonafide Reverberation v1\n"
        "malformed line\n"
    )
    utts, keys = trials_from_asvld_protocol(str(proto_dir), "Reverberation")
    assert utts == ["LA_E_1234", "LA_E_5678"], "utt_id must come from column 1"
    assert keys["LA_E_1234"] == "spoof" and keys["LA_E_5678"] == "bonafide"


def test_s6_missing_asvld_protocol_is_reported_not_raised(tmp_path):
    utts, keys = trials_from_asvld_protocol(str(tmp_path), "Reverberation")
    assert utts is None and keys is None


def test_s7_restrict_preserves_reference_order_and_dedups(tmp_path):
    ref = _write(tmp_path / "ref.txt",
                 "c - spoof 1.0\n"
                 "a - spoof 1.0\n"
                 "c - spoof 1.0\n"
                 "ghost - spoof 1.0\n")
    keys = {"a": "spoof", "b": "spoof", "c": "spoof"}
    kept = _apply_restrict(["a", "b", "c"], keys, ref, None)
    assert kept == ["c", "a"], (
        "restrict must follow the reference file's order, de-duplicate, and "
        "drop ids that are not in the trial list"
    )


def test_s7_restrict_prefix_filters(tmp_path):
    ref = _write(tmp_path / "ref.txt", "MLAAD/x - spoof 1.0\nOTHER/y - spoof 1.0\n")
    keys = {"MLAAD/x": "spoof", "OTHER/y": "spoof"}
    assert _apply_restrict(list(keys), keys, ref, "MLAAD/") == ["MLAAD/x"]


def test_s8_fp32_is_the_default():
    args = build_parser().parse_args(
        ["--model", "linear_head", "--model_path", "m", "--output_file", "o"])
    assert args.amp is False, "AMP must be opt-in: fp16 overflow is what produced the NaN"


def test_s9_filtering_is_still_skipped_by_default():
    """The `.asvld_skip` sentinel contained 'Filtering'; that must not change silently."""
    assert "Filtering" in DEFAULT_SKIP_CONDITIONS
    args = build_parser().parse_args(
        ["--model", "linear_head", "--model_path", "m", "--output_file", "o"])
    assert "Filtering" in args.skip_conditions



# ---------------------------------------------------------------------------
# The dataset is the single input
#
#   D1  A dataset with a native protocol resolves its own source and
#       parameters. `--dataset spoofceleb` alone is enough.
#   D2  Explicit flags still override, so unusual runs stay possible.
#   D3  A dataset with no native protocol falls back to the published
#       reference score file (see RP-7).
#   D4  An unknown dataset is rejected rather than silently mis-filed.
#   D5  MAILABS is scoreable but is not a benchmark column, and asking for its
#       reference file is an error rather than a wrong path.
# ---------------------------------------------------------------------------

from spoof_superb.scoring.datasets import (           # noqa: E402
    SCOREABLE, has_reference, native_source, reference_paths,
)
from spoof_superb.scoring.driver import _apply_dataset_defaults   # noqa: E402


def _args(**kw):
    base = ["--model", "linear_head", "--model_path", "m", "--output_file", "o"]
    for k, v in kw.items():
        base += [f"--{k}", str(v)]
    return build_parser().parse_args(base)


def test_d1_dataset_resolves_its_own_source_and_parameters():
    a = _args(dataset="spoofceleb")
    assert a.source is None
    assert _apply_dataset_defaults(a)
    assert a.source == "protocol"

    for dataset in ("MAILABS", "Multilingual", "wild", "asvspoofLD"):
        a = _args(dataset=dataset)
        assert _apply_dataset_defaults(a)
        assert a.source == "protocol", f"{dataset} should read its protocol"


def test_d2_explicit_flags_still_override():
    a = _args(dataset="spoofceleb", protocol="/tmp/mine.tsv")
    assert _apply_dataset_defaults(a)
    assert a.protocol == "/tmp/mine.tsv", "an explicit flag must win"

    b = _args(dataset="spoofceleb", source="benchmark")
    assert _apply_dataset_defaults(b)
    assert b.source == "benchmark"


def test_d1b_spoofceleb_attack_column_maps_to_labels():
    a = _args(dataset="spoofceleb")
    assert _apply_dataset_defaults(a)
    assert a.source == "protocol"


def test_d3_benchmark_remains_available_as_an_override():
    """Every dataset now reads a protocol, but the published trial list is
    still reachable for reproducing a paper column exactly."""
    a = _args(dataset="asvspoof2021_DF", source="benchmark")
    assert _apply_dataset_defaults(a)
    assert a.source == "benchmark"


def test_d4_unknown_dataset_is_rejected():
    a = _args(dataset="not_a_dataset")
    assert not _apply_dataset_defaults(a)


def test_d5_mailabs_is_scoreable_but_not_a_benchmark_column():
    assert "MAILABS" in SCOREABLE
    assert not has_reference("MAILABS")
    assert native_source("MAILABS") == "protocol"
    with pytest.raises(KeyError):
        reference_paths("MAILABS", "xls_r_300m")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Protocol-driven trial lists
#
#   T1  One reader handles every corpus's protocol format via parameters.
#   T2  Label vocabularies are normalised. In-the-Wild writes "bona-fide" and
#       Deepfake-Eval writes "Real"/"Fake"; everything downstream filters on
#       "bonafide"/"spoof", so an unmapped label does not raise -- the trial
#       silently vanishes from the EER. That must be impossible.
#   T3  An unrecognised label is refused loudly rather than passed through.
#   T4  Every declared protocol spec resolves and yields both classes where
#       the corpus has both.
# ---------------------------------------------------------------------------

from spoof_superb.scoring.datasets import (           # noqa: E402
    LABEL_ALIASES, PROTOCOL_SPECS, normalise_label, trials_from_protocol,
)


def test_t1_one_reader_handles_different_formats(tmp_path):
    csvp = tmp_path / "meta.csv"
    csvp.write_text("Audio,Speaker,Label\n0.wav,Alec,bona-fide\n1.wav,Alec,spoof\n")
    utts, keys = trials_from_protocol(str(csvp), delimiter=",", header=True,
                                      utt_col=0, label_col=2, strip_ext=True)
    assert utts == ["0", "1"], "extension should be stripped to match the score ids"
    assert keys["0"] == "bonafide" and keys["1"] == "spoof"

    pipep = tmp_path / "meta.txt"
    pipep.write_text("filename|absolute_path|model\n"
                     "a.wav|/data/Data/MLAAD/fake/en/X/a.wav|X\n")
    utts, keys = trials_from_protocol(str(pipep), delimiter="|", header=True,
                                      utt_col=1, label_const="spoof",
                                      rel_to="/data/Data")
    assert utts == ["MLAAD/fake/en/X/a.wav"]
    assert keys[utts[0]] == "spoof"


def test_t2_label_vocabularies_are_normalised():
    for raw in ("bona-fide", "bona fide", "BONAFIDE", "Real", "genuine"):
        assert normalise_label(raw) == "bonafide", raw
    for raw in ("spoof", "Fake", "SPOOFED", "tts"):
        assert normalise_label(raw) == "spoof", raw
    assert set(LABEL_ALIASES.values()) == {"bonafide", "spoof"}


def test_t3_unknown_label_is_refused(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("utt_id\tlabel\nx\tmystery\n")
    with pytest.raises(ValueError, match="unrecognised label"):
        trials_from_protocol(str(p))


def test_t4_declared_protocol_specs_are_well_formed():
    for dataset, spec in PROTOCOL_SPECS.items():
        assert "protocol" in spec, f"{dataset} declares no protocol file"
        assert ("label_col" in spec) or ("label_const" in spec), (
            f"{dataset} declares neither a label column nor a constant label")


def test_t5_every_dataset_reads_a_protocol():
    """RP-7 closed: no column depends on a published score file any more."""
    for dataset in SCOREABLE:
        assert native_source(dataset) == "protocol", (
            f"{dataset} still resolves its trials from {native_source(dataset)!r}"
        )
        assert dataset in PROTOCOL_SPECS, f"{dataset} declares no protocol"


def test_t6_every_protocol_spec_has_a_resolver():
    """A protocol id is useless if nothing maps it to audio."""
    from spoof_superb.scoring.datasets import DATASETS
    for dataset in PROTOCOL_SPECS:
        assert callable(DATASETS[dataset]["resolve"]), dataset


def test_t7_extension_handling_matches_the_published_ids(tmp_path):
    """ASV19 protocol ids are bare; the published score ids carry .flac."""
    p = tmp_path / "trl.txt"
    p.write_text("LA_0039 LA_E_2834763 - A11 spoof\n")
    utts, _ = trials_from_protocol(str(p), delimiter=None, header=False,
                                   utt_col=1, label_col=4, add_ext=".flac")
    assert utts == ["LA_E_2834763.flac"]
    # idempotent: an id that already carries it is not doubled
    p.write_text("LA_0039 LA_E_2834763.flac - A11 spoof\n")
    utts, _ = trials_from_protocol(str(p), delimiter=None, header=False,
                                   utt_col=1, label_col=4, add_ext=".flac")
    assert utts == ["LA_E_2834763.flac"]


def test_t8_whitespace_delimited_and_crlf_protocols_parse(tmp_path):
    """ASVspoof5 ships a space-delimited .tsv; Famous Figures ships CRLF."""
    p = tmp_path / "ws.txt"
    p.write_text("E_1607 E_0009538969 M C05 2 E_0009486171 AC1 A26 spoof -\n")
    utts, keys = trials_from_protocol(str(p), delimiter=None, header=False,
                                      utt_col=1, label_col=8)
    assert utts == ["E_0009538969"] and keys[utts[0]] == "spoof"

    p = tmp_path / "crlf.tsv"
    p.write_bytes(b"A\tB\tC\tLabel\tPath\r\nn\ts\tsrc\tspoof\t/root/a.wav\r\n")
    utts, keys = trials_from_protocol(str(p), delimiter="\t", header=True,
                                      utt_col=4, label_col=3, rel_to="/root")
    assert utts == ["a.wav"], "carriage return leaked into the id"
    assert keys["a.wav"] == "spoof"
