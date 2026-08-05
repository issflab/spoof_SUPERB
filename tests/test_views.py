"""
test_views.py
-------------
Contracts for the two analysis views and the condition resolution behind them.

  V0  A partition view PARTITIONS its source: every row lands in exactly one
      group or is dropped by an explicit rule, never silently.
  V1  A composite view POOLS partitions of several datasets into each named
      group, exactly as tab:acoustic_degradation specifies. Group membership is
      the paper's, restated once, in one place.
  V2  Each degraded condition RETAINS the corpora it does not degrade. That is
      what makes its EER comparable to the Baseline: only the degradation under
      study changes. A condition built from the degraded corpus alone would
      confound the degradation with that corpus's own difficulty.
  V3  Condition codes come from each corpus's own protocol, not from the
      utt_id, except ASVLD where the utt_id carries them.
  V4  An unrecognised condition or an unmapped TTS directory RAISES. Dropping
      one would shrink an evaluation set without saying so.
  V5  Rows excluded by an explicit rule are excluded quietly and on purpose --
      MLAAD's voice-conversion and vocoder entries are not TTS systems.

Run:  pytest tests/test_views.py
"""

import numpy as np
import pytest

from spoof_superb.analysis import conditions, views
from spoof_superb.analysis.views import (VIEW_SPECS, CompositeView, Part,
                                         PartitionView, load_view)


# --- V3/V4: condition resolution -------------------------------------------

@pytest.mark.parametrize("cond,family", [
    ("babble_0", "Additive_Noise"), ("white_20", "Additive_Noise"),
    ("cafe_10", "Additive_Noise"), ("volvo_0", "Additive_Noise"),
    ("street_20", "Additive_Noise"),
    ("RT_0_3", "Reverberation"),
    ("resample_8000", "Bandwidth"),
    ("recompression_128k", "Codec_Compression"),
    ("lpf_7000", "Channel_Distortions"),
])
def test_v3_asvld_conditions_map_to_families(cond, family):
    assert conditions.asvld_family(cond) == family


def test_v4_an_unknown_asvld_condition_raises():
    with pytest.raises(KeyError):
        conditions.asvld_family("teleport_9000")


def test_v3_asvld_condition_comes_from_the_utt_id():
    of = conditions.condition_of("asvspoofLD")
    assert of("LA_E_2834763_babble_0") == "babble_0"
    # A clean ASVLD id carries no suffix, so it has no condition. That is a
    # shape, not a bad value.
    assert of("LA_E_2834763") is None


def test_v3_asv19_has_no_condition_column():
    # Its whole set is the clean reference. Inventing a single-valued condition
    # would imply a partitioning that does not exist.
    assert conditions.condition_of("eval_2019") is None


def test_v3_the_clean_partition_of_each_corpus_is_named_once():
    # Only C1/C00 is load-bearing: every other condition enters a composition
    # as "all the rest", so it cannot drift.
    assert conditions.CLEAN["asvspoof2021_LA"] == "none"
    assert conditions.CLEAN["asvspoof2021_DF"] == "nocodec"
    assert conditions.CLEAN["asvspoof5"] == "-"
    assert conditions.ASV21_LA_CODES["none"] == "C1"
    assert conditions.ASV21_DF_CODES["nocodec"] == "C1"


# --- V1/V2: the acoustic degradation composition ---------------------------

def test_v1_the_six_conditions_are_the_papers():
    spec = VIEW_SPECS["acoustic_degradation"]
    assert isinstance(spec, CompositeView)
    assert set(spec.groups) == {
        "Baseline", "Codec_Compression", "Bandwidth", "Additive_Noise",
        "Reverberation", "Channel_Distortions"}
    assert spec.reference == "Baseline"


def test_v1_baseline_is_the_clean_partition_of_four_corpora():
    parts = VIEW_SPECS["acoustic_degradation"].groups["Baseline"]
    assert {p.dataset for p in parts} == {
        "eval_2019", "asvspoof2021_LA", "asvspoof2021_DF", "asvspoof5"}
    for p in parts:
        if p.dataset == "eval_2019":
            assert p.conditions is None       # whole set
        else:
            assert p.conditions == (conditions.CLEAN[p.dataset],)


@pytest.mark.parametrize("condition", [
    "Bandwidth", "Additive_Noise", "Reverberation"])
def test_v2_asvld_conditions_retain_the_other_three_corpora_clean(condition):
    parts = {p.dataset: p for p in
             VIEW_SPECS["acoustic_degradation"].groups[condition]}
    assert parts["asvspoof2021_LA"].conditions == ("none",)
    assert parts["asvspoof2021_DF"].conditions == ("nocodec",)
    assert parts["asvspoof5"].conditions == ("-",)
    assert parts["asvspoofLD"].family == condition


def test_v2_codec_replaces_df_and_asv5_but_retains_la_clean():
    parts = {p.dataset: p for p in
             VIEW_SPECS["acoustic_degradation"].groups["Codec_Compression"]}
    assert parts["asvspoofLD"].family == "Codec_Compression"
    # DF C2-C9 and ASV5 C01-C10 are stated as exclusions, so a corpus that
    # gained a condition would be included rather than silently dropped.
    assert parts["asvspoof2021_DF"].exclude == ("nocodec",)
    assert parts["asvspoof5"].exclude == ("-", "C11")
    assert parts["asvspoof2021_LA"].conditions == ("none",)


def test_v2_channel_distortions_uses_la_c2_c7_and_asv5_c11():
    parts = {p.dataset: p for p in
             VIEW_SPECS["acoustic_degradation"].groups["Channel_Distortions"]}
    assert parts["asvspoof2021_LA"].exclude == ("none",)
    assert parts["asvspoof5"].conditions == ("C11",)
    assert parts["asvspoof2021_DF"].conditions == ("nocodec",)
    assert parts["eval_2019"].conditions is None
    # ASVLD's own lpf_* set is deliberately NOT this condition's source.
    assert "asvspoofLD" not in parts


def test_v1_a_part_takes_at_most_one_selector():
    with pytest.raises(ValueError):
        Part("asvspoof5", conditions=("C01",), exclude=("-",))


def test_v1_part_selection_semantics():
    assert Part("d", conditions=("a", "b")).selects("a")
    assert not Part("d", conditions=("a",)).selects("b")
    assert Part("d", exclude=("a",)).selects("b")
    assert not Part("d", exclude=("a",)).selects("a")
    assert Part("d").selects("anything")          # no selector = whole set


# --- V0: partition views ----------------------------------------------------

def test_v0_a_partition_view_covers_every_row(tmp_path):
    raw = tmp_path / "raw" / "linear_head" / "mlaad_v10"
    raw.mkdir(parents=True)
    (raw / "m.tsv").write_text(
        "MLAAD/fake/de/edge_tts_dir/a.wav\t-\tspoof\t1.0\n"
        "MLAAD/fake/en/bark_dir/b.wav\t-\tspoof\t2.0\n"
        "MLAAD/fake/en/rvc_dir/c.wav\t-\tspoof\t9.0\n")
    mail = tmp_path / "raw" / "linear_head" / "mailabs"
    mail.mkdir(parents=True)
    (mail / "m.txt").write_text("MAILabs/x.wav - bonafide 3.0\n")

    # Stub the taxonomy: the real map keys on RAW CORPUS directory names, which
    # a synthetic fixture has no business reproducing.
    spec = VIEW_SPECS["tts_systems"]
    spec.key._system = {"edge_tts_dir": "Edge-TTS", "bark_dir": "Bark",
                        "rvc_dir": "EXCLUDED"}
    spec.key._mode = {"Edge-TTS": "closed_undisclosed", "Bark": "AR"}
    try:
        groups, bonafide = load_view(spec, "m", scores_root=str(tmp_path))
    finally:
        spec.key._system = spec.key._mode = None

    # Two retained, one excluded by rule -- and the excluded row is gone, not
    # filed under a group named EXCLUDED.
    assert sum(len(g[0]) for g in groups.values()) == 2
    assert set(groups) == {("closed_undisclosed", "Edge-TTS"), ("AR", "Bark")}
    assert len(bonafide[0]) == 1          # the shared reference pool


def test_v5_excluded_mlaad_directories_return_none():
    key = views._MlaadTtsKey()
    key._system = {"griffin_lim": "EXCLUDED", "Bark": "Bark"}
    key._mode = {"Bark": "AR"}
    assert key("MLAAD/fake/en/griffin_lim/x.wav") is None
    assert key("MLAAD/fake/en/Bark/x.wav") == ("AR", "Bark")


def test_v4_an_unmapped_mlaad_directory_raises():
    key = views._MlaadTtsKey()
    key._system = {"Bark": "Bark"}
    key._mode = {"Bark": "AR"}
    with pytest.raises(KeyError):
        key("MLAAD/fake/en/BrandNewTTS/x.wav")


def test_v4_a_malformed_mlaad_id_raises():
    with pytest.raises(ValueError):
        views._mlaad_parts("MLAAD/fake/de/x.wav")


# --- the registry -----------------------------------------------------------

def test_registry_holds_exactly_the_two_analyses():
    assert set(VIEW_SPECS) == {"acoustic_degradation", "tts_systems"}


def test_registry_documents_every_view():
    for name, spec in VIEW_SPECS.items():
        assert spec.doc, name
        assert isinstance(spec, (CompositeView, PartitionView)), name


def test_tts_view_scores_against_a_shared_bonafide_pool():
    # The reason the analysis is restricted to MLAAD: every system synthesises
    # from the same bonafide source, so a per-system EER measures the system
    # rather than its source corpus.
    assert VIEW_SPECS["tts_systems"].bonafide_dataset == "MAILABS"


def test_tts_view_reads_the_tab_separated_twin():
    # ~8.6% of MLAAD utt_ids contain spaces.
    assert VIEW_SPECS["tts_systems"].ext == ".tsv"
