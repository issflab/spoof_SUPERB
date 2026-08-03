"""
test_views.py
-------------
Contracts for analysis.views, the P11 view layer.

  V0  A view PARTITIONS its source. Every raw row lands in exactly one group,
      or is dropped by an explicit rule -- never silently. A view that invents
      or loses rows is a second, wrong copy of the data, which is the failure
      the whole design exists to avoid.
  V1  The grouping key is derived from the utt_id by code. That is what makes a
      view rebuildable and checkable against raw rather than hand-maintained.
  V2  The frontend is the LAST path component, at either depth (P11 D2), so one
      glob finds one model everywhere. The legacy degradation tree lost this and
      paid for it with three hand-maintained stem dictionaries.
  V3  An unrecognised grouping key RAISES. Dropping a condition or a TTS system
      would shrink a benchmark column without saying so.
  V4  Rows excluded by an explicit rule are excluded quietly and on purpose --
      MLAAD's voice-conversion and vocoder directories are not TTS systems.
  V5  Reserved names are underscore-prefixed, so no real group can collide with
      `_bonafide` or `_manifest.json` and none sorts into the middle of them.

Run:  pytest tests/test_views.py
"""

import numpy as np
import pytest

from spoof_superb.analysis import views
from spoof_superb.analysis.views import VIEW_SPECS, load_view


# --- V1/V3: the ASVLD condition key ----------------------------------------

@pytest.mark.parametrize("utt,expected", [
    ("LA_E_2834763_babble_0",         ("Additive_Noise", "babble_0")),
    ("LA_E_1_white_20",               ("Additive_Noise", "white_20")),
    ("LA_E_1_cafe_10",                ("Additive_Noise", "cafe_10")),
    ("LA_E_1_volvo_0",                ("Additive_Noise", "volvo_0")),
    ("LA_E_1_street_20",              ("Additive_Noise", "street_20")),
    ("LA_E_1_RT_0_3",                 ("Reverberation", "RT_0_3")),
    ("LA_E_1_resample_8000",          ("Resampling", "resample_8000")),
    ("LA_E_1_recompression_128k",     ("Codec_Compression", "recompression_128k")),
    ("LA_E_1_lpf_7000",               ("Channel_Distortions", "lpf_7000")),
])
def test_v1_asvld_conditions_map_to_family_and_condition(utt, expected):
    assert views._asvld_key(utt) == expected


def test_v2_the_condition_is_the_leaf_not_the_family():
    # The family is a function of the condition; the reverse is not true. The
    # legacy view kept only the family, which is why it cannot distinguish
    # babble from white, or 0 dB from 20 dB, inside one file.
    family, condition = views._asvld_key("LA_E_1_babble_0")
    assert family == "Additive_Noise" and condition == "babble_0"
    assert views._asvld_key("LA_E_1_white_0")[1] == "white_0"


def test_v3_an_unknown_asvld_condition_raises():
    with pytest.raises(KeyError):
        views._asvld_key("LA_E_1_teleport_9000")


def test_v1_an_id_with_no_condition_is_not_a_view_row():
    # A clean ASVLD id carries no condition, so it belongs to no group. That is
    # a None, not a raise: absence of a suffix is a shape, not a bad value.
    assert views._asvld_key("LA_E_2834763") is None


# --- V1: MLAAD keys ---------------------------------------------------------

def test_v1_mlaad_language_comes_from_the_id():
    assert views._mlaad_language_key(
        "MLAAD/fake/de/Edge-TTS/x.wav") == ("de",)


def test_v3_a_malformed_mlaad_id_raises():
    with pytest.raises(ValueError):
        views._mlaad_language_key("MLAAD/fake/de/x.wav")


# --- V0: a view partitions its source --------------------------------------

def test_v0_load_view_partitions_every_row(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "linear_head" / "asvspoof_ld"
    raw.mkdir(parents=True)
    (raw / "m.txt").write_text(
        "LA_E_1_babble_0 - spoof 1.0\n"
        "LA_E_2_babble_0 - bonafide 2.0\n"
        "LA_E_3_RT_0_3 - spoof 3.0\n"
        "LA_E_4_lpf_7000 - spoof 4.0\n"
    )
    groups, bonafide = load_view(VIEW_SPECS["asvld_conditions"], "m",
                                 scores_root=str(tmp_path), layout="v3")
    assert bonafide is None          # ASVLD carries its own bonafide rows
    assert sum(len(g[0]) for g in groups.values()) == 4
    assert set(groups) == {("Additive_Noise", "babble_0"),
                           ("Reverberation", "RT_0_3"),
                           ("Channel_Distortions", "lpf_7000")}
    utts, labels, scores = groups[("Additive_Noise", "babble_0")]
    assert list(utts) == ["LA_E_1_babble_0", "LA_E_2_babble_0"]
    assert list(labels) == ["spoof", "bonafide"]
    assert list(scores) == [1.0, 2.0]


def test_v0_groups_preserve_source_order(tmp_path):
    raw = tmp_path / "raw" / "linear_head" / "asvspoof_ld"
    raw.mkdir(parents=True)
    (raw / "m.txt").write_text("".join(
        f"LA_E_{i}_babble_0 - spoof {i}.0\n" for i in range(5)))
    groups, _ = load_view(VIEW_SPECS["asvld_conditions"], "m",
                          scores_root=str(tmp_path), layout="v3")
    _u, _l, scores = groups[("Additive_Noise", "babble_0")]
    assert list(scores) == [0.0, 1.0, 2.0, 3.0, 4.0]


# --- V4: explicit exclusions ------------------------------------------------

def test_v4_excluded_mlaad_directories_return_none(monkeypatch):
    key = views._MlaadTtsKey()
    key._system = {"griffin_lim": "EXCLUDED", "Bark": "Bark"}
    key._bucket = {"Bark": "AR"}
    assert key("MLAAD/fake/en/griffin_lim/x.wav") is None
    assert key("MLAAD/fake/en/Bark/x.wav") == ("AR", "Bark")


def test_v3_an_unmapped_mlaad_directory_raises():
    key = views._MlaadTtsKey()
    key._system = {"Bark": "Bark"}
    key._bucket = {"Bark": "AR"}
    with pytest.raises(KeyError):
        key("MLAAD/fake/en/BrandNewTTS/x.wav")


# --- V5: reserved names -----------------------------------------------------

def test_v5_a_group_named_like_a_reserved_path_cannot_collide(tmp_path):
    """The reserved names are unreachable as group keys, not merely unlikely.

    Legacy used `bonafide/`, which a TTS system called "bonafide" would have
    silently overwritten, and which sorted into the middle of the real systems.
    The underscore makes the collision impossible rather than improbable: group
    keys come from utt_id components, and a corpus directory beginning with an
    underscore would have to be created deliberately.
    """
    raw = tmp_path / "raw" / "linear_head" / "asvspoof_ld"
    raw.mkdir(parents=True)
    (raw / "m.txt").write_text("LA_E_1_babble_0 - spoof 1.0\n")
    groups, _ = load_view(VIEW_SPECS["asvld_conditions"], "m",
                          scores_root=str(tmp_path), layout="v3")
    produced = {part for key in groups for part in key}
    assert not any(p.startswith("_") for p in produced)
    assert produced.isdisjoint({"_bonafide", "_manifest.json"})


# --- the registry itself ----------------------------------------------------

def test_registry_declares_a_source_for_every_view():
    for name, spec in VIEW_SPECS.items():
        assert spec.dataset, name
        assert spec.doc, name
        assert callable(spec.key), name


def test_registry_has_no_acoustic_degradation_view():
    # The legacy scores_by_acoustic_degradation mixes ASVspoof2019 LA,
    # ASVspoof2021 DF and ASVspoof5, and the v3 tree holds those corpora clean
    # only. A view of that name here would claim to reproduce a population it
    # cannot. See P11.
    assert "acoustic_degradation" not in VIEW_SPECS
    assert "asvld_conditions" in VIEW_SPECS
