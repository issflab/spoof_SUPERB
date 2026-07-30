"""Contracts for the audio loader.

The loader sits under every score in the benchmark, so the contract that
matters is not "it decodes" but "it decodes to exactly what the previous
implementation produced". A decoder swap that shifted samples by 1 LSB would
change every EER in Table 5 without failing anything else.

FLAC is lossless, so bit-identity across decoders is a property that must hold,
not an approximation to tolerate. These tests assert it on real corpus files --
including the ones libsndfile refuses, which is the whole reason the module
exists.

Corpus-dependent tests skip when the data is not mounted.
"""

import os
import random
import warnings

import numpy as np
import pytest

from spoof_superb.scoring.audio import (
    _load_audioread,
    _load_pyav,
    _load_soundfile,
    have_pyav,
    load_wave,
    native_rate,
)

DF = "/data/Data/ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/flac"
pytestmark = pytest.mark.filterwarnings("ignore")


def _corpus(n=40, seed=11):
    """(readable, refused) samples of real ASV21-DF files."""
    import soundfile as sf
    if not os.path.isdir(DF):
        pytest.skip("ASVspoof2021-DF not mounted")
    files = sorted(os.listdir(DF))[:4000]
    random.seed(seed)
    ok, bad = [], []
    for name in random.sample(files, min(len(files), 600)):
        p = os.path.join(DF, name)
        try:
            sf.read(p)
            ok.append(p)
        except Exception:
            bad.append(p)
        if len(ok) >= n and len(bad) >= n:
            break
    if not bad:
        pytest.skip("no libsndfile-refused files in this sample")
    return ok[:n], bad[:n]


# ===========================================================================
# A1-A3: the loader reproduces librosa.load exactly
# ===========================================================================

def test_a1_matches_librosa_on_files_soundfile_can_read(tmp_path):
    """The fast path must be bit-identical, or every existing score moves.

    This is the regression that would be invisible: soundfile reads these files
    today through librosa, and will read them tomorrow through load_wave. The
    two must agree to the last bit.
    """
    import librosa
    import soundfile as sf
    rng = np.random.default_rng(0)
    for sr, ch in ((16000, 1), (16000, 2), (22050, 1)):
        p = tmp_path / f"t_{sr}_{ch}.wav"
        x = rng.standard_normal((4000, ch)).astype(np.float32) * 0.1
        sf.write(p, x, sr, subtype="PCM_16")
        got = load_wave(str(p), 16000)
        want, _ = librosa.load(str(p), sr=16000)
        assert got.shape == want.shape, (sr, ch)
        assert np.abs(got - want).max() == 0.0, (sr, ch)


def test_a2_matches_librosa_on_real_corpus_audio():
    """Same contract, on the actual bytes the benchmark scores."""
    import librosa
    ok, _ = _corpus()
    for p in ok:
        got = load_wave(p, 16000)
        want, _ = librosa.load(p, sr=16000)
        assert got.shape == want.shape, p
        assert np.abs(got - want).max() == 0.0, p


def test_a3_native_rate_preserves_the_lfcc_contract(tmp_path):
    """sr=None must keep the file's own rate: the LFCC-GMM reference does."""
    import librosa
    import soundfile as sf
    p = tmp_path / "native.wav"
    sf.write(p, np.zeros(8000, dtype=np.float32), 22050, subtype="PCM_16")
    got = load_wave(str(p), sr=None)
    want, want_sr = librosa.load(str(p), sr=None)
    assert native_rate(str(p)) == want_sr == 22050
    assert got.shape == want.shape and np.abs(got - want).max() == 0.0


# ===========================================================================
# A4-A6: the decoders agree on the files libsndfile refuses
# ===========================================================================

def test_a4_libsndfile_really_does_refuse_part_of_this_corpus():
    """Guards the premise. If a libsndfile upgrade fixes it, this test says so."""
    ok, bad = _corpus()
    assert bad, "no refused files -- the fallback may no longer be needed"
    with pytest.raises(Exception):
        _load_soundfile(bad[0], 16000)


@pytest.mark.skipif(not have_pyav(), reason="PyAV not installed")
def test_a5_pyav_agrees_with_audioread_bit_for_bit():
    """The substitution is only safe because FLAC is lossless. Verify it."""
    _, bad = _corpus()
    for p in bad:
        a = _load_pyav(p, 16000)
        b = _load_audioread(p, 16000)
        assert a.shape == b.shape, p
        assert np.abs(a - b).max() == 0.0, p


def test_a6_refused_files_still_load_by_some_path():
    """Whatever is installed, a readable file must never come back unreadable."""
    _, bad = _corpus()
    for p in bad[:10]:
        x = load_wave(p, 16000)
        assert x.dtype == np.float32 and x.ndim == 1 and len(x) > 0


# ===========================================================================
# A7-A9: policy
# ===========================================================================

def test_a7_reports_which_decoder_handled_each_file():
    """Needed to tell a fast run from a slow one without timing it."""
    ok, bad = _corpus(n=5)
    stats = {}
    for p in ok[:5] + bad[:5]:
        load_wave(p, 16000, _stats=stats)
    assert stats.get("soundfile") == 5
    assert stats.get("pyav" if have_pyav() else "audioread") == 5


def test_a8_unreadable_file_raises_the_soundfile_error(tmp_path):
    """The caller's missing-audio policy must see a decode failure as such."""
    p = tmp_path / "not_audio.flac"
    p.write_bytes(b"this is not audio at all")
    with pytest.raises(Exception):
        load_wave(str(p), 16000)


def test_a9_missing_pyav_is_not_an_error(tmp_path):
    """PyAV is optional: without it the loader degrades to today's behaviour."""
    import soundfile as sf
    p = tmp_path / "plain.wav"
    sf.write(p, np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
    assert load_wave(str(p), 16000).shape == (1600,)
    assert have_pyav() in (True, False)


# ===========================================================================
# A10-A12: the loader says which decoder ran, and why
# ===========================================================================

def test_a10_announces_the_fallback_and_its_reason(capsys):
    """A silent fallback cost an hour of diagnosis; it must be in the log.

    librosa's own warning names the decoder that FAILED, never the one that
    ran. Without this a sweep using the 35x-slower path looks identical in the
    log to one using the fast path.
    """
    import spoof_superb.scoring.audio as A
    _, bad = _corpus(n=3)
    A._announced.clear()
    for p in bad[:3]:
        load_wave(p, 16000)
    out = capsys.readouterr().out
    assert "[audio] using" in out
    expected = "pyav" if have_pyav() else "audioread (SLOW)"
    assert expected in out, f"expected {expected!r} in:\n{out}"
    assert "libsndfile refused it" in out or "av is not installed" in out


def test_a11_announces_once_per_reason_not_once_per_file(capsys):
    """611,829 files must not produce 611,829 lines."""
    import spoof_superb.scoring.audio as A
    _, bad = _corpus(n=8)
    A._announced.clear()
    for p in bad:
        load_wave(p, 16000)
    lines = [l for l in capsys.readouterr().out.splitlines() if "[audio]" in l]
    # libsndfile fails three distinct ways, so at most three reasons.
    assert 1 <= len(lines) <= 3, lines


def test_a12_reporting_errors_are_not_mistaken_for_decoder_failures(monkeypatch):
    """A bug in the announcement must not demote every file to the slow path.

    It did: _announce sat inside the try, so a NameError in it was caught by
    `except Exception` and recorded as "av failed", silently routing the whole
    corpus through audioread.
    """
    import spoof_superb.scoring.audio as A
    _, bad = _corpus(n=2)
    if not have_pyav():
        pytest.skip("PyAV not installed")

    def boom(*a, **k):
        raise NameError("deliberate bug in the reporting path")

    monkeypatch.setattr(A, "_announce", boom)
    stats = {}
    with pytest.raises(NameError):
        A.load_wave(bad[0], 16000, _stats=stats)
    # It counted pyav before announcing: the decode did succeed, and the error
    # propagated instead of being laundered into a fallback.
    assert stats.get("pyav") == 1
    assert "audioread" not in stats
