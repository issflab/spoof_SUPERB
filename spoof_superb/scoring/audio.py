"""One audio loader, and the reason it is not just ``librosa.load``.

libsndfile 1.2.2 cannot decode a large fraction of the ASVspoof2021 FLAC files:

    asvspoof2021_LA    43% refused
    asvspoof2021_DF    36% refused
    deepfake_eval_2024  9% refused
    every other corpus   0%

with three errors -- "unknown error in flac decoder", "flac decoder lost sync",
"Internal psf_fseek() failed". The files are not malformed: their STREAMINFO,
metadata blocks and frame headers are indistinguishable from the ones that read
fine, ffmpeg decodes them without complaint, and remuxing them (``-c:a copy``)
does not help. It is a decoder bug, not a data defect.

``librosa.load`` hides it by falling back to audioread, which spawns a
subprocess per file. That is correct and roughly 50-90x slower:

    soundfile, in process           0.8 ms/file
    PyAV (libav), in process        1.7 ms/file
    ffmpeg, one subprocess/file    63   ms/file
    librosa -> audioread           65-82 ms/file   <- what runs today

On ASVspoof2021-DF that is the difference between decode costing ~2 minutes and
~50 minutes per model, per run, before any contention -- and DF is scored once
per SSL upstream.

So: soundfile first, PyAV for what it refuses, audioread only if PyAV is not
installed. FLAC is lossless, so a correct decoder is a correct decoder; that all
three agree is not assumed but checked -- see tests/test_audio.py, which asserts
bit-identical output across the three paths on files libsndfile refuses.

PyAV is optional. Without it this module behaves exactly as before, slowly.
"""

import os

import numpy as np

__all__ = ["load_wave", "have_pyav", "DECODERS"]

DECODERS = ("soundfile", "pyav", "audioread")


def have_pyav():
    try:
        import av  # noqa: F401
        return True
    except Exception:
        return False


def _to_mono_float32(x):
    """librosa's own convention: float32, channels averaged."""
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def _resample(x, orig_sr, target_sr):
    # target_sr=None means "keep the file's own rate", which is what the
    # LFCC-GMM reference does (librosa.load(file, sr=None)).
    if target_sr is None or orig_sr == target_sr:
        return x
    import librosa
    return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)


def _load_soundfile(path, sr):
    import soundfile as sf
    x, file_sr = sf.read(path, dtype="float32", always_2d=False)
    return _resample(_to_mono_float32(x), file_sr, sr)


def _load_pyav(path, sr):
    """In-process libav decode -- the same decoder ffmpeg uses, without the fork.

    libav does the sample-format and channel-layout conversion, because
    ``to_ndarray`` returns whatever the codec produced: planar or packed,
    int16 or float, interleaved for packed multi-channel. Reducing that by hand
    is where this went wrong once already -- averaging channels before scaling
    promoted int16 to float64, skipped the /32768, and returned samples 32768x
    too large, which test_a5 caught.

    The resampler is pinned to the file's own rate so it only converts format.
    Rate conversion stays with librosa, so a resampled corpus keeps using the
    same resampler it always did.
    """
    import av

    with av.open(path) as container:
        stream = container.streams.audio[0]
        file_sr = stream.rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=file_sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):      # flush
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError(f"no audio frames decoded: {path}")
    # fltp mono: (1, n) float32 already in [-1, 1].
    x = np.concatenate(chunks, axis=1).ravel().astype(np.float32)
    return _resample(x, file_sr, sr)


def _load_audioread(path, sr):
    import librosa
    x, _ = librosa.load(path, sr=sr)
    return x


def native_rate(path):
    """The file's own sample rate, without decoding it."""
    import soundfile as sf
    try:
        return sf.info(path).samplerate
    except Exception:
        import av
        with av.open(path) as c:
            return c.streams.audio[0].rate


_announced = set()


def _announce(decoder, path, why):
    """Say once per process that a decoder is in use, and why.

    This exists because of a real hour lost: a sweep silently decoded through
    audioread despite ``av`` being installed, and the only evidence in the log
    was librosa's own "PySoundFile failed" -- which names the decoder that
    *failed*, never the one that ran or the reason the fallback happened.
    Diagnosing it needed a separate reproduction.

    Once per (decoder, reason) per process: enough to see which path a run took
    in its log, without reprinting for 611,829 files.
    """
    key = (decoder, str(why)[:60])
    if key in _announced:
        return
    _announced.add(key)
    print(f"  [audio] using {decoder} for {os.path.basename(path)} "
          f"and files like it: {why}", flush=True)


def load_wave(path, sr=16000, _stats=None):
    """Mono float32 at ``sr``, by whichever decoder can read the file.

    Raises only when every decoder fails, so the caller's missing-audio policy
    still sees a genuinely unreadable file rather than an install problem.
    """
    def count(decoder):
        if _stats is not None:
            _stats[decoder] = _stats.get(decoder, 0) + 1

    # try/except/else throughout: the bookkeeping lives in `else`, which the
    # except clauses do not cover. Putting it inside `try` made a NameError in
    # the reporting code look like a decoder failure, and silently demoted
    # every file to the slow path.
    try:
        x = _load_soundfile(path, sr)
    except Exception as sf_err:
        first = sf_err
    else:
        count("soundfile")
        return x

    try:
        x = _load_pyav(path, sr)
    except ImportError:
        pyav_err = ("av is not installed -- pip install av==17.1.0 "
                    "for a ~35x faster fallback")
    except Exception as exc:
        pyav_err = f"av failed too ({type(exc).__name__}: {exc})"
    else:
        count("pyav")
        _announce("pyav", path, f"libsndfile refused it ({first})")
        return x

    try:
        x = _load_audioread(path, sr)
    except Exception:
        raise first
    else:
        count("audioread")
        # Deliberately loud: this path is ~35x slower and avoiding it is the
        # entire reason av is a dependency. A run that lands here must say so.
        _announce("audioread (SLOW)", path, pyav_err)
        return x
