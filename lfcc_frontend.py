"""
lfcc_frontend.py
----------------
Self-contained LFCC front-end for the LFCC-GMM baseline.

Ported from the reference implementation in /home/alhashim/Rob-ASD/ASD_ML
(Feature_Library/LFCC_pipeline.py::lfcc + feature_functions.py::extract_lfcc,
lfccDeltas), which in turn depends on `spafe`. `spafe` is NOT installed in the
spoof_SUPERB environment, so the three spafe helpers actually used
(framing, windowing, linear_filter_banks) are reimplemented here in numpy.

Why reimplement rather than add the dependency:
  - Only ~40 lines of spafe are on the LFCC path; adding a package to pin the
    whole benchmark against is a worse trade than vendoring those lines.
  - Bit-exactness is verifiable: tests/test_lfcc_frontend.py compares this
    module against spafe 0.3.3 (available in the SER env) on real ASV19 audio.

Fidelity note (deliberate, see humanpending.md):
  The reference calls lfcc() with num_ceps=20, nfilts=70, low_freq=0,
  high_freq=4000 and the pipeline defaults win_len=0.030 / win_hop=0.015 /
  nfft=1024 / pre_emph=0. Those defaults are preserved verbatim here so this
  baseline reproduces the author's established LFCC-GMM system rather than a
  differently-parameterised one.

Output: (num_frames, 60) float array = 20 LFCC + delta + delta-delta.
"""

from functools import lru_cache

import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter


# --------------------------------------------------------------------------
# spafe ports (spafe 0.3.3, BSD 3-Clause, Ayoub Malek)
# --------------------------------------------------------------------------

def _stride_trick(a, stride_length, stride_step):
    """spafe.utils.preprocessing.stride_trick"""
    a = np.array(a)
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, stride_length), strides=(stride_step * n, n)
    )


def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """spafe.utils.preprocessing.framing"""
    if win_len < win_hop:
        raise ValueError("win_len must be >= win_hop")

    frame_length = int(win_len * fs)
    frame_step = int(win_hop * fs)

    frames = _stride_trick(sig, frame_length, frame_step)

    # Kept for parity with spafe. Note the strided view makes the branch a
    # no-op in practice (every row already has frame_length samples); it is
    # retained so behaviour cannot silently diverge from the reference.
    if len(frames[-1]) < frame_length:
        frames[-1] = np.append(
            frames[-1], np.array([0] * (frame_length - len(frames[0])))
        )

    return frames, frame_length


def windowing(frames, frame_len, win_type="hamming"):
    """spafe.utils.preprocessing.windowing"""
    return {
        "hanning": np.hanning,
        "bartlet": np.bartlett,
        "blackman": np.blackman,
        "hamming": np.hamming,
    }[win_type](frame_len) * frames


def linear_filter_banks(nfilts=24, nfft=512, fs=16000, low_freq=0,
                        high_freq=None, scale="constant"):
    """spafe.fbanks.linear_fbanks.linear_filter_banks (fb_type='lin' path).

    Returns (fbank, center_freqs) with fbank of shape (nfilts, nfft//2 + 1).

    Faithful to spafe: the filter edges are laid out on
    np.linspace(low_freq, high_freq, nfft//2+1), i.e. the bin axis is stretched
    onto [low_freq, high_freq] rather than onto the true rfft bin frequencies.
    That is spafe's behaviour and the reference LFCC-GMM was trained under it,
    so it is reproduced exactly rather than "fixed".
    """
    high_freq = high_freq or fs / 2
    if low_freq < 0:
        raise ValueError("low_freq must be >= 0")
    if high_freq > (fs / 2):
        raise ValueError("high_freq must be <= fs/2")

    delta_hz = abs(high_freq - low_freq) / (nfilts + 1)
    scale_freqs = low_freq + delta_hz * np.arange(0, nfilts + 2)
    lower_edges_hz = scale_freqs[:-2]
    upper_edges_hz = scale_freqs[2:]
    center_freqs_hz = scale_freqs[1:-1]

    freqs = np.linspace(low_freq, high_freq, nfft // 2 + 1)
    fbank = np.zeros((nfilts, nfft // 2 + 1))

    for j, (center, lower, upper) in enumerate(
        zip(center_freqs_hz, lower_edges_hz, upper_edges_hz)
    ):
        left_slope = (freqs >= lower) == (freqs <= center)
        fbank[j, left_slope] = (freqs[left_slope] - lower) / (center - lower)
        right_slope = (freqs >= center) == (freqs <= upper)
        fbank[j, right_slope] = (upper - freqs[right_slope]) / (upper - center)

    scaling = {
        "ascendant": np.array([i / nfilts for i in range(1, nfilts + 1)]).reshape(nfilts, 1),
        "descendant": np.array([i / nfilts for i in range(nfilts, 0, -1)]).reshape(nfilts, 1),
        "constant": np.ones(shape=(nfilts, 1)),
    }[scale]

    fbank = fbank * scaling
    return np.abs(fbank), center_freqs_hz


@lru_cache(maxsize=8)
def _cached_filter_banks(nfilts, nfft, fs, low_freq, high_freq, scale):
    """Memoised linear_filter_banks.

    The filterbank depends only on scalar config, but rebuilding it per
    utterance costs ~1.6 ms of the ~6.5 ms LFCC budget (25%). The returned
    array is marked read-only so a caller cannot corrupt the shared cache.
    """
    fbank, _ = linear_filter_banks(nfilts=nfilts, nfft=nfft, fs=fs,
                                   low_freq=low_freq, high_freq=high_freq,
                                   scale=scale)
    fbank.flags.writeable = False
    return fbank


def pre_emphasis(sig, pre_emph_coeff=0.97):
    """spafe.utils.preprocessing.pre_emphasis"""
    return lfilter([1, -pre_emph_coeff], [1], sig)


# --------------------------------------------------------------------------
# LFCC pipeline (port of Rob-ASD/ASD_ML/Feature_Library/LFCC_pipeline.py)
# --------------------------------------------------------------------------

def lfcc(sig, fs=16000, num_ceps=20, pre_emph=0, pre_emph_coeff=0.97,
         win_len=0.030, win_hop=0.015, win_type="hamming", nfilts=70,
         nfft=1024, low_freq=None, high_freq=None, scale="constant",
         dct_type=2, normalize=0):
    """Linear-frequency cepstral coefficients. Returns (num_frames, num_ceps)."""
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    if low_freq < 0:
        raise ValueError("low_freq must be >= 0")
    if high_freq > (fs / 2):
        raise ValueError("high_freq must be <= fs/2")
    if nfilts < num_ceps:
        raise ValueError("nfilts must be >= num_ceps")

    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    frames, frame_length = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)
    windows = windowing(frames=frames, frame_len=frame_length, win_type=win_type)

    fourrier_transform = np.fft.rfft(windows, nfft)
    abs_fft_values = np.abs(fourrier_transform) ** 2

    linear_fbanks_mat = _cached_filter_banks(
        nfilts, nfft, fs, low_freq, high_freq, scale)

    features = np.dot(abs_fft_values, linear_fbanks_mat.T)
    log_features = np.log10(features + 2.2204e-16)

    return dct(log_features, type=dct_type, norm="ortho", axis=1)[:, :num_ceps]


def lfcc_deltas(x, width=3):
    """Port of feature_functions.py::lfccDeltas. x is (n_ceps, n_frames)."""
    hlen = int(np.floor(width / 2))
    win = list(range(hlen, -hlen - 1, -1))
    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen * 2:]


def extract_lfcc(audio_data, sr, num_ceps=20, order_deltas=2, no_filters=70,
                 low_freq=0, high_freq=4000):
    """Port of feature_functions.py::extract_lfcc.

    Returns (num_frames, num_ceps * (1 + order_deltas)) float64.
    With the reference defaults that is (num_frames, 60).
    """
    lfccs = lfcc(sig=audio_data, fs=sr, num_ceps=num_ceps, nfilts=no_filters,
                 low_freq=low_freq, high_freq=high_freq).T

    if order_deltas > 0:
        feats = [lfccs]
        for _ in range(order_deltas):
            feats.append(lfcc_deltas(feats[-1]))
        lfccs = np.vstack(feats)

    return lfccs.T
