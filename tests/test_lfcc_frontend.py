"""
test_lfcc_frontend.py
---------------------
Contract: lfcc_frontend.extract_lfcc must reproduce the reference LFCC-GMM
front-end (/home/alhashim/Rob-ASD/ASD_ML, spafe-backed) bit-for-bit on real
ASVspoof2019 audio.

This is the contract that makes the vendored spafe port legitimate. If it
fails, every LFCC-GMM score in the benchmark is computed on different features
than the reference system and is not comparable to the author's prior results.

Mechanism: `spafe` is not installed in the spoof_SUPERB env but IS installed in
the SER env (spafe 0.3.3). This test shells out to the SER interpreter to
compute reference features with the genuine spafe code path, then compares
against the local port in-process.

External dependencies (SER interpreter, Rob-ASD checkout, ASV19 audio) are
SKIP conditions, not failures: on a machine without them this contract cannot
be evaluated either way, and a red test there would be noise rather than
signal.

Run:  pytest tests/test_lfcc_frontend.py    (or: python tests/test_lfcc_frontend.py)
"""

import os
import subprocess
import tempfile

import numpy as np
import pytest

from lfcc_frontend import extract_lfcc  # noqa: E402

SER_PYTHON = "/home/alhashim/.conda/envs/SER/bin/python"
ASD_ML_DIR = "/home/alhashim/Rob-ASD/ASD_ML"
FLAC_DIR = "/data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"

N_UTTERANCES = 5

# Tolerance: the two environments differ in numpy (2.2.6 vs 1.23.5) and scipy
# (1.15.3 vs 1.10.1), so exact equality is not required -- but anything above
# float64 round-off means the port diverges algorithmically.
ATOL = 1e-9
RTOL = 1e-9

REFERENCE_SCRIPT = r'''
import importlib.util, sys, os
import numpy as np
import librosa
from scipy.signal import lfilter

ASD_ML_DIR = sys.argv[1]
out_npz = sys.argv[2]
files = sys.argv[3:]

# Load the genuine reference LFCC pipeline (spafe-backed) directly from file,
# bypassing Feature_Library/__init__ so the CQCC imports are not triggered.
spec = importlib.util.spec_from_file_location(
    "ref_lfcc_pipeline", os.path.join(ASD_ML_DIR, "Feature_Library", "LFCC_pipeline.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

# Verbatim copy of Rob-ASD/ASD_ML/feature_functions.py::lfccDeltas
def lfccDeltas(x, width=3):
    hlen = int(np.floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]

# Verbatim copy of Rob-ASD/ASD_ML/feature_functions.py::extract_lfcc
def extract_lfcc(audio_data, sr, num_ceps=20, order_deltas=2, no_Filters=70):
    lfccs = ref.lfcc(sig=audio_data, fs=sr, num_ceps=num_ceps, nfilts=no_Filters,
                     low_freq=0, high_freq=4000).T
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(lfccDeltas(feats[-1]))
        lfccs = np.vstack(feats)
    return lfccs.T

out = {}
for f in files:
    data, sr = librosa.load(f, sr=None)
    out[os.path.basename(f)] = extract_lfcc(data, sr)
np.savez(out_npz, **out)
print("REFERENCE_OK", len(out), "spafe", __import__("spafe").__version__ if hasattr(__import__("spafe"), "__version__") else "?")
'''


@pytest.fixture(scope="module")
def reference_features():
    """(flac_names, {name: reference_lfcc}) computed by the genuine spafe path."""
    if not os.path.isfile(SER_PYTHON):
        pytest.skip(f"reference interpreter not found: {SER_PYTHON}")
    if not os.path.isdir(ASD_ML_DIR):
        pytest.skip(f"reference implementation not found: {ASD_ML_DIR}")
    if not os.path.isdir(FLAC_DIR):
        pytest.skip(f"ASV19 train audio not found: {FLAC_DIR}")

    flacs = sorted(f for f in os.listdir(FLAC_DIR) if f.endswith(".flac"))[:N_UTTERANCES]
    if not flacs:
        pytest.skip(f"no audio under {FLAC_DIR}")
    paths = [os.path.join(FLAC_DIR, f) for f in flacs]

    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "ref.py")
        npz = os.path.join(td, "ref.npz")
        with open(script, "w") as fh:
            fh.write(REFERENCE_SCRIPT)

        proc = subprocess.run(
            [SER_PYTHON, script, ASD_ML_DIR, npz] + paths,
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            "reference generation failed\n"
            f"stdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        )
        loaded = np.load(npz)
        return flacs, {name: loaded[name] for name in flacs}


def test_local_lfcc_matches_spafe_reference(reference_features):
    import librosa

    flacs, ref = reference_features
    mismatches = []
    for name in flacs:
        data, sr = librosa.load(os.path.join(FLAC_DIR, name), sr=None)
        got = extract_lfcc(data, sr)
        want = ref[name]

        if got.shape != want.shape:
            mismatches.append(f"{name}: shape {got.shape} != reference {want.shape}")
            continue
        if not np.allclose(got, want, atol=ATOL, rtol=RTOL):
            mismatches.append(
                f"{name}: max|diff| = {np.max(np.abs(got - want)):.3e} (atol={ATOL})"
            )

    assert not mismatches, (
        f"{len(mismatches)}/{len(flacs)} utterances diverge from the reference "
        f"front-end:\n  " + "\n  ".join(mismatches)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
