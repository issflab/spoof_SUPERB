"""
lfcc_gmm.py
-----------
LFCC-GMM baseline: two 512-component diagonal-covariance GMMs (bonafide,
spoof) over 60-dim LFCC features, scored as a log-likelihood ratio.

Ported from /home/alhashim/Rob-ASD/ASD_ML/gmm_asvspoof.py. The algorithm is
preserved exactly:
  - init: sklearn GaussianMixture(diag, max_iter=100).fit() on every Nth file
  - then: manual streaming EM (E-step accumulates nk/mu/sigma, M-step in closed
    form, reg_covar added), iterating until |delta log-likelihood| < tol
  - score: gmm_bona.score(X) - gmm_spoof.score(X)   [mean log-lik per frame]

Two deliberate deviations from the reference (both verified equivalent, see
humanpending.md):
  1. Features are held in RAM instead of pickled per-utterance to disk. The
     reference caches because it re-reads features on every EM iteration; /data
     has ~244 GB free and caching 10 eval sets would be reckless. ASV19 LA
     train is ~1.5 GB as float32, so RAM is strictly better and much faster.
  2. The E-step runs over frame CHUNKS rather than one utterance at a time.
     Identical arithmetic (the accumulators are additive), far fewer Python
     iterations. Chunking also bounds the responsibility matrix, which would be
     ~23 GB if the whole spoof set were done in one call.

This module is CPU-only by design: EM over diagonal GMMs is a BLAS workload
with no GPU path in the reference implementation.
"""

import os
import pickle
import time

import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

from spoof_superb.frontends.lfcc import extract_lfcc


def limit_blas_threads(n=1):
    """Pin BLAS to n threads inside the current process.

    The LFCC/GMM path is parallelised with a process Pool, so each worker
    inheriting a full-machine BLAS thread pool oversubscribes catastrophically
    (16 workers x 32 threads on 32 cores). Measured on this host: scoring ran
    at 20 utt/s unpinned vs 1160 utt/s pinned -- a 58x difference, i.e. 42 h vs
    0.7 h for the 3.07M-utterance benchmark. Called from every worker
    initializer; threadpoolctl works even when the library is already loaded,
    which env vars cannot fix once numpy has been imported.
    """
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(n)
    except Exception:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, str(n))

N_COMPONENTS = 512          # reference: config.n_comp = 512
COVARIANCE_TYPE = "diag"
INIT_STRIDE = 20            # reference: every 20th file for the init fit
EM_CHUNK_FRAMES = 100_000   # bounds the (chunk x n_components) resp matrix


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def load_lfcc(path, sr=None):
    """Extract (n_frames, 60) float32 LFCC for one audio file.

    sr=None preserves the file's native rate, exactly as the reference
    (librosa.load(file, sr=None)) does. No padding or tiling is applied: the
    reference LFCC-GMM scores whole utterances at their native length.

    Utterances shorter than one 30 ms analysis window yield zero frames and are
    therefore dropped by the caller rather than scored. This is deliberate
    (author decision, 2026-07-27). The affected files are not short speech: they
    are 322 failed generations in Famous Figures (10-20 ms, peak amplitude
    ~6e-05, i.e. inaudible; 320 of them from LLASA) and 2 in MLAAD. Tiling them
    up to a scoreable length was implemented and then reverted, because it
    manufactures a score from a fragment that carries no usable signal. The
    consequence is recorded in humanpending.md: the LFCC-GMM column for those
    two datasets covers marginally fewer trials than the SSL columns, which
    score these files by tiling them to 4 s.
    """
    from spoof_superb.scoring.audio import load_wave, native_rate
    # Same three-decoder policy as the SSL path: 43% of ASVspoof2021-LA is
    # unreadable by libsndfile, and this back-end scores it too.
    data = load_wave(path, sr=sr)
    samplerate = native_rate(path) if sr is None else sr
    return extract_lfcc(data, samplerate).astype(np.float32)


def _extract_one(args):
    path, sr = args
    try:
        return load_lfcc(path, sr), None
    except Exception as exc:  # unreadable/corrupt audio must not kill a long run
        return None, f"{path}: {type(exc).__name__}: {exc}"


def _extract_init():
    limit_blas_threads(1)


def extract_many(paths, n_jobs=8, sr=None, desc="features"):
    """Extract LFCCs for many files in parallel.

    Returns (feats, errors) where feats is a list aligned with `paths` (None
    for files that failed) and errors is a list of message strings.
    """
    from multiprocessing import Pool
    from tqdm import tqdm

    feats = [None] * len(paths)
    errors = []
    args = [(p, sr) for p in paths]

    with Pool(processes=n_jobs, initializer=_extract_init) as pool:
        for i, (feat, err) in enumerate(
            tqdm(pool.imap(_extract_one, args, chunksize=32),
                 total=len(paths), desc=desc, mininterval=10.0)
        ):
            feats[i] = feat
            if err:
                errors.append(err)

    return feats, errors


# --------------------------------------------------------------------------
# GMM persistence
# --------------------------------------------------------------------------

def save_gmm(gmm, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        pickle.dump(gmm._get_parameters(), f)
    os.replace(tmp, path)


def load_gmm(path, n_features=60):
    """Rebuild a GaussianMixture from reference-format pickled parameters."""
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE)
    with open(path, "rb") as f:
        params = pickle.load(f)
    gmm._set_parameters(params)
    # _set_parameters does not restore these, but score()/predict() need them.
    gmm.n_components = gmm.means_.shape[0]
    gmm.n_features_in_ = gmm.means_.shape[1]
    return gmm


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _iter_chunks(feats, chunk_frames=EM_CHUNK_FRAMES):
    """Yield float32 frame blocks of <= chunk_frames rows from a list of arrays."""
    buf, n = [], 0
    for Tx in feats:
        if Tx is None or Tx.shape[0] == 0:
            continue
        buf.append(Tx)
        n += Tx.shape[0]
        if n >= chunk_frames:
            yield np.vstack(buf)
            buf, n = [], 0
    if buf:
        yield np.vstack(buf)


def effective_init_stride(n_utts, init_stride=INIT_STRIDE, init_min_utts=1000):
    """Stride for the init fit, floored so the init subset stays large enough.

    The reference uses a flat every-20th-file stride, but it trains on a pooled
    train+dev+laundered set where that still leaves thousands of utterances per
    class. Here the training set is ASV19 LA train alone, whose bonafide class
    is only 2,580 utterances -- a flat stride of 20 would fit 512 components to
    129 utterances (~55 frames per component), which collapses components and
    raises "ill-defined empirical covariance".

    So the stride is capped such that the init sees at least init_min_utts
    utterances, keeping both classes at a comparable, sane init size
    (bonafide 1,290 / spoof 1,140) instead of 129 / 1,140.
    """
    if n_utts <= init_min_utts:
        return 1
    return max(1, min(init_stride, n_utts // init_min_utts))


def train_gmm(feats, save_dir, ncomp=N_COMPONENTS, init_stride=INIT_STRIDE,
              max_em_iters=1000, seed=None, log=print, init_min_utts=1000):
    """Train one class-conditional GMM. Returns the fitted GaussianMixture.

    feats        : list of (n_frames, n_dims) float32 arrays (None entries skipped)
    save_dir     : directory for init_partial.pkl / <i>_partial.pkl / gmm_final.pkl
    init_stride  : use every Nth utterance for the sklearn init fit (reference: 20)

    Checkpoints per iteration so an interrupted run resumes where it stopped,
    exactly like the reference.
    """
    os.makedirs(save_dir, exist_ok=True)
    feats = [f for f in feats if f is not None and f.shape[0] > 0]
    if not feats:
        raise ValueError("no usable features supplied")

    final_file = os.path.join(save_dir, "gmm_final.pkl")
    if os.path.isfile(final_file):
        log(f"  {final_file} already exists -> loading, skipping training")
        return load_gmm(final_file)

    # ---- init -------------------------------------------------------------
    init_file = os.path.join(save_dir, "init_partial.pkl")
    if os.path.isfile(init_file):
        log(f"  init checkpoint found -> {init_file}")
        gmm = GaussianMixture(n_components=ncomp, covariance_type=COVARIANCE_TYPE)
        with open(init_file, "rb") as tf:
            gmm._set_parameters(pickle.load(tf))
        gmm.n_components = gmm.means_.shape[0]
        gmm.n_features_in_ = gmm.means_.shape[1]
    else:
        stride = effective_init_stride(len(feats), init_stride, init_min_utts)
        subset = feats[::stride]
        X = np.vstack(subset).astype(np.float32)
        log(f"  init fit on {len(subset)} utts (stride {stride}, "
            f"requested {init_stride}) -> X {X.shape}")
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=ncomp,
            random_state=seed,
            covariance_type=COVARIANCE_TYPE,
            max_iter=100,
            verbose=2,
            verbose_interval=10,
        ).fit(X)
        log(f"  init done in {time.time()-t0:.1f}s  lower_bound={gmm.lower_bound_:.5f}")
        save_gmm(gmm, init_file)
        del X, subset

    # ---- streaming EM -----------------------------------------------------
    prev_lower_bound = -np.inf
    for i in range(max_em_iters):
        ckpt = os.path.join(save_dir, f"{i}_partial.pkl")
        if os.path.isfile(ckpt):
            with open(ckpt, "rb") as tf:
                gmm._set_parameters(pickle.load(tf))
            log(f"  iter {i}: checkpoint exists, restored")
            continue

        t0 = time.time()
        nk_acc = np.zeros_like(gmm.weights_)
        mu_acc = np.zeros_like(gmm.means_)
        sigma_acc = np.zeros_like(gmm.covariances_)
        log_prob_norm_acc = 0.0
        n_samples = 0

        for Tx in _iter_chunks(feats):
            n_samples += Tx.shape[0]

            # E-step
            weighted_log_prob = gmm._estimate_weighted_log_prob(Tx)
            log_prob_norm = logsumexp(weighted_log_prob, axis=1)
            with np.errstate(under="ignore"):
                log_resp = weighted_log_prob - log_prob_norm[:, None]
            log_prob_norm_acc += log_prob_norm.sum()

            # M-step accumulation
            resp = np.exp(log_resp)
            nk_acc += resp.sum(axis=0) + 10 * np.finfo(np.log(1).dtype).eps
            mu_acc += resp.T @ Tx
            sigma_acc += resp.T @ (Tx ** 2)

        # M-step
        gmm.means_ = mu_acc / nk_acc[:, None]
        gmm.covariances_ = sigma_acc / nk_acc[:, None] - gmm.means_ ** 2 + gmm.reg_covar
        gmm.weights_ = nk_acc / n_samples
        gmm.weights_ /= gmm.weights_.sum()
        if (gmm.covariances_ <= 0.0).any():
            raise ValueError(
                "ill-defined empirical covariance: a component collapsed. "
                "Re-run with a smaller --init_stride so the init fit sees more data."
            )
        gmm.precisions_cholesky_ = 1.0 / np.sqrt(gmm.covariances_)

        save_gmm(gmm, ckpt)

        lower_bound = log_prob_norm_acc / n_samples
        change = lower_bound - prev_lower_bound
        log(f"  iter {i}: llh={lower_bound:.6f} change={change:.3e} "
            f"frames={n_samples} ({time.time()-t0:.1f}s)")
        prev_lower_bound = lower_bound

        if abs(change) < gmm.tol:
            log(f"  converged at iter {i} (|change| < tol={gmm.tol})")
            gmm.converged_ = True
            break

    save_gmm(gmm, final_file)
    log(f"  saved -> {final_file}")
    return gmm


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def llr_score(gmm_bona, gmm_spoof, Tx):
    """Log-likelihood ratio for one utterance's features.

    Uses GaussianMixture.score, i.e. the MEAN per-frame log-likelihood, so the
    score is utterance-length normalised -- matching the reference.
    """
    Tx = Tx.astype(np.float32)
    return float(gmm_bona.score(Tx) - gmm_spoof.score(Tx))
