"""Scoring back-ends: (items, checkpoint) -> [(utt_id, score)].

Three back-ends, one signature. Each returns the scored rows plus a count of
rows dropped because the audio could not be decoded.

The score is always the class-1 logit, exactly as main.py::produce_evaluation
indexes it (``batch_out[:, 1]``).

Precision policy: fp32 unless ``amp=True`` is passed explicitly. fp16 autocast
was the cause of the 384,157 half-precision overflow NaN per model that
verify_noise_rerun.py documents for the masked-spectrogram front-ends (tera,
mockingjay, mockingjay_960hr, audio_albert_960hr), so it is opt-in and never a
default.
"""

import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from spoof_superb.scoring.datasets import CROP

__all__ = ["WavDataset", "score_linear_head", "score_aasist_raw", "score_lfcc_gmm"]

# How often tqdm may redraw. These loops write to a log file, not a terminal,
# so the interval sets both the log's growth rate and how stale the
# orchestrator's progress display can be -- it reads this same counter back out
# of the log. 10s is ~1k updates on a three-hour column: negligible on disk,
# responsive enough to watch.
PROGRESS_INTERVAL_S = 10.0


def pad(x, max_len=CROP):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


class WavDataset(Dataset):
    """(utt_id, path) -> (waveform, utt_id, ok).

    The ``ok`` flag is the second half of the missing-audio policy: the driver
    pre-filters ids whose file does not exist, and this catches the ones that
    exist but do not decode. One undecodable file must not kill a multi-hour
    run, so the row is dropped rather than scored and the count is reported.
    """

    def __init__(self, items, cut=CROP, sr=16000):
        self.items = items          # list of (utt_id, path)
        self.cut = cut
        self.sr = sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from spoof_superb.scoring.audio import load_wave  # kept out of the fork
        utt, path = self.items[i]
        try:
            return Tensor(pad(load_wave(path, self.sr), self.cut)), utt, True
        except Exception:
            return Tensor(np.zeros(self.cut, dtype=np.float32)), utt, False


def _run_torch_loop(model, items, device, batch_size, num_workers, amp, desc):
    loader = DataLoader(WavDataset(items), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False, drop_last=False)

    use_amp = amp and str(device).startswith("cuda")
    out, n_bad = [], 0
    with torch.no_grad():
        for batch_x, utt, ok in tqdm(loader, desc=desc, mininterval=PROGRESS_INTERVAL_S):
            batch_x = batch_x.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(batch_x)
            # .float() before .numpy(): under autocast the logits are fp16.
            scores = logits[:, 1].float().cpu().numpy().ravel().tolist()
            ok_list = ok.tolist() if hasattr(ok, "tolist") else list(ok)
            for u, s, good in zip(utt, scores, ok_list):
                if good:
                    out.append((u, s))
                else:
                    n_bad += 1
    return out, n_bad


#: The tensors the linear-head recipe actually trains. Everything else in a
#: full checkpoint is the frozen s3prl upstream, which the model has already
#: loaded by the time we get here.
_TRAINED_KEYS = (
    "ssl_model.featurizer.weights",
    "projector.weight",
    "projector.bias",
    "post_net.linear.weight",
    "post_net.linear.bias",
)


def _load_linear_head_state(model, model_path, device):
    """Load either a full checkpoint or a downstream-only one.

    Two checkpoint shapes exist. ``swa.pth`` as written by training is the whole
    UtteranceLevel module, frozen upstream included -- roughly 1.2 GB for a large
    encoder, of which about 1 MB is trained. The published checkpoints carry only
    the trained tensors, because the upstream is frozen and s3prl already serves
    it byte-identically.

    Both are accepted, and neither is accepted loosely: a downstream-only file
    must contain every trained tensor and nothing the model does not expect, and
    the only keys it is allowed to leave unfilled are upstream ones the frozen
    encoder has already supplied.
    """
    sd = torch.load(model_path, map_location=device)
    if any(k.startswith("ssl_model.model.") for k in sd):
        model.load_state_dict(sd, strict=True)
        return "full"

    missing_trained = [k for k in _TRAINED_KEYS if k not in sd]
    if missing_trained:
        raise RuntimeError(
            f"{model_path} looks like a downstream-only checkpoint but is missing "
            f"trained tensors: {missing_trained}"
        )
    result = model.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"{model_path} carries tensors this model does not define: "
            f"{result.unexpected_keys}"
        )
    left = [k for k in result.missing_keys if not k.startswith("ssl_model.model.")]
    if left:
        raise RuntimeError(
            f"{model_path} left non-upstream tensors unfilled: {left}"
        )
    return "downstream-only"


def score_linear_head(items, model_path, device, ssl_model,
                      batch_size=32, num_workers=6, amp=False):
    """The SSL linear head (UtteranceLevel) over an s3prl upstream.

    ``args.ssl_feature`` is read by the model but was never assigned anywhere in
    the repo; it is set to ``ssl_model`` explicitly here, as both standalone
    drivers did.
    """
    from spoof_superb.models.linear_head import UtteranceLevel as LinearHead

    args = SimpleNamespace(ssl_feature=ssl_model, ssl_model=ssl_model)
    model = LinearHead(args, device).to(device)
    _load_linear_head_state(model, model_path, device)
    model.eval()
    print(f"  model loaded ({sum(p.numel() for p in model.parameters())} params) "
          f"<- {model_path}  amp={amp}", flush=True)

    return _run_torch_loop(model, items, device, batch_size, num_workers, amp, ssl_model)


def score_aasist_raw(items, model_path, device, batch_size=64, num_workers=6, amp=False):
    from spoof_superb.models.aasist_raw import Model as AasistRaw

    model = AasistRaw(args=None, device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    print(f"  model loaded ({sum(p.numel() for p in model.parameters())} params) "
          f"<- {model_path}", flush=True)

    return _run_torch_loop(model, items, device, batch_size, num_workers, amp, "aasist_raw")


# ===========================================================================
# LFCC-GMM (no torch, no GPU: EM over two diagonal GMMs)
# ===========================================================================

_GMM = {}


def _gmm_init(model_dir):
    from spoof_superb.models.lfcc_gmm import limit_blas_threads, load_gmm
    limit_blas_threads(1)   # see lfcc_gmm.limit_blas_threads: 58x on this host
    _GMM["bona"] = load_gmm(os.path.join(model_dir, "bonafide", "gmm_final.pkl"))
    _GMM["spoof"] = load_gmm(os.path.join(model_dir, "spoof", "gmm_final.pkl"))


def _gmm_score_one(item):
    from spoof_superb.models.lfcc_gmm import llr_score, load_lfcc
    utt, path = item
    try:
        Tx = load_lfcc(path)
        if Tx.shape[0] == 0:
            return utt, None
        return utt, llr_score(_GMM["bona"], _GMM["spoof"], Tx)
    except Exception:
        return utt, None


def score_lfcc_gmm(items, model_dir, n_jobs=16):
    """Extract LFCCs and score the LLR in one parallel pass.

    Scores the FULL utterance (no 4 s crop) -- GaussianMixture.score returns
    the mean per-frame log-likelihood, so the LLR is length-normalised. This
    matches the reference implementation; the crop is an AASIST-side choice.
    """
    from multiprocessing import Pool

    for c in ("bonafide", "spoof"):
        p = os.path.join(model_dir, c, "gmm_final.pkl")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing trained GMM: {p}")
    print(f"  GMMs <- {model_dir}", flush=True)

    out, n_bad = [], 0
    with Pool(processes=n_jobs, initializer=_gmm_init, initargs=(model_dir,)) as pool:
        for utt, score in tqdm(pool.imap(_gmm_score_one, items, chunksize=64),
                               total=len(items), desc="lfcc_gmm",
                               mininterval=PROGRESS_INTERVAL_S):
            if score is None:
                n_bad += 1
            else:
                out.append((utt, score))
    return out, n_bad
