"""
eval_baselines.py
-----------------
Score the two non-SSL baselines (aasist_raw, lfcc_gmm) on the 10 benchmark
evaluation sets, emitting score files in the repo's canonical 4-column format.

Why one driver rather than reusing a per-dataset script:
  The repo has exactly three eval entry points -- eval_asvld.py (ASVLD),
  eval_mlaad.py (MLAAD / M-AILABS / SpoofCeleb) and main.py's eval branch,
  which is broken for everything except ITW (it reads cfg.eval_protocol, which
  has no default and no CLI setter, and Dataset_ASVspoof2021_eval hardcodes
  release_in_the_wild/*.wav). Six of the ten sets -- ASV19 LA eval, ASV21 LA,
  ASV21 DF, ASV5, DFEval24, Famous Figures -- have no driver at all. This file
  follows the established standalone-driver pattern of eval_asvld.py /
  eval_mlaad.py (same 4-column output, same --restrict/skip-missing discipline)
  and covers all ten.

TRIAL LISTS COME FROM THE PUBLISHED SSL SCORE FILES.
  For each dataset the eval list and the ground-truth key are read from an
  existing linear_head reference score file rather than re-derived from a raw
  protocol. This is the same --restrict_to convention eval_asvld.py and
  eval_mlaad.py already use, and it is the only way to get baselines that are
  actually comparable to the paper's numbers: several published sets are
  subsets whose selection rule is not recorded anywhere in the repo
  (ASV21 DF is 152,955 of 611,829 protocol rows; ASVLD pools a
  noise x10 / reverb x3 / resample x4 slice; Famous Figures has 346,471 of
  348,135; DFEval24 has 1,976 rows matching neither on-disk metadata file).
  Re-deriving them would silently score a different trial set and produce EERs
  that cannot be placed in the same table as the SSL models.

Output (4 cols, identical to eval_asvld.py / eval_mlaad.py):
    {utt_id} - {key} {score}
where utt_id and key are copied verbatim from the reference file, so score
files line up row-for-row with the SSL ones.

Usage
-----
    python eval_baselines.py --list_datasets
    python eval_baselines.py --model aasist_raw --model_path .../swa.pth \\
        --dataset eval_2019 --output_file .../aasist_raw_eval_2019.txt
    python eval_baselines.py --model lfcc_gmm --model_path .../lfcc_gmm \\
        --dataset wild --output_file .../lfcc_gmm_wild.txt
"""

import argparse
import os
import sys
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA = "/data/Data"
SCORES_ROOT = "/data/ssl_anti_spoofing/asd_superb_score_files"
REFERENCE_DIR = os.path.join(SCORES_ROOT, "linear_head")
DEFAULT_REFERENCE_SSL = "xls_r_300m"

CROP = 64600  # ~4 s at 16 kHz, as in data_utils_SSL.py


# ===========================================================================
# Dataset registry
# ===========================================================================

ASVLD_ROOT = os.path.join(DATA, "ASVSpoofLaunderedDatabase", "ASVspoofLD")
ASVLD_CONDITIONS = ["Noise_Addition", "Reverberation", "Resampling",
                    "Recompression", "Filtering"]
DFEVAL_AUDIO = os.path.join(DATA, "Deepfake_Eval_2024", "audio-data")
FF_NFS_PREFIX = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/"
FF_LOCAL_ROOT = os.path.join(DATA, "famousfigures")


@lru_cache(maxsize=1)
def _asvld_condition_index():
    """utt_id -> condition, parsed from the 5 ASVLD protocols.

    Needed because ASVLD audio lives at {root}/{condition}/flac/{utt}.flac but
    the pooled reference score file carries no condition column.
    """
    index = {}
    proto_dir = os.path.join(ASVLD_ROOT, "protocols")
    for cond in ASVLD_CONDITIONS:
        path = os.path.join(proto_dir, f"ASVspoofLauneredDatabase_{cond}.txt")
        if not os.path.isfile(path):
            print(f"  [WARN] ASVLD protocol missing: {path}")
            continue
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    index[parts[1]] = cond
    print(f"  ASVLD condition index: {len(index)} utt_ids")
    return index


@lru_cache(maxsize=1)
def _dfeval_stem_index():
    """basename-without-extension -> real path.

    DFEval24 score files write every id with a .wav extension, but on disk the
    files are .mp3 / .m4a / .mp4 / .wav. Match on the stem.
    """
    index = {}
    if not os.path.isdir(DFEVAL_AUDIO):
        print(f"  [WARN] DFEval24 audio dir missing: {DFEVAL_AUDIO}")
        return index
    for fn in os.listdir(DFEVAL_AUDIO):
        stem = os.path.splitext(fn)[0]
        index.setdefault(stem, os.path.join(DFEVAL_AUDIO, fn))
    print(f"  DFEval24 stem index: {len(index)} files")
    return index


def _r_asv19(utt):      # utt already carries '.flac'
    return os.path.join(DATA, "ASVSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac", utt)


def _r_asv21_la(utt):
    return os.path.join(DATA, "ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/flac", utt + ".flac")


def _r_asv21_df(utt):
    return os.path.join(DATA, "ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/flac", utt + ".flac")


def _r_asv5(utt):
    return os.path.join(DATA, "ASVSpoof5/No_Laundering_eval/flac", utt + ".flac")


def _r_itw(utt):
    return os.path.join(DATA, "ds_wild/release_in_the_wild", utt + ".wav")


def _r_dfeval(utt):
    return _dfeval_stem_index().get(os.path.splitext(utt)[0])


def _r_famous(utt):
    """Famous Figures: {root}/{Speaker}/{Source}/{name}.wav

    Two rewrites are needed, both verified against the full reference file:
      1. Reference ids are absolute paths under a stale NFS mount that does not
         exist on this host; the same tree is present under /data/Data.
      2. Bonafide rows carry the protocol's empty Source field as the literal
         directory '-', but on disk they live under 'Bonafide'. All 49,945
         '/-/' rows are key=bonafide and all 49,945 resolve after this remap;
         without it the dataset would score zero bonafide trials and its EER
         would be undefined.
    """
    if utt.startswith(FF_NFS_PREFIX):
        rel = utt[len(FF_NFS_PREFIX):]
    elif utt.startswith("/"):
        rel = os.path.relpath(utt, FF_LOCAL_ROOT)
    else:
        rel = utt

    parts = rel.split("/")
    if len(parts) >= 3 and parts[1] == "-":
        parts[1] = "Bonafide"
        rel = "/".join(parts)

    return os.path.join(FF_LOCAL_ROOT, rel)


def _r_spoofceleb(utt):
    return os.path.join(DATA, "SpoofCeleb/flac/evaluation", utt)


def _r_mlaad(utt):      # ids are relative to /data/Data and carry the extension
    return os.path.join(DATA, utt)


def _r_asvld(utt):
    cond = _asvld_condition_index().get(utt)
    if cond is None:
        return None
    return os.path.join(ASVLD_ROOT, cond, "flac", utt + ".flac")


# `ref` is resolved relative to REFERENCE_DIR; `ref_abs` is an absolute template
# for columns whose published source lives outside linear_head/. A list means the
# paper's column is the POOL of those files, and it is assembled in that order.
#
# Sources here must match scripts/recompute_table5_mlaad_v10.py, which is the
# authority for what Table 5 actually reports. Two columns are NOT the obvious
# linear_head/ file:
#   MLAAD  -> the v10 re-run (1,040,006 rows), not legacy linear_head_Multilingual
#             (307,998). Different corpus scale entirely.
#   ASVLD  -> linear_head_asvspoofLD (1,207,509: noise x10, reverb x3, resample x4)
#             POOLED WITH asvld_rerun/Recompression (427,422 = 71,237 x 6 bitrates),
#             folded in by commit 6bf39a0. Reading only the first file silently
#             reproduces a pre-6bf39a0 column.
# SpoofCeleb legacy vs the linear_head_SpoofCeleb re-run were verified to have
# identical utt sets and labels, so either supplies the same trials; the legacy
# path is kept.
DATASETS = {
    "eval_2019":          dict(ref="linear_head_eval_2019_{ssl}.txt",          resolve=_r_asv19),
    "asvspoof2021_LA":    dict(ref="linear_head_asvspoof2021_LA_{ssl}.txt",    resolve=_r_asv21_la),
    "asvspoof2021_DF":    dict(ref="linear_head_asvspoof2021_DF_{ssl}.txt",    resolve=_r_asv21_df),
    "asvspoof5":          dict(ref="linear_head_asvspoof5_{ssl}.txt",          resolve=_r_asv5),
    "deepfake_eval_2024": dict(ref="linear_head_deepfake_eval_2024_{ssl}.txt", resolve=_r_dfeval),
    "wild":               dict(ref="linear_head_wild_{ssl}.txt",               resolve=_r_itw),
    "Famous_Figures":     dict(ref="linear_head_Famous_Figures_{ssl}.txt",     resolve=_r_famous),
    "spoofceleb":         dict(ref="linear_head_spoofceleb_{ssl}.txt",         resolve=_r_spoofceleb),
    "Multilingual":       dict(ref_abs=[os.path.join(
                                  SCORES_ROOT, "linear_head_MLAAD_v10",
                                  "linear_head_MLAAD_v10_{ssl}.txt")],
                               resolve=_r_mlaad),
    "asvspoofLD":         dict(ref_abs=[os.path.join(REFERENCE_DIR,
                                            "linear_head_asvspoofLD_{ssl}.txt"),
                                        os.path.join(
                                  SCORES_ROOT, "asvld_rerun", "Recompression",
                                  "linear_head_Recompression_{ssl}.txt")],
                               resolve=_r_asvld),
}


def reference_paths(dataset, reference_ssl):
    """Absolute reference score file(s) defining a dataset's trial list."""
    spec = DATASETS[dataset]
    if "ref_abs" in spec:
        return [t.format(ssl=reference_ssl) for t in spec["ref_abs"]]
    return [os.path.join(REFERENCE_DIR, spec["ref"].format(ssl=reference_ssl))]


def read_reference(paths):
    """Read one or more 4-column reference score files -> ([utt_id], {utt_id: key}).

    Fields are peeled from the RIGHT (rsplit), never split on whitespace: utt_ids
    legitimately contain spaces. MLAAD v10 has 39,000 such rows (TTS system
    directories like "Cartesia.ai (Sonic-3)"), and a left-split silently yields
    the wrong utt_id AND reads "-" as the label for every one of them. Famous
    Figures utt_ids are absolute paths for the same reason. This matches
    scripts/recompute_table5_mlaad_v10.py::read_legacy.

    Accepts a list so a benchmark column that the paper defines as the pool of
    several score files (ASVLD) is assembled exactly as published.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    utts, keys = [], {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.rsplit(" ", 3)
                if len(parts) != 4:
                    continue
                utt, key = parts[0], parts[2]
                utts.append(utt)
                keys[utt] = key
    return utts, keys


# ===========================================================================
# Backend: standalone AASIST
# ===========================================================================

def _pad(x, max_len=CROP):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


class WavDataset(Dataset):
    def __init__(self, items, cut=CROP, sr=16000):
        self.items = items          # list of (utt_id, path)
        self.cut = cut
        self.sr = sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import librosa
        utt, path = self.items[i]
        try:
            X, _ = librosa.load(path, sr=self.sr)
            return Tensor(_pad(X, self.cut)), utt, True
        except Exception:
            # One undecodable file must not kill a multi-hour run; the row is
            # dropped rather than scored, and the count is reported.
            return Tensor(np.zeros(self.cut, dtype=np.float32)), utt, False


def score_aasist_raw(items, model_path, device, batch_size=64, num_workers=6, amp=False):
    from spoof_superb.models.aasist_raw import Model as AasistRaw

    model = AasistRaw(args=None, device=device).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  model loaded ({sum(p.numel() for p in model.parameters())} params) "
          f"<- {model_path}", flush=True)

    loader = DataLoader(WavDataset(items), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False, drop_last=False)

    use_amp = amp and str(device).startswith("cuda")
    out, n_bad = [], 0
    with torch.no_grad():
        for batch_x, utt, ok in tqdm(loader, desc="aasist_raw", mininterval=30.0):
            batch_x = batch_x.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(batch_x)
            # class-1 logit, exactly as main.py::produce_evaluation
            scores = logits[:, 1].float().cpu().numpy().ravel().tolist()
            ok_list = ok.tolist() if hasattr(ok, "tolist") else list(ok)
            for u, s, good in zip(utt, scores, ok_list):
                if good:
                    out.append((u, s))
                else:
                    n_bad += 1
    return out, n_bad


# ===========================================================================
# Backend: LFCC-GMM
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
                               total=len(items), desc="lfcc_gmm", mininterval=30.0):
            if score is None:
                n_bad += 1
            else:
                out.append((utt, score))
    return out, n_bad


# ===========================================================================
# Driver
# ===========================================================================

def run(model, model_path, dataset, output_file, reference_ssl=DEFAULT_REFERENCE_SSL,
        reference_file=None, device="cuda:0", batch_size=64, num_workers=6,
        n_jobs=16, limit=0, amp=False):
    if dataset not in DATASETS:
        print(f"[ERROR] unknown dataset {dataset!r}. Known: {', '.join(DATASETS)}")
        return 2
    spec = DATASETS[dataset]

    ref_paths = ([reference_file] if reference_file
                 else reference_paths(dataset, reference_ssl))
    for p in ref_paths:
        if not os.path.isfile(p):
            print(f"[ERROR] reference score file not found: {p}")
            return 2

    utts, keys = read_reference(ref_paths)
    for p in ref_paths:
        print(f"[{dataset}] reference {p}", flush=True)
    print(f"  {len(utts)} trials "
          f"({sum(1 for u in utts if keys[u]=='bonafide')} bonafide)", flush=True)

    if limit:
        utts = utts[:limit]
        print(f"  limited to first {len(utts)}", flush=True)

    # Resolve to audio paths, dropping anything not on disk so counts stay honest.
    items, missing = [], 0
    for u in utts:
        p = spec["resolve"](u)
        if p and os.path.isfile(p):
            items.append((u, p))
        else:
            missing += 1
    if missing:
        print(f"  [WARN] {missing}/{len(utts)} trials have no audio on disk (dropped)",
              flush=True)
    print(f"  scoring {len(items)} utterances", flush=True)
    if not items:
        print("  [ERROR] nothing to score; aborting.")
        return 1

    if model == "aasist_raw":
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print(f"[ERROR] {device} requested but CUDA unavailable; refusing CPU fallback.")
            return 2
        scored, n_bad = score_aasist_raw(items, model_path, device,
                                         batch_size=batch_size,
                                         num_workers=num_workers, amp=amp)
    elif model == "lfcc_gmm":
        scored, n_bad = score_lfcc_gmm(items, model_path, n_jobs=n_jobs)
    else:
        print(f"[ERROR] unknown model {model!r}")
        return 2

    if n_bad:
        print(f"  [WARN] {n_bad} unreadable files skipped (no score written)", flush=True)

    bad = [(u, s) for u, s in scored if not np.isfinite(s)]
    if bad:
        print(f"  [ERROR] {len(bad)} non-finite scores, e.g. {bad[:3]}")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    tmp = output_file + ".part"
    with open(tmp, "w") as fh:
        for utt, score in scored:
            fh.write("{} - {} {}\n".format(utt, keys[utt], score))
    os.replace(tmp, output_file)
    print(f"  scores saved -> {output_file}  ({len(scored)} lines)", flush=True)

    # Space-separated is the benchmark's canonical format, but some utt_ids
    # contain spaces (MLAAD v10: 39,000 rows such as "Cartesia.ai (Sonic-3)"),
    # which np.genfromtxt -- and therefore evaluation.py::calculate_EER -- cannot
    # parse. The repo's own answer is a tab-separated copy (see
    # linear_head_MLAAD_v10/tsv/), so emit one whenever it is needed.
    if any(" " in utt for utt, _ in scored):
        tsv = os.path.splitext(output_file)[0] + ".tsv"
        tmp = tsv + ".part"
        with open(tmp, "w") as fh:
            for utt, score in scored:
                fh.write("{}\t-\t{}\t{}\n".format(utt, keys[utt], score))
        os.replace(tmp, tsv)
        print(f"  tab-separated copy -> {tsv} (utt_ids contain spaces)", flush=True)

    # Computed here rather than via calculate_EER, which cannot read the above.
    try:
        bona = np.array([s for u, s in scored if keys[u] == "bonafide"])
        spoof = np.array([s for u, s in scored if keys[u] == "spoof"])
        if len(bona) and len(spoof):
            from spoof_superb.core.metrics import compute_eer
            print(f"  EER = {compute_eer(bona, spoof)[0]*100:.4f} %", flush=True)
        else:
            print(f"  [WARN] single-class output ({len(bona)} bona / {len(spoof)} spoof)")
    except Exception as exc:
        print(f"  [WARN] could not compute EER inline: {type(exc).__name__}: {exc}")

    return 0


def run_eval_from_main(args, cfg, device):
    """Entry point used by main.py when cfg.model_arch is a non-SSL baseline."""
    if not args.eval_dataset:
        print("[ERROR] --eval_dataset is required for the non-SSL baselines "
              f"(one of: {', '.join(DATASETS)})")
        return 2
    if not args.eval_output:
        print("[ERROR] --eval_output is required")
        return 2
    return run(model=cfg.model_arch,
               model_path=cfg.pretrained_checkpoint,
               dataset=args.eval_dataset,
               output_file=args.eval_output,
               device=device,
               batch_size=args.batch_size,
               n_jobs=getattr(args, "n_jobs", 16))


def main():
    ap = argparse.ArgumentParser(description="Score non-SSL baselines on the benchmark sets")
    ap.add_argument("--model", choices=["aasist_raw", "lfcc_gmm"])
    ap.add_argument("--model_path", help="swa.pth (aasist_raw) or GMM dir (lfcc_gmm)")
    ap.add_argument("--dataset", help="one of: " + ", ".join(DATASETS))
    ap.add_argument("--output_file")
    ap.add_argument("--reference_ssl", default=DEFAULT_REFERENCE_SSL,
                    help="SSL model whose score file defines the trial list")
    ap.add_argument("--reference_file", default=None,
                    help="Explicit reference score file (overrides --reference_ssl)")
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--n_jobs", type=int, default=16, help="LFCC worker processes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--list_datasets", action="store_true")
    args = ap.parse_args()

    if args.list_datasets:
        for k in DATASETS:
            refs = reference_paths(k, args.reference_ssl)
            n = sum(sum(1 for _ in open(r)) for r in refs if os.path.isfile(r))
            names = " + ".join(os.path.basename(r) for r in refs)
            print(f"{k:22s} trials={n:>9} ref={names}")
        return 0

    for req in ("model", "model_path", "dataset", "output_file"):
        if not getattr(args, req):
            ap.error(f"--{req} is required")

    return run(args.model, args.model_path, args.dataset, args.output_file,
               reference_ssl=args.reference_ssl, reference_file=args.reference_file,
               device=args.cuda_device, batch_size=args.batch_size,
               num_workers=args.num_workers, n_jobs=args.n_jobs,
               limit=args.limit, amp=args.amp)


if __name__ == "__main__":
    raise SystemExit(main())
