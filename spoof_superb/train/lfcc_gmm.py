"""
train_lfcc_gmm.py
-----------------
Train the LFCC-GMM baseline on ASVspoof2019 LA train.

Trains one 512-component diagonal GMM per class (bonafide, spoof) on 60-dim
LFCC features, following /home/alhashim/Rob-ASD/ASD_ML/train.py.

Same training data and protocol as every SSL model in this benchmark:
    /data/Data/ASVSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac
    ASVspoof2019.LA.cm.train.trn.txt

CPU-only by design (EM over diagonal GMMs; the reference has no GPU path).

Usage
-----
    python train_lfcc_gmm.py --out_dir /data/.../baselines/lfcc_gmm --n_jobs 16

or through the shared CLI:

    SSL_MODEL_ARCH=lfcc_gmm python main.py --n_jobs 16
"""

import argparse
import os
import sys
import time

from spoof_superb.data.datasets_ssl import genSpoof_list
from spoof_superb.models.lfcc_gmm import INIT_STRIDE, N_COMPONENTS, extract_many, train_gmm

from spoof_superb.config import cfg

DEFAULT_OUT_DIR = os.path.join(cfg.baseline_models_root, "lfcc_gmm")
CLASSES = ("bonafide", "spoof")


def build_file_lists(database_path, protocols_path, train_protocol):
    """Return {class_label: [abs audio paths]} for the ASV19 LA train split.

    Uses the repo's own genSpoof_list so the trial set is byte-identical to
    what the SSL models were trained on.
    """
    proto = os.path.join(protocols_path, train_protocol)
    d_label, file_list = genSpoof_list(proto, is_train=True)

    flac_dir = os.path.join(database_path, "ASVspoof2019_LA_train", "flac")

    per_class = {c: [] for c in CLASSES}
    for utt in file_list:
        # genSpoof_list maps bonafide -> 1, spoof -> 0
        label = "bonafide" if d_label[utt] == 1 else "spoof"
        per_class[label].append(os.path.join(flac_dir, utt + ".flac"))
    return per_class, proto, flac_dir


def train(database_path, protocols_path, train_protocol, out_dir,
          n_jobs=8, ncomp=N_COMPONENTS, init_stride=INIT_STRIDE, seed=None,
          limit=0):
    os.makedirs(out_dir, exist_ok=True)

    per_class, proto, flac_dir = build_file_lists(
        database_path, protocols_path, train_protocol)

    print(f"protocol : {proto}")
    print(f"audio    : {flac_dir}")
    for c in CLASSES:
        print(f"  {c:9s}: {len(per_class[c])} utterances")
    print(f"out_dir  : {out_dir}")
    print(f"ncomp={ncomp} init_stride={init_stride} n_jobs={n_jobs}", flush=True)

    if not os.path.isdir(flac_dir):
        print(f"[ERROR] audio directory does not exist: {flac_dir}")
        return 1

    for c in CLASSES:
        paths = per_class[c]
        if limit:
            paths = paths[:limit]
        print(f"\n=== {c}: extracting LFCC for {len(paths)} utterances ===", flush=True)
        t0 = time.time()
        feats, errors = extract_many(paths, n_jobs=n_jobs, desc=f"lfcc/{c}")
        n_ok = sum(1 for f in feats if f is not None)
        n_frames = sum(f.shape[0] for f in feats if f is not None)
        print(f"  {n_ok}/{len(paths)} ok, {n_frames} frames, "
              f"{time.time()-t0:.1f}s", flush=True)
        if errors:
            print(f"  [WARN] {len(errors)} unreadable files, first few:")
            for e in errors[:5]:
                print(f"    {e}")
        if n_ok == 0:
            print(f"[ERROR] no usable features for class {c}; aborting.")
            return 1

        print(f"=== {c}: training GMM ===", flush=True)
        t0 = time.time()
        train_gmm(feats, save_dir=os.path.join(out_dir, c), ncomp=ncomp,
                  init_stride=init_stride, seed=seed)
        print(f"  {c} GMM done in {time.time()-t0:.1f}s", flush=True)
        del feats

    print("\nBoth GMMs trained.")
    for c in CLASSES:
        final = os.path.join(out_dir, c, "gmm_final.pkl")
        print(f"  {c:9s} -> {final}  "
              f"({os.path.getsize(final)/1e6:.1f} MB)" if os.path.isfile(final)
              else f"  {c:9s} -> MISSING")
    return 0


def run_train_from_main(args, cfg):
    """Entry point used by main.py when cfg.model_arch == 'lfcc_gmm'."""
    out_dir = os.environ.get("LFCC_GMM_OUT_DIR", DEFAULT_OUT_DIR)
    return train(
        database_path=cfg.database_path,
        protocols_path=cfg.protocols_path,
        train_protocol=cfg.train_protocol,
        out_dir=out_dir,
        n_jobs=getattr(args, "n_jobs", 8),
        seed=getattr(args, "seed", None),
    )


def main():
    ap = argparse.ArgumentParser(description="Train the LFCC-GMM baseline on ASV19 LA train")
    ap.add_argument("--database_path", default=cfg.database_path)
    ap.add_argument("--protocols_path",
                    default=cfg.protocols_path)
    ap.add_argument("--train_protocol", default="ASVspoof2019.LA.cm.train.trn.txt")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--ncomp", type=int, default=N_COMPONENTS)
    ap.add_argument("--init_stride", type=int, default=INIT_STRIDE,
                    help="Use every Nth utterance for the sklearn init fit (reference: 20)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="Debug: cap utterances per class")
    args = ap.parse_args()

    return train(args.database_path, args.protocols_path, args.train_protocol,
                 args.out_dir, n_jobs=args.n_jobs, ncomp=args.ncomp,
                 init_stride=args.init_stride, seed=args.seed, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
