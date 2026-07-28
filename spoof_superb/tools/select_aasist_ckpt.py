"""
tools_select_aasist_ckpt.py
---------------------------
Pick the aasist_raw checkpoint to score the benchmark with, using the ASV19 LA
DEV set only. Never uses an evaluation set for selection -- that would leak.

Why this exists: main.py calls optimizer_swa.update_swa() on every dev-EER
improvement. For the SSL models the (buggy) `best_val_eer = 1` threshold meant
that only checkpoints already under 1% EER were ever averaged. The baselines had
to initialise to inf to save anything at all, which means swa.pth here averages
weights from epoch 0 (dev EER 24.5%) through epoch 44 (1.18%). Averaging that
wide a trajectory can be much worse than the best single checkpoint, so the two
are compared explicitly rather than assumed.

Scores dev clean (RawBoost algo 0): augmentation is a training-time device, and
both candidates are measured identically so the comparison is fair.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from spoof_superb.models.aasist_raw import Model as AasistRaw
from config import cfg
from spoof_superb.data.datasets_ssl import Dataset_ASVspoof2019_train, genSpoof_list
from spoof_superb.core.metrics import compute_eer

RUN_DIR = ("/data/ssl_anti_spoofing/asd_superb_models/baselines/"
           "model_weighted_CCE_50_64_aasist_raw_ASV19_none")


def score_dev(ckpt, device, batch_size=64, num_workers=8):
    dev_proto = os.path.join(cfg.protocols_path, cfg.dev_protocol)
    d_label_dev, file_dev = genSpoof_list(dev_proto, is_train=False)

    # algo=0 -> no RawBoost, i.e. clean dev audio
    args = SimpleNamespace(algo=0)
    dev_set = Dataset_ASVspoof2019_train(
        args, list_IDs=file_dev, labels=d_label_dev,
        base_dir=os.path.join(cfg.database_path, 'ASVspoof2019_LA_dev'), algo=0)
    loader = DataLoader(dev_set, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False)

    model = AasistRaw(args=None, device=device).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()

    scores, keys = [], []
    with torch.no_grad():
        for batch_x, utt_id, batch_y in loader:
            out = model(batch_x.to(device))
            scores.extend(out[:, 1].float().cpu().numpy().ravel().tolist())
            keys.extend(batch_y.view(-1).tolist())

    scores = np.asarray(scores)
    keys = np.asarray(keys)
    bona = scores[keys == 1]      # genSpoof_list: bonafide -> 1
    spoof = scores[keys == 0]
    eer = compute_eer(bona, spoof)[0] * 100
    return eer, len(scores), len(bona), len(spoof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=RUN_DIR)
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = args.cuda_device if torch.cuda.is_available() else "cpu"

    cands = []
    swa = os.path.join(args.run_dir, "swa.pth")
    if os.path.isfile(swa):
        cands.append(("swa", swa))
    epochs = []
    for fn in os.listdir(args.run_dir):
        if fn.startswith("epoch_") and fn.endswith(".pth"):
            try:
                epochs.append((float(fn[:-4].split("_")[-1]), fn))
            except ValueError:
                pass
    for _, fn in sorted(epochs)[:3]:
        cands.append((fn.replace(".pth", ""), os.path.join(args.run_dir, fn)))

    print(f"device={device}  clean ASV19 LA dev (RawBoost off)\n")
    results = []
    for name, path in cands:
        eer, n, nb, ns = score_dev(path, device, args.batch_size)
        print(f"  {name:24s} dev EER = {eer:6.3f} %   ({n} trials, {nb} bona / {ns} spoof)",
              flush=True)
        results.append((eer, name, path))

    results.sort()
    best_eer, best_name, best_path = results[0]
    print(f"\nSELECTED: {best_name}  (dev EER {best_eer:.3f} %)")
    print(f"PATH: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
