"""
eval_asvld.py
-------------
Self-contained evaluation for the ASVspoof Laundered Database (ASVLD) using the
LinearHead (UtteranceLevel) SSL classifier. One score file per (model, condition).

Why standalone (does not use main.py eval branch):
  - main.py references cfg.eval_protocol (no CLI setter / no default) -> crash.
  - Dataset_ASVspoof2021_eval hardcodes 'release_in_the_wild/*.wav' -> wrong path
    and wrong extension for ASVLD (which is {Condition}/flac/{utt_id}.flac).
  - genSpoof_list(is_eval=True) splits on ',' then takes parts[0]; ASVLD protocols
    are space-separated 6-column -> wrong utt_id parse.
  - args.ssl_feature is read by the model but never assigned anywhere in the repo;
    here we set ssl_feature := ssl_model explicitly.

Protocol format (6 space-separated cols):
    speaker  utt_id  attack_id  key  condition  variant
Audio:
    {audio_base_dir}/{condition}/flac/{utt_id}.flac
Output (4 cols, matches existing reference score files):
    {utt_id} - {key} {score}
where score = model logit for class index 1 (spoof/bonafide logit as in main.py).
"""

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from linear_model import UtteranceLevel as LinearHead


PROTOCOL_TEMPLATE = "ASVspoofLauneredDatabase_{condition}.txt"  # note upstream misspelling "Launered"
VALID_CONDITIONS = ["Noise_Addition", "Reverberation", "Resampling", "Recompression", "Filtering"]


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def parse_protocol(protocol_path):
    """Return (ordered_utt_ids, key_by_utt) from a 6-col ASVLD protocol."""
    ordered = []
    key_by_utt = {}
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"[WARN] skipping malformed protocol line: {line.strip()!r}")
                continue
            utt_id = parts[1]
            key = parts[3]
            ordered.append(utt_id)
            key_by_utt[utt_id] = key
    return ordered, key_by_utt


def read_restrict_utts(restrict_path):
    """Read utt_ids (col 0) from a reference score file to constrain the eval set."""
    utts = []
    with open(restrict_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            utts.append(line.split()[0])
    return utts


class ASVLDDataset(Dataset):
    def __init__(self, utt_ids, flac_dir, cut=64600, sr=16000):
        self.utt_ids = utt_ids
        self.flac_dir = flac_dir
        self.cut = cut
        self.sr = sr

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        import librosa  # imported in worker to keep fork light
        utt_id = self.utt_ids[index]
        path = os.path.join(self.flac_dir, utt_id + ".flac")
        X, _ = librosa.load(path, sr=self.sr)
        X_pad = pad(X, self.cut)
        return Tensor(X_pad), utt_id


def build_model(ssl_model, model_path, device):
    args = SimpleNamespace(ssl_feature=ssl_model, ssl_model=ssl_model)
    model = LinearHead(args, device).to(device)
    state = torch.load(model_path, map_location=device)
    # swa.pth is a plain state_dict of the full UtteranceLevel module.
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="ASVLD eval -> per-condition score file")
    ap.add_argument("--model_path", required=True, help="Path to swa.pth")
    ap.add_argument("--ssl_model", required=True, help="s3prl upstream name (e.g. xls_r_300m)")
    ap.add_argument("--condition", required=True, choices=VALID_CONDITIONS)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--protocols_dir", required=True)
    ap.add_argument("--audio_base_dir", required=True,
                    help="Dir containing {condition}/flac/*.flac")
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--restrict_to", default=None,
                    help="Optional reference score file; only score utt_ids present there "
                         "(used to reproduce/verify against an existing reference subset).")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    # Skip guard: conditions listed (one per line) in an .asvld_skip file next to
    # this script are no-op'd. Lets us drop a condition from an already-launched
    # run, since each condition spawns a fresh python that re-reads this file.
    skip_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".asvld_skip")
    if os.path.isfile(skip_file):
        with open(skip_file) as sf:
            skip_set = {ln.strip() for ln in sf if ln.strip() and not ln.startswith("#")}
        if args.condition in skip_set:
            print(f"[{args.condition}] listed in .asvld_skip -> skipping (no scoring).")
            return 0

    device = args.cuda_device if torch.cuda.is_available() else "cpu"
    print(f"[{args.condition}|{args.ssl_model}] device={device}")

    protocol_path = os.path.join(args.protocols_dir, PROTOCOL_TEMPLATE.format(condition=args.condition))
    if not os.path.isfile(protocol_path):
        print(f"[WARN] protocol missing for {args.condition}: {protocol_path} -- skipping condition")
        return 0

    ordered, key_by_utt = parse_protocol(protocol_path)
    print(f"  protocol utt_ids: {len(ordered)}")

    if args.restrict_to:
        restrict = read_restrict_utts(args.restrict_to)
        in_proto = [u for u in restrict if u in key_by_utt]
        n_dropped = len(restrict) - len(in_proto)
        if n_dropped:
            print(f"  [WARN] {n_dropped} restrict utt_ids not in protocol (dropped)")
        eval_list = in_proto
        print(f"  restricted to reference subset: {len(eval_list)}")
    else:
        eval_list = ordered

    flac_dir = os.path.join(args.audio_base_dir, args.condition, "flac")

    # Existence pre-pass: drop (with warning) any utt_id whose audio is missing,
    # so the DataLoader cannot crash mid-run and counts are honest.
    present, missing = [], 0
    for u in eval_list:
        if os.path.isfile(os.path.join(flac_dir, u + ".flac")):
            present.append(u)
        else:
            missing += 1
    if missing:
        print(f"  [WARN] {missing} audio files missing under {flac_dir} (dropped)")
    eval_list = present
    print(f"  scoring {len(eval_list)} utterances")

    if not eval_list:
        print("  [ERROR] nothing to score; aborting this (model,condition).")
        return 1

    model = build_model(args.ssl_model, args.model_path, device)
    nb = sum(p.numel() for p in model.parameters())
    print(f"  model loaded, nb_params={nb}")

    ds = ASVLDDataset(eval_list, flac_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False)

    fname_list, score_list = [], []
    with torch.no_grad():
        for batch_x, utt_id in tqdm(loader, desc=f"{args.condition}/{args.ssl_model}"):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel().tolist()
            fname_list.extend(utt_id)
            score_list.extend(batch_score)

    assert len(fname_list) == len(score_list) == len(eval_list), \
        f"count mismatch: {len(fname_list)} / {len(score_list)} / {len(eval_list)}"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            fh.write("{} - {} {}\n".format(fn, key_by_utt[fn], sco))
    print(f"  scores saved -> {args.output_file}  ({len(fname_list)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
