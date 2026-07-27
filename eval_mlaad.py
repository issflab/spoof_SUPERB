"""
eval_mlaad.py
-------------
Self-contained evaluation of the LinearHead (UtteranceLevel) SSL classifier on the
MLAAD fake-audio dataset. One score file per model. Additive: does NOT touch main.py's
eval branch, config.py, or data_utils_SSL.py.

Why standalone (same reasons as eval_asvld.py):
  - main.py references cfg.eval_protocol (no CLI setter / no default) -> crash.
  - Dataset_ASVspoof2021_eval hardcodes 'release_in_the_wild/*.wav' -> wrong path.
  - genSpoof_list(is_eval=True) splits on ',' -> wrong utt_id parse for MLAAD.
  - args.ssl_feature is read by the model but never assigned in the repo; set here.

utt_id / audio:
  Every MLAAD fake wav is enumerated under {mlaad_root}/fake/**/*.wav. The utt_id
  written to the score file is the path RELATIVE TO {data_base}, e.g.
      MLAAD/fake/it/tts_models_it_mai_male_vits/novelle..._f000122.wav
  which reproduces exactly the ids in the reference files
  linear_head_Multilingual_<ssl>.txt (base dir = /data/Data). Audio is loaded from
  {data_base}/{utt_id}.

Label:
  All MLAAD fake audio is spoof, so key := 'spoof' (matches the spoof rows of the
  reference; the reference's bonafide rows come from the separate M-AILABS set and
  are out of scope for an MLAAD-only run).

Output (4 cols, identical to reference score files):
  {utt_id} - {key} {score}
where score = model logit for class index 1 (as in main.py produce_evaluation).
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


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def enumerate_wavs(root, data_base):
    """Return sorted list of utt_ids (relative to data_base) for every wav under root.

    Used for both datasets:
      MLAAD    -> root=/data/Data/MLAAD/fake   label=spoof
      M-AILABS -> root=/data/Data/MAILabs      label=bonafide
    utt_ids are relative to data_base (/data/Data), reproducing the id format of
    the reference score files.
    """
    utts = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # Skip macOS AppleDouble sidecars ('._name.wav'): 176-245 byte metadata
            # stubs, not audio. M-AILABS carries 6 of them and they are absent from
            # the reference score files, so they are not real utterances.
            if fn.startswith("._"):
                continue
            if fn.endswith(".wav"):
                full = os.path.join(dirpath, fn)
                utts.append(os.path.relpath(full, data_base))
    utts.sort()
    return utts


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


def read_protocol_csv(path, attack_col_bonafide="a00"):
    """Read a SpoofCeleb-style protocol CSV -> [(utt_id, label), ...].

    Header: file,speaker,attack. The utt_id is the 'file' column VERBATIM (it
    already carries the .flac extension and reproduces the reference ids), and
    audio lives at {audio_base}/{file}.

    Unlike MLAAD (all spoof) and M-AILABS (all bonafide), SpoofCeleb's label is
    per-utterance: attack == 'a00' is bonafide, every other attack is spoof.
    That is why the single --label flag cannot drive this dataset.
    """
    import csv
    pairs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("file") or "").strip()
            if not fn:
                continue
            attack = (row.get("attack") or "").strip()
            label = "bonafide" if attack == attack_col_bonafide else "spoof"
            pairs.append((fn, label))
    return pairs


class MLAADDataset(Dataset):
    def __init__(self, utt_ids, data_base, cut=64600, sr=16000):
        self.utt_ids = utt_ids
        self.data_base = data_base
        self.cut = cut
        self.sr = sr

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        import librosa  # imported in worker to keep fork light
        utt_id = self.utt_ids[index]
        path = os.path.join(self.data_base, utt_id)
        try:
            X, _ = librosa.load(path, sr=self.sr)
            X_pad = pad(X, self.cut)
            return Tensor(X_pad), utt_id, True
        except Exception:
            # An undecodable file must not kill a multi-hour run. Return silence
            # flagged not-ok; the caller drops these rows so no bogus score is
            # ever written, and reports how many were skipped.
            return Tensor(np.zeros(self.cut, dtype=np.float32)), utt_id, False


def build_model(ssl_model, model_path, device):
    args = SimpleNamespace(ssl_feature=ssl_model, ssl_model=ssl_model)
    model = LinearHead(args, device).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)  # swa.pth is a full-module state_dict
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="MLAAD eval -> one score file per model")
    ap.add_argument("--model_path", required=True, help="Path to swa.pth")
    ap.add_argument("--ssl_model", required=True, help="s3prl upstream name (e.g. xls_r_300m)")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--mlaad_root", default="/data/Data/MLAAD")
    ap.add_argument("--enumerate_root", default=None,
                    help="Dir to walk for wavs. Default {mlaad_root}/fake. "
                         "For M-AILABS pass /data/Data/MAILabs.")
    ap.add_argument("--label", default="spoof",
                    help="Ground-truth label written in col 3: 'spoof' for MLAAD fake, "
                         "'bonafide' for M-AILABS.")
    ap.add_argument("--restrict_prefix", default="MLAAD/",
                    help="With --restrict_to, only keep reference utt_ids under this prefix.")
    ap.add_argument("--data_base", default="/data/Data",
                    help="Base dir the utt_id is written relative to (reference uses /data/Data).")
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--restrict_to", default=None,
                    help="Optional reference score file; only score utt_ids present there "
                         "(used for a fast canary/verification against the reference subset).")
    ap.add_argument("--protocol_csv", default=None,
                    help="Protocol-driven mode (SpoofCeleb): CSV with columns file,speaker,attack. "
                         "The eval set and the PER-UTTERANCE label come from this file, so --label "
                         "is ignored. Requires --audio_base.")
    ap.add_argument("--audio_base", default=None,
                    help="With --protocol_csv: dir the 'file' column is relative to, e.g. "
                         "/data/Data/SpoofCeleb/flac/evaluation")
    ap.add_argument("--limit", type=int, default=0, help="If >0, score only the first N (debug).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--amp", action="store_true",
                    help="Autocast fp16 for the forward pass: fp16 matmuls (speed) while "
                         "autocast keeps FFT/STFT frontends in fp32. Preferred over a hard "
                         ".half(), which crashes spectrogram upstreams (torch.stft has no Half).")
    args = ap.parse_args()

    # Fail loudly rather than silently falling back to CPU: a CPU run of the full
    # MLAAD set is ~25h vs ~20min on an A100, and the fallback is easy to miss.
    if args.cuda_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[{args.ssl_model}] ERROR: {args.cuda_device} requested but CUDA is "
              f"unavailable in this process; refusing to run on CPU.", flush=True)
        return 2
    device = args.cuda_device if torch.cuda.is_available() else "cpu"
    print(f"[{args.ssl_model}] device={device}", flush=True)

    enum_root = args.enumerate_root or os.path.join(args.mlaad_root, "fake")

    # label_map is None for the MLAAD/M-AILABS paths (single --label for every row);
    # protocol mode fills it so each row gets its own ground-truth label.
    label_map = None

    if args.protocol_csv:
        if not args.audio_base:
            print("  [ERROR] --protocol_csv requires --audio_base", flush=True)
            return 1
        pairs = read_protocol_csv(args.protocol_csv)
        args.data_base = args.audio_base
        eval_list = [u for u, _ in pairs]
        label_map = dict(pairs)
        n_bona = sum(1 for _, l in pairs if l == "bonafide")
        print(f"  protocol {args.protocol_csv}: {len(eval_list)} utts "
              f"({n_bona} bonafide, {len(eval_list) - n_bona} spoof), "
              f"audio_base={args.audio_base}", flush=True)
    elif args.restrict_to:
        restrict = read_restrict_utts(args.restrict_to)
        # Keep only ids under the requested prefix that exist on disk.
        eval_list = []
        missing = 0
        for u in restrict:
            if not u.startswith(args.restrict_prefix):
                continue
            if os.path.isfile(os.path.join(args.data_base, u)):
                eval_list.append(u)
            else:
                missing += 1
        # de-dup while preserving order
        seen = set()
        eval_list = [u for u in eval_list if not (u in seen or seen.add(u))]
        print(f"  restrict_to reference: {len(eval_list)} {args.restrict_prefix} utts on disk "
              f"({missing} referenced but missing)", flush=True)
    else:
        eval_list = enumerate_wavs(enum_root, args.data_base)
        print(f"  enumerated {len(eval_list)} wavs under {enum_root}", flush=True)

    if args.limit and args.limit > 0:
        eval_list = eval_list[: args.limit]
        print(f"  limited to first {len(eval_list)}", flush=True)

    if not eval_list:
        print("  [ERROR] nothing to score; aborting.", flush=True)
        return 1

    model = build_model(args.ssl_model, args.model_path, device)
    nb = sum(p.numel() for p in model.parameters())
    print(f"  model loaded, nb_params={nb} amp={args.amp}", flush=True)

    ds = MLAADDataset(eval_list, args.data_base)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False)

    use_amp = args.amp and device.startswith("cuda")
    fname_list, score_list = [], []
    n_unreadable = 0
    with torch.no_grad():
        for batch_x, utt_id, ok in tqdm(loader, desc=args.ssl_model, mininterval=30.0):
            batch_x = batch_x.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                batch_out = model(batch_x)
            batch_score = batch_out[:, 1].float().data.cpu().numpy().ravel().tolist()
            ok_list = ok.tolist() if hasattr(ok, "tolist") else list(ok)
            for u, s, good in zip(utt_id, batch_score, ok_list):
                if good:
                    fname_list.append(u)
                    score_list.append(s)
                else:
                    n_unreadable += 1

    if n_unreadable:
        print(f"  [WARN] {n_unreadable} unreadable audio files skipped (no score written)",
              flush=True)

    assert len(fname_list) == len(score_list), f"{len(fname_list)} != {len(score_list)}"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    tmp = args.output_file + ".part"
    with open(tmp, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            key = label_map[fn] if label_map is not None else args.label
            fh.write("{} - {} {}\n".format(fn, key, sco))
    os.replace(tmp, args.output_file)
    print(f"  scores saved -> {args.output_file}  ({len(fname_list)} lines)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
