"""
orchestrate_baselines.py
------------------------
Phase 3: score both non-SSL baselines on all 10 benchmark sets and report the
model x dataset EER table.

Runs eval_baselines.py once per (model, dataset), resuming by skipping outputs
that already exist and validate. Sets are ordered smallest-first so failures
surface in seconds rather than after the 1.2M-row ASVLD run.

Score files land in:
    /data/ssl_anti_spoofing/asd_superb_score_files/baselines/{model}/
        {model}_{dataset}.txt

Usage
-----
    python orchestrate_baselines.py                       # both models, all 10
    python orchestrate_baselines.py --models lfcc_gmm     # one model
    python orchestrate_baselines.py --datasets wild eval_2019
    python orchestrate_baselines.py --report_only         # just the EER table
"""

import argparse
import os
import subprocess
import sys
import time

from eval_baselines import DATASETS, DEFAULT_REFERENCE_SSL, reference_paths

PY = sys.executable
REPO = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/data/ssl_anti_spoofing/asd_superb_score_files/baselines"
LOG_DIR = os.path.join(REPO, "outputs", "logs", "baseline_eval")

MODELS_ROOT = "/data/ssl_anti_spoofing/asd_superb_models/baselines"
AASIST_RUN_DIR = os.path.join(MODELS_ROOT,
                              "model_weighted_CCE_50_64_aasist_raw_ASV19_none")


def resolve_model_path(model):
    """Locate the trained artefact for a baseline.

    aasist_raw uses the BEST DEV-EER EPOCH CHECKPOINT, not swa.pth -- a
    deliberate departure from the SSL models, which are scored from swa.pth.

    Reason (measured on clean ASV19 LA dev by tools_select_aasist_ckpt.py):
        swa.pth          1.806 %
        epoch_44_1.178   0.670 %   <- selected
        epoch_31_1.254   1.178 %
        epoch_28_1.413   1.293 %

    main.py calls optimizer_swa.update_swa() on every dev-EER improvement. The
    SSL runs initialise best_val_eer = 1, so only checkpoints already under 1%
    EER were ever averaged and their SWA is a sane average of good models. The
    baselines had to initialise to inf (otherwise nothing is saved at all -- see
    humanpending.md item 8), which means SWA here averaged the whole trajectory
    starting from epoch 0 at 24.5% EER, and the average is 2.7x worse than the
    best single checkpoint.

    Selection used the DEV set only; no evaluation set influenced this choice.

    main.py names checkpoints epoch_{n}_{dev_eer:.3f}.pth, so the lowest EER in
    the filename is the best model.
    """
    if model == "lfcc_gmm":
        return os.path.join(MODELS_ROOT, "lfcc_gmm")

    swa = os.path.join(AASIST_RUN_DIR, "swa.pth")
    if not os.path.isdir(AASIST_RUN_DIR):
        return swa  # report the canonical missing path

    epochs = []
    for fn in os.listdir(AASIST_RUN_DIR):
        if fn.startswith("epoch_") and fn.endswith(".pth"):
            try:
                epochs.append((float(fn[:-4].split("_")[-1]), fn))
            except ValueError:
                continue
    if epochs:
        return os.path.join(AASIST_RUN_DIR, min(epochs)[1])
    return swa


MODEL_PATHS = {m: resolve_model_path(m) for m in ("aasist_raw", "lfcc_gmm")}

# Smallest first: a broken resolver or checkpoint fails in seconds, not hours.
ORDER = ["deepfake_eval_2024", "wild", "eval_2019", "spoofceleb",
         "asvspoof2021_DF", "asvspoof2021_LA", "Multilingual",
         "Famous_Figures", "asvspoof5", "asvspoofLD"]


def out_path(model, dataset):
    return os.path.join(OUT_ROOT, model, f"{model}_{dataset}.txt")


def expected_rows(dataset, reference_ssl=DEFAULT_REFERENCE_SSL):
    refs = reference_paths(dataset, reference_ssl)
    if not all(os.path.isfile(r) for r in refs):
        return -1
    return sum(sum(1 for _ in open(r)) for r in refs)


def read_scores(path):
    """Read a 4-column score file -> (labels, scores).

    Peels the three trailing fields from the RIGHT. utt_ids legitimately contain
    spaces (MLAAD v10 has 39,000 such rows, e.g. "Cartesia.ai (Sonic-3)"), so a
    whitespace split silently mis-reads them. Note evaluation.py::calculate_EER
    uses np.genfromtxt and therefore CANNOT read those files at all, which is why
    the repo also ships tab-separated copies under linear_head_MLAAD_v10/tsv/.
    """
    labels, scores = [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) != 4:
                labels.append(None)
                scores.append(float("nan"))
                continue
            labels.append(parts[2])
            try:
                scores.append(float(parts[3]))
            except ValueError:
                scores.append(float("nan"))
    return labels, scores


def eer_from_file(path):
    """EER (%) for a 4-column score file, bonafide = target."""
    import numpy as np
    from spoof_superb.core.metrics import compute_eer

    labels, scores = read_scores(path)
    labels = np.asarray(labels, dtype=object)
    scores = np.asarray(scores, dtype=float)
    bona = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    return compute_eer(bona, spoof)[0] * 100


def score_file_ok(path):
    """A score file counts as done only if it parses and has both classes."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False, "missing/empty"
    labels, scores = read_scores(path)
    n = len(labels)
    bad = sum(1 for l, s in zip(labels, scores) if l is None or s != s or s in (float("inf"), float("-inf")))
    bona = sum(1 for l in labels if l == "bonafide")
    spoof = sum(1 for l in labels if l == "spoof")
    if bad:
        return False, f"{bad} malformed/non-finite rows"
    if bona == 0 or spoof == 0:
        return False, f"single-class ({bona} bona / {spoof} spoof) -> EER undefined"
    return True, f"{n} rows ({bona} bona / {spoof} spoof)"


# torch.cuda.is_available() swallows the real reason into a UserWarning, so the
# probe re-raises it on stderr. The first outage in this run was diagnosed only
# from a 160-char-truncated copy of that warning; never lose it again.
CUDA_PROBE = (
    "import sys, torch\n"
    "try:\n"
    "    torch.cuda.init()\n"
    "    n = torch.cuda.device_count()\n"
    "    sys.exit(0 if n > 0 else 1)\n"
    "except Exception as e:\n"
    "    sys.stderr.write('CUDA_PROBE_ERROR: %s: %s\\n' % (type(e).__name__, e))\n"
    "    sys.exit(1)\n"
)
MAX_ATTEMPTS = 3
CUDA_WAIT_S = 3600      # how long to hold a dataset waiting for the driver
CUDA_POLL_S = 60


def cuda_healthy(verbose=True):
    """Probe CUDA in a FRESH process, reporting the reason on failure.

    This host has a recurring fault where CUDA initialisation starts failing for
    every NEW process while nvidia-smi still reports three healthy idle GPUs and
    the kernel log shows no NVRM/Xid error at all. It took out 6 of 10 datasets
    on the first aasist_raw sweep and later blocked MLAAD for 60+ minutes.
    Probing in-process would be useless (an already initialised context keeps
    working), so a subprocess is required.
    """
    p = subprocess.run([PY, "-c", CUDA_PROBE], capture_output=True, text=True)
    if p.returncode != 0 and verbose:
        for line in p.stderr.splitlines():
            if "CUDA_PROBE_ERROR" in line or "CUDA" in line.upper():
                print(f"  [cuda] {line.strip()}", flush=True)
    return p.returncode == 0


def wait_for_cuda(tag, wait_s=CUDA_WAIT_S):
    if cuda_healthy():
        return True
    waited = 0
    print(f"  [WAIT] {tag}: CUDA unavailable, holding (driver fault is recurring "
          f"on this host)", flush=True)
    while waited < wait_s:
        time.sleep(CUDA_POLL_S)
        waited += CUDA_POLL_S
        if cuda_healthy():
            print(f"  [OK]  {tag}: CUDA returned after {waited//60} min", flush=True)
            return True
    print(f"  [GIVE UP] {tag}: CUDA down for {wait_s//60} min", flush=True)
    return False


def run_one(model, dataset, device, n_jobs, batch_size, force=False):
    out = out_path(model, dataset)
    ok, why = score_file_ok(out)
    if ok and not force:
        print(f"  SKIP {model}/{dataset}: already complete -- {why}", flush=True)
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, f"{model}_{dataset}.log")

    cmd = [PY, "-u", os.path.join(REPO, "eval_baselines.py"),
           "--model", model, "--model_path", MODEL_PATHS[model],
           "--dataset", dataset, "--output_file", out,
           "--n_jobs", str(n_jobs), "--batch_size", str(batch_size),
           "--cuda_device", device]

    needs_gpu = (model == "aasist_raw")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if needs_gpu and not wait_for_cuda(f"{model}/{dataset}"):
            return 1

        t0 = time.time()
        print(f"  RUN  {model}/{dataset} -> {os.path.basename(out)}"
              f"{'' if attempt == 1 else f'  (attempt {attempt})'}  (log: {log})",
              flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0

        ok, why = score_file_ok(out)
        if rc == 0 and ok:
            print(f"  OK   {model}/{dataset}  rc={rc}  {dt/60:.1f} min  {why}", flush=True)
            return 0

        print(f"  FAIL {model}/{dataset}  rc={rc}  {dt/60:.1f} min  {why}"
              f"{'  -> retrying' if attempt < MAX_ATTEMPTS else ''}", flush=True)
        with open(log) as fh:
            tail = [l for l in fh.read().splitlines() if l.strip()][-6:]
        for l in tail:
            print(f"       | {l[:160]}", flush=True)

    return 1


def report(models, datasets, reference_ssl=DEFAULT_REFERENCE_SSL):
    """Per-dataset EER plus Mean and Pooled, matching Table 5's definitions.

    Mean   = arithmetic mean of the per-dataset EERs.
    Pooled = raw scores concatenated across all datasets, EER computed once, with
             NO score normalisation. This is the definition used by
             scripts/recompute_table5_mlaad_v10.py; sigmoid/z-score variants do
             not reproduce the published column.
    """
    import numpy as np
    from spoof_superb.core.metrics import compute_eer

    print("\n" + "=" * 96)
    print("BASELINE EER (%) -- trained on ASVspoof2019 LA train")
    print("=" * 96)
    header = f"{'dataset':22s} {'trials':>9} " + "".join(f"{m:>14s}" for m in models)
    print(header)
    print("-" * len(header))

    per_model_eers = {m: [] for m in models}
    pooled = {m: ([], []) for m in models}
    rows = {}

    for d in datasets:
        cells, n_rows = [], None
        for m in models:
            p = out_path(m, d)
            ok, _ = score_file_ok(p)
            if not ok:
                cells.append("--")
                per_model_eers[m].append(None)
                continue
            try:
                labels, scores = read_scores(p)
                labels_a = np.asarray(labels, dtype=object)
                scores_a = np.asarray(scores, dtype=float)
                bona = scores_a[labels_a == "bonafide"]
                spoof = scores_a[labels_a == "spoof"]
                eer = compute_eer(bona, spoof)[0] * 100
                cells.append(f"{eer:.3f}")
                per_model_eers[m].append(eer)
                pooled[m][0].extend(bona.tolist())
                pooled[m][1].extend(spoof.tolist())
                n_rows = len(labels)
            except Exception as exc:
                cells.append(f"ERR({type(exc).__name__})")
                per_model_eers[m].append(None)
        rows[d] = cells
        exp = expected_rows(d, reference_ssl)
        n_show = n_rows if n_rows is not None else exp
        flag = "" if (n_rows is None or n_rows == exp) else f"  (ref {exp})"
        print(f"{d:22s} {n_show:>9} " + "".join(f"{c:>14s}" for c in cells) + flag)

    print("-" * len(header))
    mean_cells, pooled_cells = [], []
    for m in models:
        vals = per_model_eers[m]
        if any(v is None for v in vals) or not vals:
            mean_cells.append("--")
            pooled_cells.append("--")
            continue
        mean_cells.append(f"{sum(vals)/len(vals):.3f}")
        b, s = pooled[m]
        pooled_cells.append(f"{compute_eer(np.asarray(b), np.asarray(s))[0]*100:.3f}")
    print(f"{'Mean':22s} {'':>9} " + "".join(f"{c:>14s}" for c in mean_cells))
    print(f"{'Pooled':22s} {'':>9} " + "".join(f"{c:>14s}" for c in pooled_cells))
    print("=" * 96)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Score both baselines on all 10 benchmark sets")
    ap.add_argument("--models", nargs="+", default=["aasist_raw", "lfcc_gmm"],
                    choices=["aasist_raw", "lfcc_gmm"])
    ap.add_argument("--datasets", nargs="+", default=ORDER, choices=list(DATASETS))
    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="Re-score even if output exists")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    datasets = [d for d in ORDER if d in args.datasets]

    if not args.report_only:
        for m in args.models:
            path = MODEL_PATHS[m]
            if not os.path.exists(path):
                print(f"[ERROR] {m}: trained model not found at {path}")
                return 2

        n_fail = 0
        for m in args.models:
            print(f"\n### {m} ###", flush=True)
            for d in datasets:
                n_fail += run_one(m, d, args.cuda_device, args.n_jobs,
                                  args.batch_size, force=args.force)
        if n_fail:
            print(f"\n[WARN] {n_fail} (model, dataset) runs failed", flush=True)

    report(args.models, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
