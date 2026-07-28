"""
orchestrate_spoofceleb.py
-------------------------
Run eval_mlaad.py (protocol mode) for all 24 linear-head models over the
SpoofCeleb evaluation set, scheduling models greedily across N GPUs (one model
per GPU at a time). After each model, verify against the SpoofCeleb reference
with verify_spoofceleb.py and record the result. Live status file + summary.

Differences from orchestrate_mlaad.py:
  - protocol-driven eval set (--protocol_csv/--audio_base), per-utterance labels
  - no SKIP list: byol_a_2048 and mockingjay are in scope here (fp32 fixes byol's
    fp16 STFT crash; mockingjay's SpoofCeleb reference is usable)
  - a model exiting rc=2 (CUDA init raced) is retried once in a fresh process
  - verdict comes from Spearman (see verify_spoofceleb.py)

Nothing here mutates main.py/config.py/data_utils_SSL.py, the SpoofCeleb data,
or the read-only reference score files.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

REPO = "/home/alhashim/ASD_SUPERB/spoof_SUPERB"
PY = "/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python"
MODELS_ROOT = "/data/ssl_anti_spoofing/asd_superb_models/linear_head_models"
OUT_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_SpoofCeleb"
REF_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head"
PROTOCOL = "/data/Data/SpoofCeleb/metadata/evaluation.csv"
AUDIO_BASE = "/data/Data/SpoofCeleb/flac/evaluation"
LOG_DIR = os.path.join(OUT_DIR, "logs")
STATUS = os.path.join(OUT_DIR, "run_status.json")
SUMMARY = os.path.join(OUT_DIR, "SUMMARY.txt")
GPUS = [0, 1, 2]
PREFIX = "model_weighted_CCE_50_64_linear_head_ASV19_"
EXPECT_LINES = 91130
# This host's CUDA driver intermittently fails cuInit (error 3) for minutes at a
# time; nvidia-smi stays healthy throughout. eval_mlaad.py correctly refuses to
# run on CPU (rc=2), but that is an ENVIRONMENT fault, not a model fault, so a
# model hitting it must go back on the queue rather than be marked failed --
# otherwise one outage drains all 24 models into failures in a few minutes.
MAX_ATTEMPTS = 6          # per model, across CUDA outages
CUDA_WAIT_S = 3 * 3600    # max wait for the driver to come back before giving up
CUDA_POLL_S = 60


def gpu_uuids():
    """Map GPU index -> UUID.

    CUDA_VISIBLE_DEVICES by *index* fails to initialise on this host once another
    process holds a different device ("CUDA driver initialization failed"), which
    previously sent whole models to CPU. Selecting by UUID is unaffected, so every
    launch pins its device by UUID.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).decode()
    return [u.strip() for u in out.splitlines() if u.strip()]


_GPU_UUIDS = gpu_uuids()

_lock = threading.Lock()
_results = {}


def discover_models(only=None):
    out = []
    for name in sorted(os.listdir(MODELS_ROOT)):
        d = os.path.join(MODELS_ROOT, name)
        ckpt = os.path.join(d, "swa.pth")
        if os.path.isdir(d) and name.startswith(PREFIX) and os.path.isfile(ckpt):
            ssl = name[len(PREFIX):]
            if only and ssl not in only:
                continue
            out.append((ssl, ckpt))
    return out


def write_status():
    # Serialize snapshot+write under the lock with a per-thread tmp name; two
    # threads sharing one tmp path race on os.replace otherwise.
    with _lock:
        snap = dict(_results)
        tmp = f"{STATUS}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        os.replace(tmp, STATUS)


def cuda_healthy(gpu):
    """True if a fresh process can actually initialise CUDA on this GPU.

    Checked in a subprocess with the same UUID pinning the real run uses:
    torch caches its failed init, so an in-process check would be stale.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_GPU_UUIDS[gpu])
    try:
        return subprocess.call(
            [PY, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180) == 0
    except Exception:
        return False


def wait_for_cuda(gpu, ssl):
    """Block until CUDA is usable on this GPU. Returns False if it never returns."""
    if cuda_healthy(gpu):
        return True
    waited = 0
    print(f"[orchestrate] {ssl}: CUDA down on gpu {gpu}; holding the model on the "
          f"queue until the driver returns", flush=True)
    while waited < CUDA_WAIT_S:
        time.sleep(CUDA_POLL_S)
        waited += CUDA_POLL_S
        if cuda_healthy(gpu):
            print(f"[orchestrate] {ssl}: CUDA back on gpu {gpu} after {waited}s", flush=True)
            return True
        if waited % 900 == 0:
            print(f"[orchestrate] {ssl}: still waiting for CUDA on gpu {gpu} ({waited}s)",
                  flush=True)
    return False


def output_is_complete(out_file):
    """A finished, NaN-free score file -- lets a restart resume instead of redo."""
    if not os.path.isfile(out_file):
        return False
    n = 0
    with open(out_file) as f:
        for line in f:
            p = line.rstrip("\n").rsplit(" ", 3)
            if len(p) < 4:
                continue
            if "nan" in p[3].lower() or "inf" in p[3].lower():
                return False
            n += 1
    return n == EXPECT_LINES


def _launch(ssl, ckpt, gpu, out_file, log_file):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_GPU_UUIDS[gpu])
    cmd = [PY, "-m", "spoof_superb.scoring.driver",
           "--model", "linear_head", "--source", "protocol_csv",
           "--model_path", ckpt, "--ssl_model", ssl,
           "--output_file", out_file, "--cuda_device", "cuda:0",
           "--protocol_csv", PROTOCOL, "--audio_base", AUDIO_BASE,
           "--batch_size", "32", "--num_workers", "6"]  # fp32: autocast NaNs spectrogram upstreams
    with open(log_file, "a") as lf:
        lf.write(f"\n=== launch gpu={gpu} uuid={_GPU_UUIDS[gpu]} {time.ctime()} ===\n")
        lf.flush()
        return subprocess.call(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)


def run_one(ssl, ckpt, gpu):
    out_file = os.path.join(OUT_DIR, f"linear_head_SpoofCeleb_{ssl}.txt")
    log_file = os.path.join(LOG_DIR, f"eval_SpoofCeleb_{ssl}.log")
    ref_file = os.path.join(REF_DIR, f"linear_head_spoofceleb_{ssl}.txt")

    with _lock:
        _results[ssl] = {"status": "running", "gpu": gpu, "started": time.time()}
    write_status()

    t0 = time.time()

    # Resume: a complete, NaN-free file from an earlier run is not recomputed.
    if output_is_complete(out_file):
        print(f"[orchestrate] {ssl}: existing output is complete; re-verifying only", flush=True)
        rc = 0
    else:
        open(log_file, "w").close()
        attempts = 0
        rc = 2
        while attempts < MAX_ATTEMPTS:
            if not wait_for_cuda(gpu, ssl):
                print(f"[orchestrate] {ssl}: CUDA never returned within {CUDA_WAIT_S}s", flush=True)
                rc = 2
                break
            rc = _launch(ssl, ckpt, gpu, out_file, log_file)
            attempts += 1
            if rc != 2:
                break
            # Environment fault, not a model fault: wait out the outage and retry
            # this same model rather than letting it consume its queue slot.
            print(f"[orchestrate] {ssl}: rc=2 (CUDA init) attempt {attempts}/{MAX_ATTEMPTS} "
                  f"on gpu {gpu}; waiting for the driver", flush=True)
            time.sleep(30)
        with _lock:
            _results[ssl]["attempts"] = attempts
    dur = time.time() - t0

    rec = dict(_results.get(ssl, {}))
    rec.update({"gpu": gpu, "seconds": round(dur, 1), "output": out_file, "rc": rc})
    rec.pop("started", None)

    if rc != 0 or not os.path.isfile(out_file):
        rec["status"] = "failed"
        tail = ""
        if os.path.isfile(log_file):
            tail = "".join(open(log_file).readlines()[-5:]).strip()
        rec["error"] = tail[-800:]
        with _lock:
            _results[ssl] = rec
        write_status()
        print(f"FAIL {ssl}: rc={rc} | {tail.splitlines()[-1] if tail else ''}", flush=True)
        return

    n_lines = 0
    n_bona = n_spoof = n_nan = 0
    with open(out_file) as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 3)
            if len(parts) < 4:
                continue
            n_lines += 1
            if parts[2] == "bonafide":
                n_bona += 1
            elif parts[2] == "spoof":
                n_spoof += 1
            s = parts[3].lower()
            if "nan" in s or "inf" in s:
                n_nan += 1
    rec.update({"n_lines": n_lines, "n_bonafide": n_bona, "n_spoof": n_spoof,
                "n_nan": n_nan})

    if os.path.isfile(ref_file):
        vlog = os.path.join(LOG_DIR, f"verify_SpoofCeleb_{ssl}.log")
        with open(vlog, "w") as vf:
            vrc = subprocess.call(
                [PY, os.path.join(REPO, "verify_spoofceleb.py"),
                 "--new", out_file, "--ref", ref_file],
                cwd=REPO, stdout=vf, stderr=subprocess.STDOUT)
        vline = open(vlog).read().strip().splitlines()[-1] if os.path.getsize(vlog) else ""
        rec["verify"] = vline
        rec["verify_pass"] = (vrc == 0)
        for k, pat in (("spearman", r"spearman=([-\d.]+)"), ("pearson", r"\br=([-\d.]+)"),
                       ("offset", r"offset=([-+\d.]+)")):
            m = re.search(pat, vline)
            if m:
                rec[k] = m.group(1)
    else:
        rec["verify"] = "no reference"
        rec["verify_pass"] = None

    rec["status"] = "ok" if n_lines == EXPECT_LINES and n_nan == 0 else "suspect"
    with _lock:
        _results[ssl] = rec
    write_status()
    print(f"OK {ssl}: {n_lines} -> {out_file} | "
          f"r={rec.get('pearson','-')} spearman={rec.get('spearman','-')} "
          f"offset={rec.get('offset','-')}", flush=True)


def gpu_worker(gpu, work_q):
    while True:
        try:
            ssl, ckpt = work_q.get_nowait()
        except queue.Empty:
            return
        try:
            run_one(ssl, ckpt, gpu)
        except Exception as e:  # a crashed worker must not take the batch down
            with _lock:
                _results[ssl] = {"status": "failed", "error": f"orchestrator: {e!r}"}
            write_status()
            print(f"FAIL {ssl}: orchestrator exception {e!r}", flush=True)
        finally:
            work_q.task_done()


def main():
    only = set(sys.argv[1:]) or None
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    models = discover_models(only)
    print(f"[orchestrate] {len(models)} models, GPUs={GPUS}, protocol={PROTOCOL}", flush=True)

    # Preserve any results already on disk from a previous (e.g. canary) run.
    if os.path.isfile(STATUS):
        try:
            _results.update(json.load(open(STATUS)))
        except Exception:
            pass
    for ssl, _ in models:
        _results[ssl] = {"status": "pending"}
    write_status()

    work_q = queue.Queue()
    for m in models:
        work_q.put(m)

    threads = [threading.Thread(target=gpu_worker, args=(g, work_q), daemon=True)
               for g in GPUS[:max(1, min(len(GPUS), len(models)))]]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with _lock:
            done = sum(1 for s, r in _results.items()
                       if s in dict(models) and r.get("status") in ("ok", "failed", "suspect"))
            running = [s for s, r in _results.items() if r.get("status") == "running"]
        print(f"[orchestrate] {done}/{len(models)} done; running={running}", flush=True)
        write_status()

    for t in threads:
        t.join()

    lines = [f"{'model':<40} {'#utts':>7} {'spearman':>9} {'pearson':>8} {'offset':>9} {'status':>8}"]
    n_ok = n_fail = n_pass = 0
    for ssl, _ in models:
        r = _results.get(ssl, {})
        st = r.get("status", "?")
        n_ok += st == "ok"
        n_fail += st in ("failed", "suspect")
        n_pass += r.get("verify_pass") is True
        lines.append(f"{ssl:<40} {r.get('n_lines','-'):>7} {r.get('spearman','-'):>9} "
                     f"{r.get('pearson','-'):>8} {r.get('offset','-'):>9} {st:>8}")
        if st in ("failed", "suspect"):
            lines.append(f"    error: {r.get('error', r.get('verify',''))}")
    lines += ["", f"TOTAL: {len(models)} models | ok={n_ok} failed={n_fail} verify_pass={n_pass}", ""]
    for ssl, _ in models:
        r = _results.get(ssl, {})
        if r.get("verify"):
            lines.append(f"{ssl:<40} {r['verify']}")
    text = "\n".join(lines)
    with open(SUMMARY, "w") as f:
        f.write(text + "\n")
    print(text, flush=True)
    write_status()


if __name__ == "__main__":
    main()
