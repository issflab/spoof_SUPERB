"""
orchestrate_mailabs.py
----------------------
Run eval_mlaad.py over the full M-AILABS bonafide set (584,012 wavs) for the same
22 linear-head models, scheduling greedily across 3 GPUs.

Scores are written to a STAGING dir (mailabs/), not appended in place. Appending
is a separate, guarded step (append_mailabs.py) so that a crash mid-run can never
leave a half-appended MLAAD score file.

Verification: each output is compared against the Multilingual reference. The
intersection is the reference's bonafide rows (~153,998 of the 584,012 we score),
since the reference sampled ~26.4% of M-AILABS to balance its MLAAD spoof count.
"""
import json
import os
import queue
import random
import subprocess
import threading
import time

REPO = "/home/alhashim/ASD_SUPERB/spoof_SUPERB"
PY = "/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python"
MODELS_ROOT = "/data/ssl_anti_spoofing/asd_superb_models/linear_head_models"
BASE_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10"
OUT_DIR = os.path.join(BASE_DIR, "mailabs")
REF_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head"
LOG_DIR = os.path.join(OUT_DIR, "logs")
STATUS = os.path.join(OUT_DIR, "run_status.json")
SUMMARY = os.path.join(OUT_DIR, "SUMMARY.txt")
# Only one GPU *device* can be active per session (a second device fails CUDA
# init), but multiple processes may share one device. So: two workers, both on
# GPU 0. Heaviest model needs ~6 GB, so 2 concurrent fits 40 GB comfortably.
GPUS = [0, 1, 2]
PREFIX = "model_weighted_CCE_50_64_linear_head_ASV19_"
SKIP = {"byol_a_2048", "mockingjay"}          # skipped per user request
MAILABS_ROOT = "/data/Data/MAILabs"

def gpu_uuids():
    """Map GPU index -> UUID.

    CUDA_VISIBLE_DEVICES by *index* fails to initialise on this host once another
    process holds a different device ("CUDA driver initialization failed"), which
    previously sent whole models to CPU. Selecting by UUID is unaffected and lets
    all three GPUs run concurrently, so every launch pins its device by UUID.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).decode()
    return [u.strip() for u in out.splitlines() if u.strip()]


_GPU_UUIDS = gpu_uuids()

_lock = threading.Lock()
_results = {}


def discover_models():
    out = []
    for name in sorted(os.listdir(MODELS_ROOT)):
        d = os.path.join(MODELS_ROOT, name)
        ckpt = os.path.join(d, "swa.pth")
        if os.path.isdir(d) and name.startswith(PREFIX) and os.path.isfile(ckpt):
            ssl = name[len(PREFIX):]
            if ssl in SKIP:
                continue
            out.append((ssl, ckpt))
    return out


def write_status():
    with _lock:
        snap = dict(_results)
        tmp = f"{STATUS}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        os.replace(tmp, STATUS)


def run_one(ssl, ckpt, gpu):
    out_file = os.path.join(OUT_DIR, f"linear_head_MAILABS_{ssl}.txt")
    log_file = os.path.join(LOG_DIR, f"eval_MAILABS_{ssl}.log")
    ref_file = os.path.join(REF_DIR, f"linear_head_Multilingual_{ssl}.txt")

    # Resume support: a complete output from an earlier run is not redone.
    if os.path.isfile(out_file) and os.path.getsize(out_file) > 0:
        n = sum(1 for _ in open(out_file))
        if n >= 584006:
            with _lock:
                _results[ssl] = {"status": "ok", "gpu": gpu, "n_lines": n,
                                 "seconds": 0, "output": out_file,
                                 "verify": "skipped (already complete)"}
            write_status()
            return

    with _lock:
        _results[ssl] = {"status": "running", "gpu": gpu, "started": time.time()}
    write_status()

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_GPU_UUIDS[gpu])
    cmd = [PY, "-m", "spoof_superb.scoring.driver",
           "--model", "linear_head", "--source", "walk",
           "--model_path", ckpt, "--ssl_model", ssl,
           "--walk_root", MAILABS_ROOT, "--label", "bonafide",
           "--output_file", out_file, "--cuda_device", "cuda:0",
           "--batch_size", "32", "--num_workers", "6"]

    # rc==2 is eval_mlaad's "CUDA requested but unavailable" guard. Simultaneous
    # CUDA context creation from several processes can transiently fail the driver
    # init, so retry in a fresh process after a pause rather than losing the model.
    t0 = time.time()
    rc = None
    for attempt in range(4):
        with open(log_file, "w") as lf:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
        if rc != 2:
            break
        wait = 30 * (attempt + 1) + random.uniform(0, 10)
        print(f"[mailabs] {ssl}: CUDA init failed on gpu {gpu}, "
              f"retry {attempt + 1}/3 in {wait:.0f}s", flush=True)
        time.sleep(wait)
    dur = time.time() - t0

    rec = {"gpu": gpu, "seconds": round(dur, 1), "output": out_file}
    if rc != 0 or not os.path.isfile(out_file):
        rec["status"] = "failed"
        rec["rc"] = rc
        with _lock:
            _results[ssl] = rec
        write_status()
        return

    rec["n_lines"] = sum(1 for _ in open(out_file))

    if os.path.isfile(ref_file):
        vlog = os.path.join(LOG_DIR, f"verify_MAILABS_{ssl}.log")
        with open(vlog, "w") as vf:
            vrc = subprocess.call(
                [PY, os.path.join(REPO, "verify_mlaad.py"),
                 "--new", out_file, "--ref", ref_file],
                cwd=REPO, stdout=vf, stderr=subprocess.STDOUT)
        lines = open(vlog).read().strip().splitlines()
        rec["verify"] = lines[-1] if lines else ""
        rec["verify_pass"] = (vrc == 0)
    else:
        rec["verify"] = "no reference"
        rec["verify_pass"] = None

    rec["status"] = "ok"
    with _lock:
        _results[ssl] = rec
    write_status()


def gpu_worker(gpu, work_q, start_delay=0.0):
    # Stagger the first CUDA context creation per GPU; three processes initialising
    # the driver in the same instant is what triggers the transient init failure.
    if start_delay:
        time.sleep(start_delay)
    while True:
        try:
            ssl, ckpt = work_q.get_nowait()
        except queue.Empty:
            return
        try:
            run_one(ssl, ckpt, gpu)
        finally:
            work_q.task_done()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    models = discover_models()
    print(f"[mailabs] {len(models)} models, GPUs={GPUS}", flush=True)
    for ssl, _ in models:
        _results.setdefault(ssl, {"status": "pending"})
    write_status()

    work_q = queue.Queue()
    for m in models:
        work_q.put(m)

    threads = [threading.Thread(target=gpu_worker, args=(g, work_q, i * 25.0), daemon=True)
               for i, g in enumerate(GPUS)]
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(20)
        with _lock:
            done = sum(1 for r in _results.values() if r.get("status") in ("ok", "failed"))
            running = [s for s, r in _results.items() if r.get("status") == "running"]
        print(f"[mailabs] {done}/{len(models)} done; running={running}", flush=True)
        write_status()

    for t in threads:
        t.join()

    lines = [f"{'model':<40} {'#utts':>8} {'sec':>7} {'status':>7}  verify"]
    n_ok = n_fail = 0
    for ssl, _ in models:
        r = _results.get(ssl, {})
        st = r.get("status", "?")
        n_ok += st == "ok"
        n_fail += st == "failed"
        lines.append(f"{ssl:<40} {r.get('n_lines','-'):>8} {r.get('seconds','-'):>7} "
                     f"{st:>7}  {r.get('verify','')}")
    lines.append("")
    lines.append(f"TOTAL: {len(models)} models | ok={n_ok} failed={n_fail}")
    text = "\n".join(lines)
    with open(SUMMARY, "w") as f:
        f.write(text + "\n")
    print(text, flush=True)
    write_status()


if __name__ == "__main__":
    main()
