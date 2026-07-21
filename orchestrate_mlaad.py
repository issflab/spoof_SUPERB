"""
orchestrate_mlaad.py
--------------------
Run eval_mlaad.py for all 24 linear-head models over the full MLAAD v10 fake set,
scheduling models greedily across N GPUs (one model per GPU at a time). After each
model finishes, verify its output against the Multilingual reference and record the
result. Writes a live status file and a final summary table.

Nothing here mutates main.py/config.py/data_utils_SSL.py or the MLAAD data.
"""
import json
import os
import queue
import subprocess
import threading
import time

REPO = "/home/alhashim/ASD_SUPERB/spoof_SUPERB"
PY = "/home/alhashim/miniconda3/envs/spoof_SUPERB/bin/python"
MODELS_ROOT = "/data/ssl_anti_spoofing/asd_superb_models/linear_head_models"
OUT_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10"
REF_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head"
LOG_DIR = os.path.join(OUT_DIR, "logs")
STATUS = os.path.join(OUT_DIR, "run_status.json")
SUMMARY = os.path.join(OUT_DIR, "SUMMARY.txt")
GPUS = [0, 1, 2]
PREFIX = "model_weighted_CCE_50_64_linear_head_ASV19_"
# Models to skip (per user request). mockingjay_960hr is intentionally kept.
SKIP = {"byol_a_2048", "mockingjay"}

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
    # Serialize the whole snapshot+write under the lock and use a per-thread tmp
    # name; two threads sharing one tmp path race on os.replace otherwise.
    with _lock:
        snap = dict(_results)
        tmp = f"{STATUS}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        os.replace(tmp, STATUS)


def run_one(ssl, ckpt, gpu):
    out_file = os.path.join(OUT_DIR, f"linear_head_MLAAD_v10_{ssl}.txt")
    log_file = os.path.join(LOG_DIR, f"eval_MLAAD_v10_{ssl}.log")
    ref_file = os.path.join(REF_DIR, f"linear_head_Multilingual_{ssl}.txt")

    with _lock:
        _results[ssl] = {"status": "running", "gpu": gpu, "started": time.time()}
    write_status()

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [PY, os.path.join(REPO, "eval_mlaad.py"),
           "--model_path", ckpt, "--ssl_model", ssl,
           "--output_file", out_file, "--cuda_device", "cuda:0",
           "--batch_size", "32", "--num_workers", "6"]  # fp32: autocast NaNs spectrogram upstreams

    t0 = time.time()
    with open(log_file, "w") as lf:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
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

    # Verify against reference (detection-equivalence).
    if os.path.isfile(ref_file):
        vlog = os.path.join(LOG_DIR, f"verify_MLAAD_v10_{ssl}.log")
        with open(vlog, "w") as vf:
            vrc = subprocess.call(
                [PY, os.path.join(REPO, "verify_mlaad.py"),
                 "--new", out_file, "--ref", ref_file],
                cwd=REPO, stdout=vf, stderr=subprocess.STDOUT)
        vline = open(vlog).read().strip().splitlines()[-1] if os.path.getsize(vlog) else ""
        rec["verify"] = vline
        rec["verify_pass"] = (vrc == 0)
    else:
        rec["verify"] = "no reference"
        rec["verify_pass"] = None

    rec["status"] = "ok"
    with _lock:
        _results[ssl] = rec
    write_status()


def gpu_worker(gpu, work_q):
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
    print(f"[orchestrate] {len(models)} models, GPUs={GPUS}", flush=True)
    for ssl, _ in models:
        _results[ssl] = {"status": "pending"}
    write_status()

    work_q = queue.Queue()
    for m in models:
        work_q.put(m)

    threads = [threading.Thread(target=gpu_worker, args=(g, work_q), daemon=True)
               for g in GPUS]
    for t in threads:
        t.start()

    # Progress heartbeat until all workers drain the queue.
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        with _lock:
            done = sum(1 for r in _results.values() if r.get("status") in ("ok", "failed"))
            running = [s for s, r in _results.items() if r.get("status") == "running"]
        print(f"[orchestrate] {done}/{len(models)} done; running={running}", flush=True)
        write_status()

    for t in threads:
        t.join()

    # Final summary table.
    lines = []
    lines.append(f"{'model':<40} {'#utts':>8} {'sec':>7} {'status':>7}  verify")
    n_ok = n_fail = n_pass = 0
    for ssl, _ in models:
        r = _results.get(ssl, {})
        st = r.get("status", "?")
        n_ok += st == "ok"
        n_fail += st == "failed"
        vp = r.get("verify_pass")
        n_pass += vp is True
        lines.append(f"{ssl:<40} {r.get('n_lines','-'):>8} {r.get('seconds','-'):>7} "
                     f"{st:>7}  {r.get('verify','')}")
    lines.append("")
    lines.append(f"TOTAL: {len(models)} models | ok={n_ok} failed={n_fail} "
                 f"verify_pass={n_pass}")
    text = "\n".join(lines)
    with open(SUMMARY, "w") as f:
        f.write(text + "\n")
    print(text, flush=True)
    write_status()


if __name__ == "__main__":
    main()
