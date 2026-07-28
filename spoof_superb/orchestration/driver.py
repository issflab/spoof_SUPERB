"""The single orchestration entry point.

Replaces orchestrate_mlaad.py, orchestrate_mailabs.py, orchestrate_spoofceleb.py
and orchestrate_baselines.py. Those four shared their scheduler, their UUID
pinning, their status file, their resume logic and their summary table, and
differed only in the constants now living in jobs.py.

    python -m spoof_superb.orchestration.driver --job spoofceleb
    python -m spoof_superb.orchestration.driver --job baselines --jobs 1
    python -m spoof_superb.orchestration.driver --job mlaad --only xls_r_300m
    python -m spoof_superb.orchestration.driver --job baselines --list

`--jobs 1` reproduces orchestrate_baselines.py's sequential behaviour; the
other three ran a pool over GPUS=[0,1,2], which is the default.
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

from spoof_superb.config import cfg
from spoof_superb.orchestration import cuda
from spoof_superb.orchestration.jobs import JOBS
from spoof_superb.scoring.datasets import PROTOCOL_SPECS

_lock = threading.Lock()
_results = {}


def _write_status(job):
    """Snapshot and write under the lock.

    The tmp name carries the thread id: two threads sharing one tmp path race
    on os.replace and can leave a truncated status file.
    """
    with _lock:
        snap = dict(_results)
    path = job.status_path()
    tmp = f"{path}.{threading.get_ident()}.tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f, indent=2)
    os.replace(tmp, path)


def read_score_file(path):
    """(n_lines, n_bonafide, n_spoof, n_nonfinite) for a canonical score file."""
    n = n_bona = n_spoof = n_nan = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 3)
            if len(parts) < 4:
                continue
            n += 1
            if parts[2] == "bonafide":
                n_bona += 1
            elif parts[2] == "spoof":
                n_spoof += 1
            s = parts[3].lower()
            if "nan" in s or "inf" in s:
                n_nan += 1
    return n, n_bona, n_spoof, n_nan


def output_is_complete(path, expect_lines):
    """A finished, NaN-free score file -- lets a restart resume instead of redo."""
    if not os.path.isfile(path):
        return False
    n, _, _, n_nan = read_score_file(path)
    if n_nan:
        return False
    return n > 0 if expect_lines is None else n == expect_lines


def _verify(job, task, rec, python):
    """Run the job's verification policy against the reference, if there is one."""
    if not job.verify or not task.ref_file or not os.path.isfile(task.ref_file):
        rec["verify"] = "no reference"
        rec["verify_pass"] = None
        return

    vlog = os.path.join(job.logs(), f"verify_{job.name}_{task.name.replace('/', '_')}.log")
    with open(vlog, "w") as vf:
        vrc = subprocess.call(
            [python, "-m", "spoof_superb.verification.driver",
             "--check", job.verify, "--new", task.out_file, "--ref", task.ref_file],
            stdout=vf, stderr=subprocess.STDOUT)
    lines = open(vlog).read().strip().splitlines()
    vline = lines[-1] if lines else ""
    rec["verify"] = vline
    rec["verify_pass"] = (vrc == 0)
    for key, pat in (("spearman", r"spearman=([-\d.]+)"), ("pearson", r"\br=([-\d.]+)"),
                     ("offset", r"offset=([-+\d.]+)")):
        m = re.search(pat, vline)
        if m:
            rec[key] = m.group(1)


def run_task(job, task, gpu, python, force=False):
    log_file = os.path.join(job.logs(), f"{job.name}_{task.name.replace('/', '_')}.log")
    os.makedirs(os.path.dirname(os.path.abspath(task.out_file)), exist_ok=True)
    os.makedirs(job.logs(), exist_ok=True)

    with _lock:
        _results[task.name] = {"status": "running", "gpu": gpu, "started": time.time()}
    _write_status(job)

    t0 = time.time()
    if not force and output_is_complete(task.out_file, task.expect_lines):
        print(f"[orchestrate] {task.name}: existing output is complete; "
              f"re-verifying only", flush=True)
        rc = 0
    else:
        open(log_file, "w").close()
        env = cuda.visible_device_env(gpu) if task.needs_gpu else dict(os.environ)
        rc, attempts = 2, 0
        while attempts < job.max_attempts:
            if task.needs_gpu and not cuda.wait_for_cuda(
                    task.name, gpu, wait_s=job.cuda_wait_s, python=python):
                print(f"[orchestrate] {task.name}: CUDA never returned within "
                      f"{job.cuda_wait_s}s", flush=True)
                rc = 2
                break
            with open(log_file, "a") as lf:
                lf.write(f"\n=== launch gpu={gpu} {time.ctime()} ===\n")
                lf.flush()
                rc = subprocess.call([python, "-u", *task.argv], env=env,
                                     stdout=lf, stderr=subprocess.STDOUT)
            attempts += 1
            if rc != 2:
                break
            # rc=2 is the driver's "CUDA requested but unavailable" guard: an
            # environment fault, not a model fault. Retry this same task in a
            # fresh process rather than losing it.
            print(f"[orchestrate] {task.name}: rc=2 (CUDA init) attempt "
                  f"{attempts}/{job.max_attempts} on gpu {gpu}", flush=True)
            time.sleep(30)
        with _lock:
            _results[task.name]["attempts"] = attempts

    rec = dict(_results.get(task.name, {}))
    rec.update({"gpu": gpu, "seconds": round(time.time() - t0, 1),
                "output": task.out_file, "rc": rc})
    rec.pop("started", None)

    if rc != 0 or not os.path.isfile(task.out_file):
        rec["status"] = "failed"
        tail = ""
        if os.path.isfile(log_file):
            tail = "".join(open(log_file).readlines()[-5:]).strip()
        rec["error"] = tail[-800:]
        with _lock:
            _results[task.name] = rec
        _write_status(job)
        print(f"FAIL {task.name}: rc={rc} | "
              f"{tail.splitlines()[-1] if tail else ''}", flush=True)
        return

    n, n_bona, n_spoof, n_nan = read_score_file(task.out_file)
    rec.update({"n_lines": n, "n_bonafide": n_bona, "n_spoof": n_spoof, "n_nan": n_nan})
    _verify(job, task, rec, python)

    complete = (task.expect_lines is None) or (n == task.expect_lines)
    rec["status"] = "ok" if complete and n_nan == 0 else "suspect"
    with _lock:
        _results[task.name] = rec
    _write_status(job)
    print(f"{rec['status'].upper():8s} {task.name}: {n} lines, {n_nan} NaN, "
          f"{rec.get('verify', '')}", flush=True)


def _worker(job, work_q, gpu, python, force):
    while True:
        try:
            task = work_q.get_nowait()
        except queue.Empty:
            return
        try:
            run_task(job, task, gpu, python, force=force)
        finally:
            work_q.task_done()


def write_summary(job, tasks):
    path = job.summary_path()
    with open(path, "w") as f:
        f.write(f"{job.name}: {len(tasks)} tasks\n\n")
        f.write(f"{'task':40s} {'status':9s} {'lines':>10s} {'NaN':>8s}  verify\n")
        for t in tasks:
            r = _results.get(t.name, {})
            f.write(f"{t.name:40s} {r.get('status', '?'):9s} "
                    f"{r.get('n_lines', 0):>10} {r.get('n_nan', 0):>8}  "
                    f"{r.get('verify', '')}\n")
    print(f"\nSummary -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.orchestration.driver",
        description="Run a scoring job across models/datasets and GPUs")
    ap.add_argument("--job", choices=sorted(JOBS), required=True,
                    help="all = every system on every dataset")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict the sweep to these datasets")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these task keys (SSL model, or dataset for baselines)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel workers; 0 = one per GPU in the job spec, 1 = sequential")
    ap.add_argument("--gpus", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-score even if the output is complete")
    ap.add_argument("--list", action="store_true", help="list tasks and exit")
    ap.add_argument("--python", default=cfg.python,
                    help="interpreter for the scoring subprocesses")
    args = ap.parse_args(argv)

    job = JOBS[args.job]
    if args.gpus:
        job.gpus = tuple(args.gpus)
    if args.datasets:
        job.datasets = args.datasets
    tasks = job.enumerate_tasks(args.only)

    if not tasks:
        print(f"[orchestrate] {job.name}: no tasks (checked {job.out_dir})")
        return 1

    # Pre-flight: a dataset whose protocol is missing cannot be scored, and
    # finding that out 200 tasks into an overnight sweep is expensive.
    missing = {}
    for t in tasks:
        spec = PROTOCOL_SPECS.get(t.dataset, {})
        path = spec.get("protocol")
        if path and not os.path.isfile(path):
            missing[t.dataset] = spec.get("built_by")
    if missing:
        print(f"[orchestrate] {len(missing)} dataset(s) have no protocol on disk; "
              f"their tasks will fail:")
        for ds, built_by in sorted(missing.items()):
            hint = f"  build it with: {built_by}" if built_by else ""
            print(f"    {ds}{hint}")
        print("    (pass --datasets to exclude them)")
    if args.list:
        for t in tasks:
            print(f"{t.name:40s} -> {t.out_file}")
        return 0

    os.makedirs(job.out_dir, exist_ok=True)
    os.makedirs(job.logs(), exist_ok=True)

    work_q = queue.Queue()
    for t in tasks:
        work_q.put(t)

    n_workers = args.jobs if args.jobs else len(job.gpus)
    print(f"[orchestrate] {job.name}: {len(tasks)} tasks over {n_workers} worker(s), "
          f"gpus={list(job.gpus)}", flush=True)

    threads = []
    for i in range(n_workers):
        gpu = job.gpus[i % len(job.gpus)] if job.gpus else 0
        th = threading.Thread(target=_worker, args=(job, work_q, gpu, args.python, args.force),
                              daemon=True)
        th.start()
        threads.append(th)
        time.sleep(2)   # stagger CUDA context creation
    for th in threads:
        th.join()

    _write_status(job)
    write_summary(job, tasks)

    failed = [n for n, r in _results.items() if r.get("status") != "ok"]
    if failed:
        print(f"[orchestrate] {len(failed)} task(s) not ok: {', '.join(sorted(failed))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
