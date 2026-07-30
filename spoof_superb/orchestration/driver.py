"""The single orchestration entry point.

Replaces orchestrate_mlaad.py, orchestrate_mailabs.py, orchestrate_spoofceleb.py
and orchestrate_baselines.py. Those four shared their scheduler, their UUID
pinning, their status file, their resume logic and their summary table, and
differed only in the constants now living in jobs.py.

    python -m spoof_superb.orchestration.driver --job all
    python -m spoof_superb.orchestration.driver --job baselines --workers 1
    python -m spoof_superb.orchestration.driver --job all --datasets spoofceleb
    python -m spoof_superb.orchestration.driver --job all --list

`--workers 1` runs sequentially; the default is a pool over GPUS=[0,1,2].

Three defaults worth knowing:

  * Only the 21 SSL upstreams Table 5 reports are scored (`--all-models` for
    all 24).
  * Each invocation gets its own `_runs/{job}/{run}/`, so two runs of one job
    cannot overwrite each other's record.
  * **No verification happens.** Scoring never reads a score file it did not
    just write, so a fresh tree is built from protocols alone and cannot
    inherit an older tree's coverage. Pass `--verify-against OLD_ROOT` to
    compare as you go, or run spoof_superb.verification.driver afterwards.
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
from spoof_superb.core.scorepath import score_path
from spoof_superb.orchestration import cuda
from spoof_superb.orchestration.jobs import JOBS
from spoof_superb.orchestration.progress import NullReporter, make_reporter
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


def reference_for(task, root, layout):
    """The file in ``root`` that corresponds to this task, or None.

    Resolved from an explicitly named tree, never from the configured
    scores_root. Scoring writes to scores_root; comparing it against itself
    would be meaningless, and comparing it against a tree nobody named is how
    a fresh build silently inherits an old one's coverage.
    """
    if not root:
        return None
    try:
        path = score_path(task.system, task.dataset, task.frontend,
                          scores_root=root, layout=layout)
    except KeyError:
        return None          # the old tree had no convention for this column
    return path if os.path.isfile(path) else None


def _verify(job, task, rec, python, ref_root=None, ref_layout="legacy"):
    """Compare a finished score file against an older tree, if asked.

    Off unless the caller names a reference tree. Verification is a separate
    concern from scoring by design: a score file must be producible without any
    previously produced score file existing.
    """
    ref = reference_for(task, ref_root, ref_layout)
    if not task.verify or not ref:
        rec["verify"] = "not compared" if not ref_root else "no reference"
        rec["verify_pass"] = None
        return
    rec["ref_file"] = ref

    vlog = os.path.join(job.logs(), f"verify_{job.name}_{task.name.replace('/', '_')}.log")
    with open(vlog, "w") as vf:
        vrc = subprocess.call(
            [python, "-m", "spoof_superb.verification.driver",
             "--check", task.verify, "--new", task.out_file, "--ref", ref],
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


def run_task(job, task, gpu, python, force=False, reporter=None, slot=None,
             ref_root=None, ref_layout="legacy"):
    reporter = reporter or NullReporter()
    slot = slot or f"gpu{gpu}"
    log_file = os.path.join(job.logs(), f"{job.name}_{task.name.replace('/', '_')}.log")
    os.makedirs(os.path.dirname(os.path.abspath(task.out_file)), exist_ok=True)
    os.makedirs(job.logs(), exist_ok=True)

    with _lock:
        _results[task.name] = {"status": "running", "gpu": gpu, "started": time.time()}
    _write_status(job)

    t0 = time.time()
    # Decided before the slot is registered: the resume scan reads the whole
    # score file, and a stale log from the previous run would otherwise be
    # displayed as live progress for a task that is only re-verifying.
    resume = not force and output_is_complete(task.out_file, task.expect_lines)
    reporter.start_task(slot, task.name, None if resume else log_file, task.expect_lines)

    if resume:
        reporter.write(f"[orchestrate] {task.name}: existing output is complete; "
                       f"re-verifying only")
        rc = 0
    else:
        open(log_file, "w").close()
        env = cuda.visible_device_env(gpu) if task.needs_gpu else dict(os.environ)
        rc, attempts = 2, 0
        while attempts < job.max_attempts:
            if task.needs_gpu and not cuda.wait_for_cuda(
                    task.name, gpu, wait_s=job.cuda_wait_s, python=python):
                reporter.write(f"[orchestrate] {task.name}: CUDA never returned "
                               f"within {job.cuda_wait_s}s")
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
            reporter.write(f"[orchestrate] {task.name}: rc=2 (CUDA init) attempt "
                           f"{attempts}/{job.max_attempts} on gpu {gpu}")
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
        reporter.finish_task(slot, "failed")
        reporter.write(f"FAIL     {task.name}: rc={rc} | "
                       f"{tail.splitlines()[-1] if tail else ''}")
        return

    # Both of these read the whole score file back; on ASVLD that is minutes,
    # and the display would otherwise show a slot frozen at 100%.
    reporter.set_phase(slot, "count")
    n, n_bona, n_spoof, n_nan = read_score_file(task.out_file)
    rec.update({"n_lines": n, "n_bonafide": n_bona, "n_spoof": n_spoof, "n_nan": n_nan})
    reporter.set_phase(slot, "verify")
    _verify(job, task, rec, python, ref_root=ref_root, ref_layout=ref_layout)

    complete = (task.expect_lines is None) or (n == task.expect_lines)
    rec["status"] = "ok" if complete and n_nan == 0 else "suspect"
    with _lock:
        _results[task.name] = rec
    _write_status(job)
    reporter.finish_task(slot, rec["status"])
    reporter.write(f"{rec['status'].upper():8s} {task.name}: {n:,} lines, "
                   f"{n_nan} NaN, {rec.get('verify', '')}")


def _worker(job, work_q, gpu, python, force, reporter=None, slot=None,
            ref_root=None, ref_layout="legacy"):
    while True:
        try:
            task = work_q.get_nowait()
        except queue.Empty:
            return
        try:
            run_task(job, task, gpu, python, force=force, reporter=reporter,
                     slot=slot, ref_root=ref_root, ref_layout=ref_layout)
        finally:
            # No-op when run_task reported normally; releases the slot if it
            # raised, so one crashed worker cannot stall the counter.
            if reporter:
                reporter.finish_task(slot, "failed")
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
    ap.add_argument("--systems", nargs="*", default=None,
                    help="restrict to these back-ends: linear_head aasist_raw lfcc_gmm")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these datasets")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these SSL upstreams (linear_head only)")
    ap.add_argument("--only", nargs="*", default=None,
                    help=argparse.SUPPRESS)   # deprecated alias for --models
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers; 0 = one per GPU in the job spec, 1 = sequential")
    ap.add_argument("--jobs", type=int, default=None,
                    help=argparse.SUPPRESS)   # deprecated alias for --workers
    ap.add_argument("--gpus", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-score even if the output is complete")
    ap.add_argument("--list", action="store_true", help="list tasks and exit")
    ap.add_argument("--all-models", dest="all_models", action="store_true",
                    help="score every trained head, not just the 21 Table 5 reports")
    ap.add_argument("--verify-against", dest="verify_against", default=None,
                    metavar="SCORES_ROOT",
                    help="after each task, compare its output against the same "
                         "column in this tree. Off by default: scoring never "
                         "reads a score file it did not just write.")
    ap.add_argument("--verify-layout", dest="verify_layout", default="legacy",
                    choices=("legacy", "v2"),
                    help="layout of --verify-against (default: legacy)")
    ap.add_argument("--run-name", dest="run_name", default=None,
                    help="identity for this run; defaults to a timestamp. Two runs "
                         "of one job no longer share a status file.")
    ap.add_argument("--progress", choices=("auto", "bar", "plain", "none"),
                    default="auto",
                    help="auto = redrawing bar on a terminal, periodic lines "
                         "when redirected")
    ap.add_argument("--python", default=cfg.python,
                    help="interpreter for the scoring subprocesses")
    args = ap.parse_args(argv)

    job = JOBS[args.job]
    if args.gpus:
        job.gpus = tuple(args.gpus)
    if args.run_name:
        job.run = args.run_name
    # --jobs wins whenever it is present, rather than only when --workers is
    # absent: bin/orchestrate.sh always emits --workers from its settings block,
    # so "only if --workers is unset" made a hand-typed --jobs silently do
    # nothing through the wrapper. Nothing but a human types --jobs.
    if args.jobs is not None:
        print("[orchestrate] --jobs is deprecated; use --workers "
              "(--job selects the job; --jobs used to mean worker count)")
        args.workers = args.jobs
    if args.workers is None:
        args.workers = 0
    if args.only and not args.models:
        print("[orchestrate] --only is deprecated; use --models (SSL upstreams) "
              "or --datasets")
        args.models = args.only
    tasks = job.enumerate_tasks(systems=args.systems,
                                datasets=args.datasets,
                                models=args.models,
                                paper_only=not args.all_models)

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
    job.link_latest()

    work_q = queue.Queue()
    for t in tasks:
        work_q.put(t)

    n_workers = args.workers if args.workers else len(job.gpus)
    print(f"[orchestrate] {job.name}: {len(tasks)} tasks over {n_workers} worker(s), "
          f"gpus={list(job.gpus)}", flush=True)

    reporter = make_reporter(args.progress, len(tasks), title=job.name)
    reporter.start()

    threads = []
    for i in range(n_workers):
        gpu = job.gpus[i % len(job.gpus)] if job.gpus else 0
        # One line per worker, so the label must be unique even when several
        # workers share a GPU.
        slot = f"gpu{gpu}" if n_workers <= len(job.gpus or [0]) else f"w{i}/gpu{gpu}"
        th = threading.Thread(target=_worker,
                              args=(job, work_q, gpu, args.python, args.force,
                                    reporter, slot, args.verify_against,
                                    args.verify_layout),
                              daemon=True)
        th.start()
        threads.append(th)
        time.sleep(2)   # stagger CUDA context creation
    for th in threads:
        th.join()

    reporter.stop()
    _write_status(job)
    write_summary(job, tasks)

    failed = [n for n, r in _results.items() if r.get("status") != "ok"]
    if failed:
        print(f"[orchestrate] {len(failed)} task(s) not ok: {', '.join(sorted(failed))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
