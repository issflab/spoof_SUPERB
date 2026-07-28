"""Job specifications: what the four orchestrators actually differed in.

orchestrate_mlaad.py, orchestrate_mailabs.py, orchestrate_spoofceleb.py and
orchestrate_baselines.py were the same program four times. Everything that
genuinely varied between them is data, and it is here; everything that did not
is in driver.py.

A job answers four questions:
  * which tasks exist            (enumerate)
  * what command scores one      (task.argv)
  * where the output and its reference live
  * how completion is judged     (expect_lines, and the verify policy)
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from spoof_superb.config import cfg
from spoof_superb.core.scorepath import score_path
from spoof_superb.scoring.datasets import (
    DATASETS,
    MAILABS_ROOT,
    SCORES_ROOT,
    SPOOFCELEB_AUDIO,
    SPOOFCELEB_PROTOCOL,
)

MODELS_ROOT = cfg.models_root
BASELINE_MODELS_ROOT = cfg.baseline_models_root
LINEAR_HEAD_PREFIX = cfg.linear_head_prefix

DRIVER = ("-m", "spoof_superb.scoring.driver")


@dataclass
class Task:
    """One unit of work: score one model on one set."""
    name: str
    argv: Sequence[str]
    out_file: str
    ref_file: Optional[str] = None
    needs_gpu: bool = True


@dataclass
class Job:
    name: str
    out_dir: str
    enumerate_tasks: Callable[["Job", Optional[Sequence[str]]], list]
    log_dir: Optional[str] = None
    ref_dir: Optional[str] = None
    expect_lines: Optional[int] = None
    verify: Optional[str] = None          # verification policy name
    skip: frozenset = frozenset()
    gpus: tuple = (0, 1, 2)
    max_attempts: int = 3
    cuda_wait_s: int = 3600
    batch_size: int = 32
    num_workers: int = 6
    extra: dict = field(default_factory=dict)

    def logs(self):
        return self.log_dir or os.path.join(self.out_dir, "logs")

    def status_path(self):
        return os.path.join(self.out_dir, "run_status.json")

    def summary_path(self):
        return os.path.join(self.out_dir, "SUMMARY.txt")


# ===========================================================================
# Model discovery
# ===========================================================================

def discover_linear_heads(only=None, skip=frozenset()):
    """(ssl_name, checkpoint) for every trained linear head on disk."""
    out = []
    if not os.path.isdir(MODELS_ROOT):
        return out
    for name in sorted(os.listdir(MODELS_ROOT)):
        d = os.path.join(MODELS_ROOT, name)
        ckpt = os.path.join(d, "swa.pth")
        if not (os.path.isdir(d) and name.startswith(LINEAR_HEAD_PREFIX)
                and os.path.isfile(ckpt)):
            continue
        ssl = name[len(LINEAR_HEAD_PREFIX):]
        if ssl in skip or (only and ssl not in only):
            continue
        out.append((ssl, ckpt))
    return out


def resolve_baseline_model(model):
    """Checkpoint path for a non-SSL baseline."""
    if model == "lfcc_gmm":
        return os.path.join(BASELINE_MODELS_ROOT, "lfcc_gmm")
    run_dir = os.path.join(BASELINE_MODELS_ROOT,
                           "model_weighted_CCE_50_64_aasist_raw_ASV19_none")
    return os.path.join(run_dir, "swa.pth")


# ===========================================================================
# Task enumeration, one function per job shape
# ===========================================================================

def _linear_head_tasks(job, only, source_argv, dataset, ref_name):
    """Tasks for one linear-head sweep.

    Output paths come from core.scorepath, so the on-disk layout is decided in
    one place and follows cfg.score_layout.
    """
    tasks = []
    for ssl, ckpt in discover_linear_heads(only=only, skip=job.skip):
        out_file = score_path("linear_head", dataset, ssl)
        ref_file = (os.path.join(job.ref_dir, ref_name.format(ssl=ssl))
                    if job.ref_dir and ref_name else None)
        argv = [*DRIVER, "--model", "linear_head",
                "--model_path", ckpt, "--ssl_model", ssl,
                "--output_file", out_file, "--cuda_device", "cuda:0",
                "--batch_size", str(job.batch_size),
                "--num_workers", str(job.num_workers),
                *source_argv]
        tasks.append(Task(name=ssl, argv=argv, out_file=out_file, ref_file=ref_file))
    return tasks


def mlaad_tasks(job, only=None):
    return _linear_head_tasks(
        job, only, ["--source", "walk"],
        "Multilingual", "linear_head_Multilingual_{ssl}.txt")


def mailabs_tasks(job, only=None):
    return _linear_head_tasks(
        job, only, ["--source", "walk", "--walk_root", MAILABS_ROOT,
                    "--label", "bonafide"],
        "MAILABS", "linear_head_Multilingual_{ssl}.txt")


def spoofceleb_tasks(job, only=None):
    return _linear_head_tasks(
        job, only, ["--source", "protocol_csv",
                    "--protocol_csv", SPOOFCELEB_PROTOCOL,
                    "--audio_base", SPOOFCELEB_AUDIO],
        "spoofceleb", "linear_head_spoofceleb_{ssl}.txt")


# Smallest-first, so a failure surfaces in seconds rather than after the 1.6M-row
# ASVLD run.
BASELINE_ORDER = ["deepfake_eval_2024", "wild", "eval_2019", "spoofceleb",
                  "asvspoof2021_DF", "asvspoof2021_LA", "Famous_Figures",
                  "asvspoof5", "Multilingual", "asvspoofLD"]


def baseline_tasks(job, only=None):
    """(model, dataset) pairs for the two non-SSL baselines."""
    models = job.extra.get("models") or ["aasist_raw", "lfcc_gmm"]
    datasets = job.extra.get("datasets") or BASELINE_ORDER
    tasks = []
    for model in models:
        ckpt = resolve_baseline_model(model)
        for ds in datasets:
            if ds not in DATASETS or (only and ds not in only):
                continue
            out_file = score_path(model, ds, "none")
            argv = [*DRIVER, "--model", model, "--model_path", ckpt,
                    "--dataset", ds, "--output_file", out_file,
                    "--cuda_device", "cuda:0",
                    "--batch_size", str(job.batch_size),
                    "--n_jobs", str(job.extra.get("n_jobs", 16))]
            tasks.append(Task(name=f"{model}/{ds}", argv=argv, out_file=out_file,
                              needs_gpu=(model == "aasist_raw")))
    return tasks


# ===========================================================================
# The registry
# ===========================================================================

JOBS = {
    # byol_a_2048 and mockingjay are skipped on the MLAAD/M-AILABS runs per an
    # earlier request; SpoofCeleb deliberately keeps them (fp32 fixes byol's
    # fp16 STFT crash and mockingjay's SpoofCeleb reference is usable).
    "mlaad": Job(
        name="mlaad",
        out_dir=os.path.join(SCORES_ROOT, "linear_head_MLAAD_v10"),
        ref_dir=os.path.join(SCORES_ROOT, "linear_head"),
        enumerate_tasks=mlaad_tasks,
        verify="mlaad",
        skip=frozenset({"byol_a_2048", "mockingjay"}),
        max_attempts=1,
    ),
    "mailabs": Job(
        name="mailabs",
        # A staging dir, not an in-place append: a crash mid-run must never
        # leave a half-appended MLAAD score file. Appending is the separate,
        # guarded data/prep/append_mailabs.py step.
        out_dir=os.path.join(SCORES_ROOT, "linear_head_MLAAD_v10", "mailabs"),
        ref_dir=os.path.join(SCORES_ROOT, "linear_head"),
        enumerate_tasks=mailabs_tasks,
        verify="mlaad",
        skip=frozenset({"byol_a_2048", "mockingjay"}),
        max_attempts=1,
    ),
    "spoofceleb": Job(
        name="spoofceleb",
        out_dir=os.path.join(SCORES_ROOT, "linear_head_SpoofCeleb"),
        ref_dir=os.path.join(SCORES_ROOT, "linear_head"),
        enumerate_tasks=spoofceleb_tasks,
        verify="spoofceleb",
        expect_lines=91130,
        max_attempts=6,
        cuda_wait_s=3 * 3600,
    ),
    "baselines": Job(
        name="baselines",
        out_dir=os.path.join(SCORES_ROOT, "baselines"),
        enumerate_tasks=baseline_tasks,
        batch_size=64,
        max_attempts=3,
        cuda_wait_s=3600,
    ),
}
