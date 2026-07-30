"""Job specifications: which (system, dataset, frontend) combinations to score.

Once every dataset resolves its own trial list from a protocol and every output
path comes from score_path, a task is fully described by three things:

    system    linear_head | aasist_raw | lfcc_gmm
    dataset   a registry key
    frontend  the s3prl upstream, or 'none'

so there is one enumerator rather than one per job. A job is a selection over
that space plus the runtime policy: GPUs, retries, verification.

That is what makes `--job all` possible. It is not a special case; it is the
selection with nothing excluded.
"""

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

from spoof_superb.config import cfg
from spoof_superb.core.scorepath import score_path
from spoof_superb.scoring.datasets import (
    DEFAULT_DATASETS,
    PROTOCOL_SPECS,
    SCOREABLE,
    verify_policy,
)
from spoof_superb.scoring.models import paper_models

MODELS_ROOT = cfg.models_root
BASELINE_MODELS_ROOT = cfg.baseline_models_root
LINEAR_HEAD_PREFIX = cfg.linear_head_prefix

DRIVER = ("-m", "spoof_superb.scoring.driver")
BASELINE_SYSTEMS = ("aasist_raw", "lfcc_gmm")

# Smallest-first, so a failure surfaces in seconds rather than after the
# 2M-row ASVLD run.
DATASET_ORDER = [
    "deepfake_eval_2024", "wild", "eval_2019", "spoofceleb",
    "asvspoof2021_LA", "deepfake_eval_2024_segmented", "Famous_Figures",
    "MAILABS", "asvspoof2021_DF", "asvspoof5", "Multilingual", "asvspoofLD",
]


def ordered_datasets(datasets=None):
    """Requested datasets, smallest-first, unknown names dropped.

    With nothing requested this is DEFAULT_DATASETS, not every scoreable set:
    the two Deepfake-Eval variants measure the same corpus two ways and only the
    segmented one belongs in a sweep by default. Naming either explicitly works.
    """
    wanted = list(datasets) if datasets else list(DEFAULT_DATASETS)
    known = [d for d in DATASET_ORDER if d in wanted]
    return known + [d for d in wanted if d not in DATASET_ORDER and d in SCOREABLE]


@lru_cache(maxsize=None)
def expected_rows(dataset):
    """How many trials the protocol declares, or None if it cannot be read.

    Derived rather than hardcoded. A resume check that knows the real row count
    works for every dataset, instead of only the one whose number someone
    happened to write down.
    """
    spec = PROTOCOL_SPECS.get(dataset)
    if not spec:
        return None
    path = spec.get("protocol")
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1) if spec.get("header", True) else n


@dataclass
class Task:
    name: str
    argv: Sequence[str]
    out_file: str
    dataset: str
    system: str = "linear_head"
    frontend: str = "none"
    needs_gpu: bool = True
    expect_lines: Optional[int] = None
    # Which policy would grade this task if the user asks for a comparison.
    # Carried per task because it is a property of the dataset, not the sweep.
    verify: Optional[str] = None


def default_run_name():
    """A run identity that sorts chronologically and never collides."""
    return time.strftime("%Y%m%d-%H%M%S")


@dataclass
class Job:
    name: str
    systems: Sequence[str] = ("linear_head",)
    datasets: Optional[Sequence[str]] = None      # None = every scoreable one
    gpus: tuple = (0, 1, 2)
    max_attempts: int = 3
    cuda_wait_s: int = 3600
    batch_size: int = 0                           # 0 = the back-end default
    num_workers: int = 6
    gmm_processes: int = 16                       # LFCC-GMM pool only
    log_dir: Optional[str] = None
    # Which invocation this is. Keyed separately from the job name because two
    # runs of the same job used to share one directory: the second overwrote
    # the first's run_status.json, and their logs interleaved silently.
    run: str = field(default_factory=default_run_name)

    @property
    def job_dir(self):
        """Everything this job has ever produced, across runs."""
        return os.path.join(cfg.scores_root, "_runs", self.name)

    @property
    def out_dir(self):
        """This one run's status, summary and logs -- not the score files."""
        return os.path.join(self.job_dir, self.run)

    def logs(self):
        return self.log_dir or os.path.join(self.out_dir, "logs")

    def status_path(self):
        return os.path.join(self.out_dir, "run_status.json")

    def summary_path(self):
        return os.path.join(self.out_dir, "SUMMARY.txt")

    def link_latest(self):
        """Point {job_dir}/latest at this run.

        Keeps every documented path and any existing tooling working: what used
        to be {job_dir}/run_status.json is now {job_dir}/latest/run_status.json,
        one level deeper but still a fixed location.
        """
        link = os.path.join(self.job_dir, "latest")
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(self.run, link)
        except OSError:
            pass          # a status symlink is never worth failing a sweep for
        return link

    def enumerate_tasks(self, systems=None, datasets=None, models=None,
                        paper_only=None):
        return enumerate_tasks(self, systems=systems, datasets=datasets,
                               models=models, paper_only=paper_only)


# ===========================================================================
# Model discovery
# ===========================================================================

def discover_linear_heads(only=None, paper_only=False):
    """(ssl_name, checkpoint) for every trained linear head on disk.

    ``paper_only`` keeps just the upstreams the paper's results table reports.
    It is applied only when nothing was asked for by name: naming a model with
    ``only`` is an explicit request, and a filter that silently discarded it
    would be a trap of the same kind as the old --only overload.
    """
    out = []
    if not os.path.isdir(MODELS_ROOT):
        return out
    keep = paper_models() if paper_only else None
    for name in sorted(os.listdir(MODELS_ROOT)):
        d = os.path.join(MODELS_ROOT, name)
        ckpt = os.path.join(d, "swa.pth")
        if not (os.path.isdir(d) and name.startswith(LINEAR_HEAD_PREFIX)
                and os.path.isfile(ckpt)):
            continue
        ssl = name[len(LINEAR_HEAD_PREFIX):]
        # An explicit request overrides the default. Naming a model and getting
        # nothing back is the --only trap in a new costume, and it is worse here
        # because the filter that would swallow it is silent.
        if only:
            if ssl not in only:
                continue
        elif keep is not None and ssl not in keep:
            continue
        out.append((ssl, ckpt))
    return out


def resolve_baseline_model(system):
    if system == "lfcc_gmm":
        return os.path.join(BASELINE_MODELS_ROOT, "lfcc_gmm")
    return os.path.join(BASELINE_MODELS_ROOT,
                        "model_weighted_CCE_50_64_aasist_raw_ASV19_none", "swa.pth")


def _frontends(job, system, models=None, paper_only=False):
    """(frontend, checkpoint) pairs for one system."""
    if system == "linear_head":
        return discover_linear_heads(only=models, paper_only=paper_only)
    if models and "none" not in models:
        return []          # this system has no upstream to select
    return [("none", resolve_baseline_model(system))]


# ===========================================================================
# The one enumerator
# ===========================================================================

def enumerate_tasks(job, systems=None, datasets=None, models=None,
                    paper_only=None):
    """Every (system, dataset, frontend) this job selects.

    The three filters are the three axes of the task space, so any slice of it
    can be named directly: one model on one dataset, every model on one
    dataset, one model everywhere.

    ``paper_only`` defaults to True: the 21 upstreams Table 5 reports, rather
    than all 24 heads on disk. Naming models explicitly overrides it.
    """
    if paper_only is None:
        paper_only = True
    chosen_systems = [s for s in job.systems if not systems or s in systems]
    chosen_datasets = ordered_datasets(datasets or job.datasets)
    tasks = []
    for system in chosen_systems:
        for dataset in chosen_datasets:
            for frontend, ckpt in _frontends(job, system, models, paper_only):
                out_file = score_path(system, dataset, frontend)
                argv = [*DRIVER, "--model", system, "--model_path", ckpt,
                        "--dataset", dataset, "--output_file", out_file,
                        "--cuda_device", "cuda:0",
                        "--batch_size", str(job.batch_size),
                        "--num_workers", str(job.num_workers),
                        "--n_jobs", str(job.gmm_processes)]
                if system == "linear_head":
                    argv += ["--ssl_model", frontend]

                # No reference file is resolved here. Scoring must not depend
                # on a previously produced score file, or a fresh tree can only
                # ever reproduce the old one's coverage. Comparison is a
                # separate step against an explicitly named tree.
                name = (f"{system}/{dataset}/{frontend}" if system == "linear_head"
                        else f"{system}/{dataset}")
                tasks.append(Task(name=name, argv=argv, out_file=out_file,
                                  dataset=dataset, system=system,
                                  frontend=frontend,
                                  needs_gpu=(system != "lfcc_gmm"),
                                  expect_lines=expected_rows(dataset),
                                  verify=verify_policy(dataset)))
    return tasks


# ===========================================================================
# The registry
# ===========================================================================

# Three jobs, one per set of back-ends. There used to be three more -- mlaad,
# mailabs, spoofceleb -- one per dataset. They were a dataset name plus three
# facts about that dataset (which policy grades it, its retry budget, which
# upstreams to skip), and once --datasets existed the name was redundant while
# the facts belonged to the dataset registry. They now live there, so
# `--job all --datasets Multilingual` behaves like `--job mlaad` did, which
# `--job all` never used to.
JOBS = {
    # Everything: every system on every dataset.
    "all": Job(name="all", systems=("linear_head", *BASELINE_SYSTEMS)),

    # Every SSL linear head on every dataset.
    "linear_head": Job(name="linear_head", systems=("linear_head",)),

    # The two non-SSL baselines on every dataset.
    "baselines": Job(name="baselines", systems=BASELINE_SYSTEMS, batch_size=64),
}
