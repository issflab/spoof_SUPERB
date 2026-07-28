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
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

from spoof_superb.config import cfg
from spoof_superb.core.scorepath import score_path
from spoof_superb.scoring.datasets import (
    PROTOCOL_SPECS,
    SCOREABLE,
    has_reference,
    reference_paths,
)

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
    """Requested datasets, smallest-first, unknown names dropped."""
    wanted = list(datasets) if datasets else list(SCOREABLE)
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
    ref_file: Optional[str] = None
    needs_gpu: bool = True
    expect_lines: Optional[int] = None


@dataclass
class Job:
    name: str
    systems: Sequence[str] = ("linear_head",)
    datasets: Optional[Sequence[str]] = None      # None = every scoreable one
    verify: Optional[str] = None
    skip: frozenset = frozenset()
    gpus: tuple = (0, 1, 2)
    max_attempts: int = 3
    cuda_wait_s: int = 3600
    batch_size: int = 0                           # 0 = the back-end default
    num_workers: int = 6
    n_jobs: int = 16
    log_dir: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def out_dir(self):
        """Where this job's status and summary live -- not the score files."""
        return os.path.join(cfg.scores_root, "_runs", self.name)

    def logs(self):
        return self.log_dir or os.path.join(self.out_dir, "logs")

    def status_path(self):
        return os.path.join(self.out_dir, "run_status.json")

    def summary_path(self):
        return os.path.join(self.out_dir, "SUMMARY.txt")

    def enumerate_tasks(self, systems=None, datasets=None, models=None):
        return enumerate_tasks(self, systems=systems, datasets=datasets,
                               models=models)


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


def resolve_baseline_model(system):
    if system == "lfcc_gmm":
        return os.path.join(BASELINE_MODELS_ROOT, "lfcc_gmm")
    return os.path.join(BASELINE_MODELS_ROOT,
                        "model_weighted_CCE_50_64_aasist_raw_ASV19_none", "swa.pth")


def _frontends(job, system, models=None):
    """(frontend, checkpoint) pairs for one system."""
    if system == "linear_head":
        return discover_linear_heads(only=models, skip=job.skip)
    if models and "none" not in models:
        return []          # this system has no upstream to select
    return [("none", resolve_baseline_model(system))]


# ===========================================================================
# The one enumerator
# ===========================================================================

def enumerate_tasks(job, systems=None, datasets=None, models=None):
    """Every (system, dataset, frontend) this job selects.

    The three filters are the three axes of the task space, so any slice of it
    can be named directly: one model on one dataset, every model on one
    dataset, one model everywhere.
    """
    chosen_systems = [s for s in job.systems if not systems or s in systems]
    chosen_datasets = ordered_datasets(datasets or job.datasets)
    tasks = []
    for system in chosen_systems:
        for frontend, ckpt in _frontends(job, system, models):
            for dataset in chosen_datasets:
                out_file = score_path(system, dataset, frontend)
                argv = [*DRIVER, "--model", system, "--model_path", ckpt,
                        "--dataset", dataset, "--output_file", out_file,
                        "--cuda_device", "cuda:0",
                        "--batch_size", str(job.batch_size),
                        "--num_workers", str(job.num_workers),
                        "--n_jobs", str(job.n_jobs)]
                if system == "linear_head":
                    argv += ["--ssl_model", frontend]

                # Only meaningful where a published score file exists to
                # compare against; protocol-scored columns in a fresh tree
                # have none.
                ref = None
                if job.verify and system == "linear_head" and has_reference(dataset):
                    try:
                        ref = reference_paths(dataset, frontend)[0]
                    except KeyError:
                        ref = None

                name = (f"{system}/{dataset}/{frontend}" if system == "linear_head"
                        else f"{system}/{dataset}")
                tasks.append(Task(name=name, argv=argv, out_file=out_file,
                                  dataset=dataset, ref_file=ref,
                                  needs_gpu=(system != "lfcc_gmm"),
                                  expect_lines=expected_rows(dataset)))
    return tasks


# ===========================================================================
# The registry
# ===========================================================================

# byol_a_2048 and mockingjay were excluded from the MLAAD/M-AILABS runs by an
# earlier request. Kept on those named jobs rather than made a global default,
# so the exclusion is visible where it applies instead of silently everywhere.
_MLAAD_SKIP = frozenset({"byol_a_2048", "mockingjay"})

JOBS = {
    # Everything: every system on every dataset.
    "all": Job(name="all", systems=("linear_head", *BASELINE_SYSTEMS)),

    # Every SSL linear head on every dataset. This is the sweep that had no
    # job before -- it needed one script invocation per dataset by hand.
    "linear_head": Job(name="linear_head", systems=("linear_head",)),

    # The two non-SSL baselines on every dataset.
    "baselines": Job(name="baselines", systems=BASELINE_SYSTEMS, batch_size=64),

    # Dataset-restricted sweeps, kept because they are how the published runs
    # were organised and because their skip lists and retry budgets differ.
    "mlaad": Job(name="mlaad", datasets=["Multilingual"], verify="mlaad",
                 skip=_MLAAD_SKIP, max_attempts=1),
    "mailabs": Job(name="mailabs", datasets=["MAILABS"], verify="mlaad",
                   skip=_MLAAD_SKIP, max_attempts=1),
    "spoofceleb": Job(name="spoofceleb", datasets=["spoofceleb"],
                      verify="spoofceleb", max_attempts=6, cuda_wait_s=3 * 3600),
}
