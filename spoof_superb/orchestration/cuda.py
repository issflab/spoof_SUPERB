"""CUDA health and device pinning for the orchestrators.

Both behaviours here were learned the hard way on this host and are preserved
verbatim from the orchestrators they came from.
"""

import os
import subprocess
import sys
import time
from functools import lru_cache

PROBE = "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"

__all__ = ["gpu_uuids", "visible_device_env", "cuda_healthy", "wait_for_cuda"]


@lru_cache(maxsize=1)
def gpu_uuids():
    """GPU index -> UUID.

    CUDA_VISIBLE_DEVICES by *index* fails to initialise on this host once
    another process holds a different device ("CUDA driver initialization
    failed"), which previously sent whole models to CPU. Selecting by UUID is
    unaffected, so every launch pins its device by UUID.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).decode()
    except Exception as exc:
        print(f"[orchestrate] cannot read GPU UUIDs ({exc}); falling back to indices")
        return []
    return [u.strip() for u in out.splitlines() if u.strip()]


def visible_device_env(gpu):
    """Environment pinning one GPU, by UUID when available."""
    uuids = gpu_uuids()
    value = uuids[gpu] if gpu < len(uuids) else str(gpu)
    return dict(os.environ, CUDA_VISIBLE_DEVICES=value)


def cuda_healthy(gpu=None, python=sys.executable, timeout=180):
    """True if a FRESH process can initialise CUDA (on `gpu`, if given).

    Must be a subprocess: torch caches its failed init, so an in-process check
    after an outage is stale.
    """
    env = visible_device_env(gpu) if gpu is not None else dict(os.environ)
    try:
        return subprocess.call([python, "-c", PROBE], env=env,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=timeout) == 0
    except Exception:
        return False


def wait_for_cuda(tag, gpu=None, wait_s=3600, poll_s=60, python=sys.executable):
    """Block until CUDA is usable. False if it never comes back within wait_s."""
    if cuda_healthy(gpu, python=python):
        return True
    where = "" if gpu is None else f" on gpu {gpu}"
    print(f"[orchestrate] {tag}: CUDA down{where}; holding the task on the queue "
          f"until the driver returns", flush=True)
    waited = 0
    while waited < wait_s:
        time.sleep(poll_s)
        waited += poll_s
        if cuda_healthy(gpu, python=python):
            print(f"[orchestrate] {tag}: CUDA back{where} after {waited}s", flush=True)
            return True
        if waited % 900 == 0:
            print(f"[orchestrate] {tag}: still waiting for CUDA{where} ({waited}s)",
                  flush=True)
    return False
