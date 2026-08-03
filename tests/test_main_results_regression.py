"""
test_main_results_regression.py
-------------------------
Contract: the reorganisation must not move a single published number.

tests/baseline_main_results_table.json is the output of

    python -m spoof_superb.analysis.recompute_main_results --out_dir <tmp>

captured on the pre-reorg tree at commit 018115d, with that script's own
"REPRODUCTION GATE (untouched columns)" passing. Every per-model, per-dataset
EER in the paper's two results tables is in there.

This is the gate for the package migration: file moves, import rewrites, and
the eval/orchestrator merges are all behaviour-preserving refactors, so a fresh
run must reproduce this payload exactly.

Opt-in, because a run reads ~15 GB of score files off /data and takes minutes:

    RUN_MAIN_RESULTS=1 pytest tests/test_main_results_regression.py

TOL is 0.0 by design. These are deterministic recomputations over fixed score
files -- not a re-inference -- so any drift at all means something moved that
should not have.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "baseline_main_results_table.json"
RECOMPUTE = REPO_ROOT / "spoof_superb" / "analysis" / "recompute_main_results.py"

TOL = 0.0

#: The tree the baseline was measured on. The baseline records the PUBLISHED
#: numbers, so the gate must read the tree those numbers came from, whatever
#: tree the working config currently points at.
LEGACY_TREE = "/data/ssl_anti_spoofing/asd_superb_score_files"


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MAIN_RESULTS") != "1",
    reason="slow (~minutes, reads ~15 GB); set RUN_MAIN_RESULTS=1 to run",
)


@pytest.fixture(scope="module")
def fresh_payload():
    if not RECOMPUTE.is_file():
        pytest.skip(f"reproducer not found: {RECOMPUTE}")
    with tempfile.TemporaryDirectory() as td:
        # Run as a module, not by file path: the analysis scripts are package
        # modules now and no longer inject the repo root into sys.path
        # themselves. PYTHONPATH makes that work from any cwd.
        env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}")
        # Pin BOTH the tree and its layout. Pinning only the root left the gate
        # hermetic by accident: legacy paths were the reproducer's only
        # behaviour, so the configured score_layout could not affect it. Once
        # the reproducer became layout-aware and the shipped config moved to v3,
        # an unpinned gate read the legacy tree through v3 paths and reported
        # every column missing -- a failure about configuration, not numbers.
        proc = subprocess.run(
            [sys.executable, "-m", "spoof_superb.analysis.recompute_main_results",
             "--out_dir", td,
             "--scores_root", LEGACY_TREE, "--layout", "legacy"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
        )
        assert proc.returncode == 0, (
            f"reproducer failed rc={proc.returncode}\n"
            f"stdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        )
        out = Path(td) / "main_results.json"
        assert out.is_file(), f"reproducer wrote no {out.name}"
        return json.loads(out.read_text())


@pytest.fixture(scope="module")
def baseline():
    assert BASELINE.is_file(), f"missing baseline: {BASELINE}"
    return json.loads(BASELINE.read_text())


def test_model_set_is_unchanged(fresh_payload, baseline):
    assert set(fresh_payload["results"]) == set(baseline["results"]), (
        "the set of models in the table changed"
    )


_ABSENT = object()


def test_every_published_eer_is_unchanged(fresh_payload, baseline):
    """A null cell is data, not a gap.

    Some cells are legitimately null -- e.g. Mockingjay/MLAAD, whose v10 tsv
    does not exist -- so "key present with value None" and "key absent" are
    different states and only the second is a regression.
    """
    drifted = []
    for model, want in baseline["results"].items():
        got = fresh_payload["results"].get(model, _ABSENT)
        if got is _ABSENT:
            drifted.append(f"{model}: missing from fresh run")
            continue
        for dataset, want_cell in want.get("datasets", {}).items():
            got_cell = got.get("datasets", {}).get(dataset, _ABSENT)
            if got_cell is _ABSENT:
                drifted.append(f"{model}/{dataset}: missing from fresh run")
                continue
            if (want_cell is None) != (got_cell is None):
                drifted.append(f"{model}/{dataset}: {want_cell!r} -> {got_cell!r}")
                continue
            if want_cell is None:
                continue
            for field in ("eer", "n", "nan_frac"):
                a = want_cell.get(field, _ABSENT)
                b = got_cell.get(field, _ABSENT)
                if a is _ABSENT and b is _ABSENT:
                    continue
                if a is _ABSENT or b is _ABSENT or a is None or b is None:
                    if a is not b:
                        drifted.append(f"{model}/{dataset}.{field}: {a!r} -> {b!r}")
                    continue
                if abs(a - b) > TOL:
                    drifted.append(f"{model}/{dataset}.{field}: {a} -> {b}")

    assert not drifted, (
        f"{len(drifted)} published value(s) changed:\n  " + "\n  ".join(drifted[:40])
    )


def test_reproduction_gate_still_passes(fresh_payload):
    """The script's own check that untouched published cells still reproduce."""
    failures = fresh_payload.get("reproduction_failures") or []
    assert not failures, f"reproducer's internal gate failed:\n  " + "\n  ".join(failures)


def test_known_problem_set_does_not_grow(fresh_payload, baseline):
    """Pre-existing problems (fp16 NaN, EER>50) are allowed -- new ones are not."""
    new = set(fresh_payload.get("problems") or []) - set(baseline.get("problems") or [])
    assert not new, f"new problems appeared:\n  " + "\n  ".join(sorted(new))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
