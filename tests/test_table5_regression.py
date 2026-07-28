"""
test_table5_regression.py
-------------------------
Contract: the reorganisation must not move a single published number.

tests/baseline_table5.json is the output of

    python -m spoof_superb.analysis.recompute_table5_mlaad_v10 --out_dir <tmp>

captured on the pre-reorg tree at commit 018115d, with that script's own
"REPRODUCTION GATE (untouched columns)" passing. Every per-model, per-dataset
EER in the paper's Tables 5 and 6 is in there.

This is the gate for the package migration: file moves, import rewrites, and
the eval/orchestrator merges are all behaviour-preserving refactors, so a fresh
run must reproduce this payload exactly.

Opt-in, because a run reads ~15 GB of score files off /data and takes minutes:

    RUN_TABLE5=1 pytest tests/test_table5_regression.py

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
BASELINE = Path(__file__).resolve().parent / "baseline_table5.json"
RECOMPUTE = REPO_ROOT / "spoof_superb" / "analysis" / "recompute_table5_mlaad_v10.py"

TOL = 0.0


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_TABLE5") != "1",
    reason="slow (~minutes, reads ~15 GB); set RUN_TABLE5=1 to run",
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
        proc = subprocess.run(
            [sys.executable, "-m", "spoof_superb.analysis.recompute_table5_mlaad_v10",
             "--out_dir", td],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
        )
        assert proc.returncode == 0, (
            f"reproducer failed rc={proc.returncode}\n"
            f"stdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}"
        )
        out = Path(td) / "table5_mlaad_v10.json"
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
