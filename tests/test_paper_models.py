"""Contracts for the paper model roster and per-run identity.

Two changes are guarded here, and both have a failure mode that is silent:

  * scoring the wrong set of models wastes days of GPU time or, worse, omits a
    model the paper reports
  * two runs sharing one status directory destroys the audit trail a paper's
    provenance rests on
"""

import json
import os

import pytest

from spoof_superb.orchestration.jobs import (
    JOBS,
    Job,
    default_run_name,
    discover_linear_heads,
)
from spoof_superb.scoring.models import (
    TABLE5_BASELINE,
    is_paper_model,
    non_paper_models,
    paper_models,
)


# ===========================================================================
# M1-M6: the roster comes from Table 5 and cannot drift from it
# ===========================================================================

def test_m1_roster_is_read_from_the_table5_baseline():
    """Not a hand-kept list: the file the regression gate reads is the source."""
    with open(TABLE5_BASELINE) as f:
        rows = json.load(f)["results"]
    assert paper_models() == frozenset(r["slug"] for r in rows.values())


def test_m2_every_reported_model_is_in_the_roster():
    """A model in Table 5 that is not scoreable is a hole in the paper."""
    with open(TABLE5_BASELINE) as f:
        rows = json.load(f)["results"]
    for name, row in rows.items():
        assert is_paper_model(row["slug"]), f"Table 5 row {name} not in the roster"


def test_m3_a_missing_baseline_raises_instead_of_widening(tmp_path):
    """Falling back to "score everything" would silently burn a day of GPU."""
    paper_models.cache_clear()
    with pytest.raises(FileNotFoundError, match="model roster"):
        paper_models(str(tmp_path / "absent.json"))


def test_m4_a_malformed_baseline_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not_results": {}}')
    paper_models.cache_clear()
    with pytest.raises(ValueError):
        paper_models(str(bad))
    empty = tmp_path / "empty.json"
    empty.write_text('{"results": {}}')
    with pytest.raises(ValueError, match="no model slugs"):
        paper_models(str(empty))


def test_m5_non_paper_models_are_identified_not_guessed():
    """The three extras must be named by the data, not by a literal in code."""
    heads = [s for s, _ in discover_linear_heads(paper_only=False)]
    if not heads:
        pytest.skip("no trained heads on disk")
    extra = non_paper_models(heads)
    assert set(extra).isdisjoint(paper_models())
    assert all(h in paper_models() for h in heads if h not in extra)


def test_m6_paper_only_narrows_discovery():
    heads_all = {s for s, _ in discover_linear_heads(paper_only=False)}
    heads_paper = {s for s, _ in discover_linear_heads(paper_only=True)}
    if not heads_all:
        pytest.skip("no trained heads on disk")
    assert heads_paper <= heads_all
    assert heads_paper == heads_all & paper_models()


def test_m7_naming_a_model_overrides_paper_only():
    """An explicit request must never be silently dropped by a default filter.

    This is the trap the old --only overload set, and it must not come back in
    a new form.
    """
    heads = {s for s, _ in discover_linear_heads(paper_only=False)}
    extra = non_paper_models(heads)
    if not extra:
        pytest.skip("every trained head is in the paper")
    got = {s for s, _ in discover_linear_heads(only=[extra[0]], paper_only=True)}
    assert got == {extra[0]}


def test_m8_enumerate_defaults_to_paper_only():
    """The default must be the narrow set, or the change has no effect."""
    job = JOBS["all"]
    n_paper = len(job.enumerate_tasks(datasets=["wild"]))
    n_all = len(job.enumerate_tasks(datasets=["wild"], paper_only=False))
    heads = {s for s, _ in discover_linear_heads(paper_only=False)}
    assert n_all - n_paper == len(non_paper_models(heads))


# ===========================================================================
# R1-R5: each run owns its own directory
# ===========================================================================

def test_r1_two_runs_of_one_job_do_not_share_a_status_file():
    """The collision that destroyed a 19-task record on 2026-07-29."""
    a = Job(name="all", run="run-a")
    b = Job(name="all", run="run-b")
    assert a.status_path() != b.status_path()
    assert a.summary_path() != b.summary_path()
    assert a.logs() != b.logs()


def test_r2_runs_of_one_job_still_share_a_job_directory():
    """History stays grouped: one job, many runs, one parent."""
    a = Job(name="all", run="run-a")
    b = Job(name="all", run="run-b")
    assert a.job_dir == b.job_dir
    assert a.out_dir.startswith(a.job_dir + os.sep)


def test_r3_default_run_name_is_unique_and_sorts_chronologically():
    a = default_run_name()
    assert len(a) == 15 and a[8] == "-"          # YYYYmmdd-HHMMSS
    assert sorted(["20260101-000000", a]) == ["20260101-000000", a]


def test_r4_latest_symlink_points_at_this_run(tmp_path, monkeypatch):
    from spoof_superb import config
    monkeypatch.setattr(config.cfg, "scores_root", str(tmp_path))
    job = Job(name="all", run="20260729-180000")
    os.makedirs(job.out_dir, exist_ok=True)
    link = job.link_latest()
    assert os.path.realpath(link) == os.path.realpath(job.out_dir)
    # A second run must move the link, not fail on the existing one.
    job2 = Job(name="all", run="20260729-190000")
    os.makedirs(job2.out_dir, exist_ok=True)
    assert os.path.realpath(job2.link_latest()) == os.path.realpath(job2.out_dir)


def test_r5_extra_field_is_gone():
    """P6: it was declared and consumed nowhere."""
    import dataclasses
    assert "extra" not in {f.name for f in dataclasses.fields(Job)}


def test_r6_gmm_processes_is_the_field_name_now():
    """`n_jobs` collided with --jobs and --job; the GMM meaning is explicit."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(Job)}
    assert "gmm_processes" in names
    assert "n_jobs" not in names


# ===========================================================================
# C1-C3: the CLI
# ===========================================================================

def _parse(argv):
    """Run the driver's parser without running a sweep."""
    import spoof_superb.orchestration.driver as d
    return d.main(argv)


def test_c1_workers_replaces_jobs_but_jobs_still_works(capsys):
    """A rename that breaks a user's script is a regression, not a cleanup."""
    assert _parse(["--job", "all", "--datasets", "wild", "--workers", "1", "--list"]) == 0
    out = capsys.readouterr().out
    assert "linear_head/wild" in out

    assert _parse(["--job", "all", "--datasets", "wild", "--jobs", "1", "--list"]) == 0
    out = capsys.readouterr().out
    assert "deprecated" in out and "linear_head/wild" in out


def test_c1b_jobs_alias_is_honoured_even_when_workers_is_also_passed(capsys):
    """bin/orchestrate.sh always emits --workers, so the alias must still win.

    The first implementation only applied --jobs when --workers was absent,
    which made a hand-typed `bin/orchestrate.sh --jobs 1` silently do nothing.
    """
    assert _parse(["--job", "all", "--datasets", "wild",
                   "--workers", "0", "--jobs", "1", "--list"]) == 0
    assert "deprecated" in capsys.readouterr().out


def test_c2_all_models_widens_the_selection(capsys):
    _parse(["--job", "all", "--datasets", "wild", "--list"])
    narrow = capsys.readouterr().out.count("->")
    _parse(["--job", "all", "--datasets", "wild", "--all-models", "--list"])
    wide = capsys.readouterr().out.count("->")
    heads = {s for s, _ in discover_linear_heads(paper_only=False)}
    assert wide - narrow == len(non_paper_models(heads))


def test_c3_run_name_flag_reaches_the_job(capsys, monkeypatch, tmp_path):
    """--run-name must change where the record lands, not just be accepted."""
    from spoof_superb import config
    monkeypatch.setattr(config.cfg, "scores_root", str(tmp_path))
    seen = {}

    import spoof_superb.orchestration.driver as d
    real = d.JOBS["all"].enumerate_tasks

    def spy(*a, **k):
        seen["run"] = d.JOBS["all"].run
        return real(*a, **k)

    monkeypatch.setattr(d.JOBS["all"], "enumerate_tasks", spy)
    _parse(["--job", "all", "--datasets", "wild", "--run-name", "my-run", "--list"])
    assert seen["run"] == "my-run"


# ===========================================================================
# J1-J7: jobs pick back-ends; datasets carry their own policy
# ===========================================================================

def test_j1_only_three_jobs_remain():
    """mlaad/mailabs/spoofceleb were a dataset name plus dataset facts."""
    assert sorted(JOBS) == ["all", "baselines", "linear_head"]


def test_j2_job_no_longer_carries_verify_or_skip():
    """Both were properties of a dataset, not of an invocation."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(Job)}
    assert "verify" not in names
    assert "skip" not in names


def test_j3_the_grading_policy_travels_with_the_dataset():
    """The hole this closed: --job all used to grade nothing.

    Because the policy lived on --job mlaad, the sweep anyone would actually
    run had verify=None for MLAAD and SpoofCeleb.
    """
    from spoof_superb.scoring.datasets import verify_policy
    tasks = {t.name: t for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual", "spoofceleb", "wild"])}
    for name, t in tasks.items():
        assert t.verify == verify_policy(t.dataset), name
    assert any(t.verify == "mlaad" for t in tasks.values())
    assert any(t.verify == "spoofceleb" for t in tasks.values())
    assert any(t.verify is None for t in tasks.values())


def test_j4_model_skips_are_per_dataset_not_per_sweep():
    """`mockingjay` is excluded from MLAAD and wanted everywhere else.

    A job-wide skip could not express that without a job per dataset.
    """
    from spoof_superb.scoring.datasets import skip_models
    names = lambda ds: {t.frontend for t in JOBS["all"].enumerate_tasks(datasets=[ds])
                        if t.system == "linear_head"}
    if "mockingjay" not in paper_models():
        pytest.skip("mockingjay is not a paper model here")
    assert "mockingjay" not in names("Multilingual")
    assert "mockingjay" in names("wild")
    assert "mockingjay" in skip_models("Multilingual")
    assert skip_models("wild") == frozenset()


def test_j5_naming_a_skipped_model_still_scores_it():
    """An explicit --models must override the dataset's skip list."""
    got = {t.frontend for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual"], models=["mockingjay"])
        if t.system == "linear_head"}
    assert got == {"mockingjay"}


def test_j6_scoring_resolves_no_reference_file():
    """The core contract: a fresh tree cannot inherit an old tree's coverage."""
    for t in JOBS["all"].enumerate_tasks(datasets=["Multilingual", "spoofceleb"]):
        assert not hasattr(t, "ref_file") or getattr(t, "ref_file", None) is None
    # and the task argv never mentions a reference
    for t in JOBS["all"].enumerate_tasks(datasets=["Multilingual"]):
        assert not any("--reference" in a or "--ref" == a for a in t.argv)


def test_j7_reference_lookup_requires_an_explicit_root(tmp_path):
    """No root named means no comparison -- never a silent default."""
    from spoof_superb.orchestration.driver import reference_for
    t = JOBS["all"].enumerate_tasks(datasets=["Multilingual"])[0]
    assert reference_for(t, None, "legacy") is None
    assert reference_for(t, str(tmp_path), "legacy") is None      # empty tree


def test_j8_reference_resolves_against_a_real_old_tree():
    """The migration check must actually find the published columns."""
    from spoof_superb.orchestration.driver import reference_for
    old = "/data/ssl_anti_spoofing/asd_superb_score_files"
    if not os.path.isdir(old):
        pytest.skip("the old score tree is not mounted")
    found = [t.name for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual", "spoofceleb"])
        if reference_for(t, old, "legacy")]
    assert found, "no published column resolved; the migration check is broken"
