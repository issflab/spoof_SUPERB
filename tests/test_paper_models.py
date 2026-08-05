"""Contracts for the paper model roster and per-run identity.

Two changes are guarded here, and both have a failure mode that is silent:

  * scoring the wrong set of models wastes days of GPU time or, worse, omits a
    model the paper reports
  * two runs sharing one status directory destroys the audit trail a paper's
    provenance rests on
"""

import inspect
import json
import os
import re

import pytest

from spoof_superb.orchestration.jobs import (
    JOBS,
    Job,
    default_run_name,
    discover_linear_heads,
)
from spoof_superb.scoring.models import (
    PAPER_TABLE_ROWS,
    PAPER_ROSTER,
    _slug_by_display,
    is_paper_model,
    non_paper_models,
    paper_models,
)


# ===========================================================================
# M1-M6: the roster comes from the paper and cannot drift from it
# ===========================================================================

PAPER_TEX = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "spoof_SUPERB_IEEE_ACCESS",
    "access.tex")


def _main_results_rows():
    """The SSL row labels actually printed by the paper's main results table.

    Parsed from access.tex, which is the authority for what the paper reports.
    Skips when the paper repo is not checked out beside this one.
    """
    if not os.path.isfile(PAPER_TEX):
        pytest.skip("the paper repo is not checked out beside this one")
    tex = open(PAPER_TEX).read()
    try:
        body = tex[tex.index("\\label{tab:results_main}"):]
        body = body[:body.index("\\end{table*}")]
    except ValueError:
        pytest.skip("tab:results_main not found in access.tex")
    rows = []
    for line in body.splitlines():
        m = re.match(r"\s*([A-Za-z][A-Za-z0-9 .\-+]*?)\s*&", line)
        if m and not line.lstrip().startswith("\\textbf{SSL"):
            rows.append(m.group(1).strip())
    # LFCC-GMM and AASIST are complete detectors, not upstreams.
    return [r for r in rows if r not in ("LFCC-GMM", "AASIST")]


def test_m1_roster_matches_the_paper_table_exactly():
    """The reconciliation this module exists for.

    The first implementation derived the roster from the regression baseline
    and claimed it "cannot drift from the paper". It had already drifted: the
    roster file carries FBANK and Mockingjay, which the table does not print.
    This test is what makes the explicit list trustworthy.
    """
    printed = _main_results_rows()
    assert list(PAPER_TABLE_ROWS) == printed, (
        f"PAPER_TABLE_ROWS disagrees with the paper's results table.\n"
        f"  only in code : {[r for r in PAPER_TABLE_ROWS if r not in printed]}\n"
        f"  only in paper: {[r for r in printed if r not in PAPER_TABLE_ROWS]}")


def test_m2_the_roster_file_is_a_superset_of_the_paper():
    """paper_roster.json maps more models than the table prints.

    That is fine -- it is a name-to-slug dictionary, not the roster -- but it
    means the file cannot BE the roster, which is the mistake this replaced.
    Nothing in it distinguishes FBANK and Mockingjay from the 19 printed rows.
    """
    with open(PAPER_ROSTER) as f:
        tracked = set(json.load(f)["roster"].values())
    assert paper_models() < tracked, "the roster file should map at least the paper's models"
    extra = sorted(tracked - paper_models())
    assert extra, "expected the roster file to map unprinted models"


def test_m3_every_printed_row_maps_to_a_slug():
    """A printed row with no slug means the two sources have diverged."""
    for name in PAPER_TABLE_ROWS:
        assert name in _slug_by_display(), f"{name} has no slug in the roster file"
    assert len(paper_models()) == len(PAPER_TABLE_ROWS)


def test_m4_a_missing_roster_file_raises_instead_of_widening(tmp_path):
    """Falling back to "score everything" would silently burn a day of GPU."""
    paper_models.cache_clear()
    _slug_by_display.cache_clear()
    with pytest.raises(FileNotFoundError, match="model roster"):
        paper_models(str(tmp_path / "absent.json"))
    paper_models.cache_clear()
    _slug_by_display.cache_clear()


def test_m4b_a_row_the_roster_file_does_not_know_raises(tmp_path):
    """Renaming a table row must fail loudly, not silently shrink the roster."""
    part = tmp_path / "partial.json"
    part.write_text(json.dumps({"roster": {"APC": "apc"}}))
    paper_models.cache_clear()
    _slug_by_display.cache_clear()
    with pytest.raises(ValueError, match="diverged"):
        paper_models(str(part))
    paper_models.cache_clear()
    _slug_by_display.cache_clear()


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


def test_j3_no_task_carries_a_grading_policy():
    """Scoring grades nothing, so a Task has nothing to grade with.

    A per-dataset policy on the Task was the hook a sweep-time comparison hung
    off. Removing the hook is what makes "scoring never reads a score file it
    did not just write" a structural property rather than a default that a flag
    can switch off.
    """
    import dataclasses
    import spoof_superb.scoring.datasets as D
    from spoof_superb.orchestration.jobs import Task

    assert "verify" not in {f.name for f in dataclasses.fields(Task)}
    assert not hasattr(D, "verify_policy")
    assert not hasattr(D, "VERIFY_POLICY")


def test_j4_unreported_models_are_excluded_by_the_roster_alone():
    """SKIP_MODELS is gone; paper_only is the only exclusion mechanism left.

    Both its entries -- mockingjay and byol_a_2048 -- were already outside the
    19-model roster, so it excluded nothing the roster did not. Removing it means
    --all-models now really does mean every trained head, on every dataset.
    """
    import spoof_superb.scoring.datasets as D
    assert not hasattr(D, "SKIP_MODELS")
    assert not hasattr(D, "skip_models")
    default = {t.frontend for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual"]) if t.system == "linear_head"}
    widened = {t.frontend for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual"], paper_only=False) if t.system == "linear_head"}
    assert default == paper_models() & widened
    assert widened == {s for s, _ in discover_linear_heads(paper_only=False)}


def test_j4b_discover_no_longer_takes_a_skip_argument():
    """The parameter went with the data: nothing else ever passed one."""
    import inspect
    assert "skip" not in inspect.signature(discover_linear_heads).parameters


def test_j5_naming_an_unreported_model_still_scores_it():
    """An explicit --models must override the paper_only default."""
    got = {t.frontend for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual"], models=["mockingjay"])
        if t.system == "linear_head"}
    assert got == {"mockingjay"}
    assert not is_paper_model("mockingjay")


def test_j6_scoring_resolves_no_reference_file():
    """The core contract: a fresh tree cannot inherit an old tree's coverage."""
    for t in JOBS["all"].enumerate_tasks(datasets=["Multilingual", "spoofceleb"]):
        assert not hasattr(t, "ref_file") or getattr(t, "ref_file", None) is None
    # and the task argv never mentions a reference
    for t in JOBS["all"].enumerate_tasks(datasets=["Multilingual"]):
        assert not any("--reference" in a or "--ref" == a for a in t.argv)


def test_j7_the_orchestrator_cannot_be_pointed_at_a_second_tree():
    """There is no --verify-against, and no code path that would honour one.

    Not merely off by default: absent. A comparison inside the sweep grades a
    tree against whatever is on disk at the time and makes the sweep's exit
    code depend on a second tree, so the flag and its machinery are gone rather
    than defaulted to empty.
    """
    import spoof_superb.orchestration.driver as drv

    assert not hasattr(drv, "reference_for")
    assert not hasattr(drv, "_verify")

    # The module docstring names the flag in order to explain its absence, so
    # the check is on the code below it, not on the prose.
    src = inspect.getsource(drv).replace(drv.__doc__ or "", "")
    for banned in ("--verify-against", "--verify-layout", "verification.driver"):
        assert banned not in src, f"{banned} is back in the orchestrator"


def test_j8_verification_is_reachable_as_its_own_step():
    """Removing the hook must not remove the capability.

    The check is that both levels exist and are runnable independently -- the
    replacement for the sweep-time comparison, not merely its deletion.
    """
    from spoof_superb.verification import analysis, scores
    from spoof_superb.verification.verdicts import ANALYSIS_LADDER, SCORE_LADDER

    assert callable(scores.verify_scores)
    assert callable(analysis.verify_analysis)
    # The two levels answer different questions and must not share a ladder.
    assert SCORE_LADDER != ANALYSIS_LADDER


def test_j9_only_the_960hr_variant_is_scored_on_mlaad():
    """mockingjay_960hr must be scored on MLAAD; plain mockingjay must not.

    It is the variant the paper's MLAAD table lists, and the only mockingjay*
    file in the published MLAAD tree. The two names share a prefix, so a
    substring match anywhere in the roster logic would take out the wrong one.
    """
    frontends = {t.frontend for t in JOBS["all"].enumerate_tasks(
        datasets=["Multilingual"]) if t.system == "linear_head"}
    heads = {s for s, _ in discover_linear_heads(paper_only=False)}
    if "mockingjay_960hr" not in heads:
        pytest.skip("mockingjay_960hr is not trained here")
    assert "mockingjay_960hr" in frontends
    assert "mockingjay" not in frontends


def test_j10_only_the_960hr_mockingjay_is_a_paper_model():
    """The paper's results table prints Mockingjay-960h and no plain Mockingjay.

    The regression baseline carries both, which is why the roster is not derived
    from it.
    """
    assert is_paper_model("mockingjay_960hr")
    assert not is_paper_model("mockingjay")
    assert not is_paper_model("fbank")


# ===========================================================================
# D1-D4: the default sweep uses the segmented Deepfake-Eval
# ===========================================================================

def test_d1_segmented_dfeval_is_in_the_default_sweep_and_unsegmented_is_not():
    """The two variants measure the same corpus two ways; only one belongs.

    Scoring both by default would put two DFEval columns in the tree with no
    statement of which one the benchmark means.
    """
    from spoof_superb.scoring.datasets import DEFAULT_DATASETS, SCOREABLE
    assert "deepfake_eval_2024_segmented" in DEFAULT_DATASETS
    assert "deepfake_eval_2024" not in DEFAULT_DATASETS
    assert "deepfake_eval_2024" in SCOREABLE       # still scoreable by name
    assert set(DEFAULT_DATASETS) < set(SCOREABLE)


def test_d2_a_default_sweep_scores_the_segmented_set_only():
    datasets = {t.dataset for t in JOBS["all"].enumerate_tasks()}
    assert "deepfake_eval_2024_segmented" in datasets
    assert "deepfake_eval_2024" not in datasets


def test_d3_naming_the_unsegmented_set_still_scores_it():
    """Excluded from the default is not removed: the published column is
    reproducible on request."""
    tasks = JOBS["all"].enumerate_tasks(datasets=["deepfake_eval_2024"])
    assert tasks
    assert {t.dataset for t in tasks} == {"deepfake_eval_2024"}


def test_d4_the_two_variants_write_to_different_paths():
    """Both can sit on disk at once without either overwriting the other."""
    from spoof_superb.core.scorepath import score_path
    a = score_path("linear_head", "deepfake_eval_2024", "xls_r_300m")
    b = score_path("linear_head", "deepfake_eval_2024_segmented", "xls_r_300m")
    assert a != b


def test_d5_the_dfeval_column_resolves_to_the_segmented_measurement():
    """The DFEval24 column must read the SEGMENTED trial set under v2/v3.

    Two different measurements share one column name:

        deepfake_eval_2024              1,980 trials -- one 4 s window per file
        deepfake_eval_2024_segmented   56,481 trials -- every 4 s window

    A 4 s model never saw past the first four seconds of a minutes-long
    recording, so the unsegmented column left most of the corpus unexamined.
    Per-segment trials weight long recordings far more heavily, so the two EERs
    are different quantities -- not a corrected value.

    This asserts the RESOLUTION, deliberately, rather than a trial count printed
    in the paper. An earlier version compared against the literal 1,976 as "the
    paper's number"; the paper is being updated to report the segmented column,
    which would have made that test assert a premise that had stopped being
    true. What must hold either way is that the mapping does not silently flip
    -- doing so would change every DFEval number in the table while the code
    went on reporting the same column name.

    The unsegmented key still exists in the dataset registry, so it remains
    scoreable by name; it is simply not what the benchmark column reads.
    """
    from spoof_superb.core.scorepath import column_key
    from spoof_superb.orchestration.jobs import expected_rows

    assert column_key("deepfake_eval_2024") == "deepfake_eval_2024_segmented"
    # every other column reads its own name
    assert column_key("wild") == "wild"

    n_seg = expected_rows("deepfake_eval_2024_segmented")
    n_whole = expected_rows("deepfake_eval_2024")
    assert n_seg > 10 * n_whole, (
        f"the two DFEval sets are meant to be different measurements, but "
        f"segmented={n_seg} is not an order of magnitude over "
        f"unsegmented={n_whole}")


def test_d5b_the_column_mapping_has_exactly_one_definition():
    """The producer and the verifier must resolve through the same object.

    `recompute_main_results` builds the paper's table; `verification.cells`
    checks it. Both used to carry a private copy of this mapping. Had those
    drifted, the checker would have compared a segmented column against an
    unsegmented one and reported EQUIVALENT -- blind to precisely the error it
    exists to catch.
    """
    from spoof_superb.core import scorepath
    from spoof_superb.verification import cells

    assert cells.COLUMN_KEYS is scorepath.COLUMN_KEYS
    assert cells.column_key is scorepath.column_key

    import spoof_superb.analysis.recompute_main_results as rmr
    assert not hasattr(rmr, "COLUMN_KEYS"), (
        "recompute_main_results has its own copy again")
    assert not hasattr(rmr, "dataset_key"), (
        "the layout-aware wrapper should be gone with the layouts")
