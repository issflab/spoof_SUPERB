"""Contracts for the orchestrator's live progress display.

The display's only job is to be true. Each test names the way it could lie:
reading a stale bar, inventing a counter out of a path, double-counting a
finished task, or writing escape codes into a redirected log.
"""

import io
import os

import pytest

from spoof_superb.orchestration.progress import (
    NullReporter,
    ProgressReporter,
    fmt_hms,
    make_reporter,
    parse_progress,
    tail_progress,
)

# A real tqdm frame, carriage-return separated exactly as it lands in a log.
TQDM = ("xls_r_300m:   0%|          | 0/994 [00:00<?, ?it/s]\r"
        "xls_r_300m:  23%|##3       | 231/994 [00:41<02:16,  5.59it/s]\r"
        "xls_r_300m:  45%|####5     | 450/994 [01:20<01:37,  5.58it/s]")


# ===========================================================================
# P1-P5: the counter is parsed from tqdm, and only from tqdm
# ===========================================================================

def test_p1_parses_latest_tqdm_update():
    """The bar must show the newest frame, not the first one in the tail."""
    assert parse_progress(TQDM) == (450, 994)


def test_p2_ignores_the_rate_suffix():
    """`5.59it/s` contains a slash. It is not a progress counter."""
    assert parse_progress("xls_r_300m: 100%|#| 994/994 [02:58<00:00, 5.59it/s]") == (994, 994)


def test_p3_second_bar_supersedes_a_finished_first():
    """Two sequential loops in one log: the live one is the second."""
    text = ("stage1: 100%|##########| 100/100 [00:10<00:00, 10it/s]\n"
            "stage2:  10%|#         | 20/200 [00:02<00:18, 10it/s]")
    assert parse_progress(text) == (20, 200)


def test_p4_no_counter_is_none_not_a_guess():
    """Ordinary driver output must never be mistaken for progress."""
    assert parse_progress("  31779 trials (19963 bonafide)\n  model loaded") is None
    assert parse_progress("") is None
    assert parse_progress("wrote /data/scores/raw/linear_head/wild/x.txt") is None


def test_p5_rejects_an_impossible_counter():
    """done > total means it is not a progress counter -- a date, a ratio."""
    assert parse_progress("computed 2024/1980 something [x") is None
    assert parse_progress("split 5/0 [") is None


# ===========================================================================
# P6-P7: reading it back off disk
# ===========================================================================

def test_p6_tail_reads_only_the_end_of_a_large_log(tmp_path):
    """A multi-hour log is megabytes; the reader must not depend on its size."""
    log = tmp_path / "task.log"
    log.write_text("filler line\n" * 50000 + TQDM)
    assert tail_progress(str(log)) == (450, 994)


def test_p7_missing_log_is_none_not_an_exception(tmp_path):
    """A task that has not opened its log yet must not crash the display."""
    assert tail_progress(str(tmp_path / "nope.log")) is None


# ===========================================================================
# P8-P11: the reporter's accounting
# ===========================================================================

def _reporter(total=4, style="plain"):
    return ProgressReporter(total, title="t", stream=io.StringIO(), style=style)


def test_p8_finish_is_idempotent():
    """The worker's `finally` calls finish again; it must not count twice."""
    r = _reporter()
    r.start_task("gpu0", "a/b/c", None)
    r.finish_task("gpu0", "ok")
    r.finish_task("gpu0", "failed")
    assert (r.done, r.failed) == (1, 0)


def test_p9_unknown_slot_is_not_counted():
    """finish without start is the crashed-before-reporting path, not a task."""
    r = _reporter()
    r.finish_task("gpu2", "failed")
    assert (r.done, r.failed) == (0, 0)


def test_p10_in_flight_work_counts_toward_the_headline(tmp_path):
    """A half-finished task must move the bar, or a 3-hour column looks stuck."""
    log = tmp_path / "t.log"
    log.write_text(TQDM)                        # 450/994 = 0.4527
    r = _reporter(total=4)
    r.done = 1
    r.start_task("gpu0", "a/b/c", str(log))
    _, effective = r._snapshot()
    assert effective == pytest.approx(1 + 450 / 994, abs=1e-9)


def test_p10b_headline_reaches_100_only_when_the_last_task_ends():
    """A finished inner loop is not a finished task -- verification follows it."""
    r = _reporter(total=2)
    r.done = 1
    assert r._display_fraction(2.0) == pytest.approx(0.999)   # 1 done + 1 at 100%
    r.done = 2
    assert r._display_fraction(2.0) == 1.0


def test_p11_eta_scales_with_remaining_work():
    """ETA is throughput-based: half done after T means about T remaining."""
    r = _reporter(total=10)
    r.t0 -= 100.0
    assert r._eta(5) == pytest.approx(100.0, rel=0.05)
    assert r._eta(0) is None                    # nothing measured yet


# ===========================================================================
# P12-P14: rendering
# ===========================================================================

def test_p12_plain_style_emits_no_escape_codes():
    """`nohup ... > run.log` must stay readable -- no cursor movement in it."""
    out = io.StringIO()
    r = ProgressReporter(3, title="all", stream=out, style="plain")
    r.start_task("gpu0", "linear_head/wild/xls_r_300m", None, expect_lines=31779)
    r.finish_task("gpu0", "ok")
    r._render()
    text = out.getvalue()
    assert "\x1b" not in text
    assert "1/3 tasks" in text


def test_p12b_plain_style_writes_only_on_its_timer():
    """State changes must not add lines, or a 312-task log is mostly status.

    The completion message itself still goes through; what must not appear is
    a status line beside it.
    """
    out = io.StringIO()
    r = ProgressReporter(3, title="all", stream=out, style="plain")
    r.start_task("gpu0", "a/b/c", None)
    r.write("OK       a/b/c: 40 lines")
    r.finish_task("gpu0", "ok")
    assert out.getvalue() == "OK       a/b/c: 40 lines\n"


def test_p13_bar_clears_exactly_what_it_drew():
    """One erase per line drawn, or completion messages get overwritten."""
    out = io.StringIO()
    r = ProgressReporter(3, title="all", stream=out, style="bar")
    r.start_task("gpu0", "a/b/c", None)
    r.start_task("gpu1", "d/e/f", None)
    assert r._lines == 4                        # headline + bar + 2 slots
    out.truncate(0), out.seek(0)
    r._clear()
    assert out.getvalue().count("\x1b[1A\x1b[2K") == 4
    assert r._lines == 0


def test_p14_slot_line_reports_trials_from_the_protocol():
    """The percentage is the subprocess's; the total is the protocol's.

    They come from different sources and must not be presented as one number:
    the torch loop counts batches, so 450/994 is not 450 trials.
    """
    line = ProgressReporter._slot_line("gpu0", "linear_head/wild/xls_r_300m",
                                       450 / 994, 31779, 130.0)
    assert "45.3%" in line
    assert "31,779" in line
    assert "994" not in line
    assert "0:02:10" in line


def test_p15_unstarted_slot_shows_no_percentage():
    """Before the first tqdm frame there is no fraction; do not print 0.0%."""
    line = ProgressReporter._slot_line("gpu0", "a/b/c", None, 100, 5.0)
    assert "%" not in line


def test_p15b_phase_replaces_a_stalled_hundred_percent():
    """After the loop ends the slot is counting and verifying, not hung."""
    r = _reporter()
    r.start_task("gpu0", "a/b/c", None, expect_lines=2065873)
    r.set_phase("gpu0", "verify")
    rows, _ = r._snapshot()
    line = ProgressReporter._slot_line(*rows[0])
    assert "verify" in line
    assert "%" not in line and "2,065,873" not in line


# ===========================================================================
# P16-P17: selection
# ===========================================================================

def test_p16_auto_follows_the_tty():
    """`auto` is only safe to default to if it actually detects redirection."""
    class Tty(io.StringIO):
        def isatty(self): return True

    assert make_reporter("auto", 5, stream=Tty()).style == "bar"
    assert make_reporter("auto", 5, stream=io.StringIO()).style == "plain"


def test_p17_none_is_a_silent_passthrough(capsys):
    """--progress none still prints task results; it drops only the display."""
    r = make_reporter("none", 5)
    assert isinstance(r, NullReporter)
    r.start_task("gpu0", "a/b/c", None)
    r.write("OK       a/b/c: 100 lines")
    out = capsys.readouterr().out
    assert "OK       a/b/c: 100 lines" in out
    assert "tasks" not in out


def test_p18_fmt_hms_handles_the_unknown_eta():
    assert fmt_hms(None) == "--:--:--"
    assert fmt_hms(float("nan")) == "--:--:--"
    assert fmt_hms(3661) == "1:01:01"
    assert fmt_hms(0) == "0:00:00"


def test_p19_driver_wires_the_reporter_through():
    """run_task must accept the reporter, or the display shows an empty sweep."""
    import inspect

    from spoof_superb.orchestration import driver
    params = inspect.signature(driver.run_task).parameters
    assert "reporter" in params and "slot" in params
    assert params["reporter"].default is None   # importable without a display


def test_p20_backends_interval_matches_the_reader():
    """The log is the channel; a huge tqdm interval would freeze the display."""
    from spoof_superb.scoring import backends
    assert 0 < backends.PROGRESS_INTERVAL_S <= 15.0
