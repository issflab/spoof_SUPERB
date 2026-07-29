"""Live progress for a running sweep.

The orchestrator launches each scoring task as a subprocess whose stdout goes to
a per-task log file, so the terminal stayed silent between task completions --
on the 2M-row ASVLD column that is hours with no evidence the run is alive.

Two levels of progress exist and both are recovered here:

  outer   how many of the job's tasks are done. The orchestrator owns this
          number exactly; it is just never displayed.

  inner   how far the current subprocess is through its trials. The subprocess
          owns it, and already writes it -- ``backends.py`` wraps its loader in
          tqdm. That output lands in the task log. Rather than adding a second
          progress channel to the scoring driver (two writers, one truth, the
          duplication this repo has been removing), the log tail is read and
          tqdm's own ``n/total`` counter is parsed back out.

The coupling that buys is to tqdm's ``450/1000 [`` fragment, which is stable
across tqdm's whole 4.x line, and its failure mode is graceful: an unparseable
log costs the inner percentage, never the outer count.

Two renderers, chosen by whether stdout is a terminal:

  bar     redraws a block in place. For someone watching.
  plain   prints one status line every ``interval`` seconds. For ``nohup``,
          where a redrawing bar would write megabytes of escape codes into a
          log nobody can read.

``auto`` picks between them, which is what makes it safe to leave on by default.
"""

import os
import re
import sys
import threading
import time

# tqdm renders "desc:  45%|####  | 450/1000 [00:30<00:37, 14.8it/s]". The
# trailing bracket anchors the counter to the position tqdm puts it in, so a
# number pair appearing in a model name or a path cannot be mistaken for it.
_COUNTER_ANCHORED = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
_COUNTER_LOOSE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

BAR_WIDTH = 26
TAIL_BYTES = 16384
# tqdm and the warnings share one stream, so a frame can be buried arbitrarily
# far back: the ASV21-DF logs carry 121,640 librosa audioread warnings and push
# the live counter ~200 KB behind the end of the file. The window grows until it
# finds one rather than assuming a depth.
MAX_TAIL_BYTES = 8 << 20
_WINDOW = {}          # path -> the window size that last worked for it


def parse_progress(text):
    """(done, total) from the most recent tqdm update in ``text``, or None.

    tqdm separates updates with a carriage return, so the last chunk is the
    current state. Chunks are scanned newest-first: a finished bar followed by
    a second one must report the second.
    """
    for chunk in reversed(re.split(r"[\r\n]", text)):
        if not chunk.strip():
            continue
        m = _COUNTER_ANCHORED.search(chunk) or _COUNTER_LOOSE.search(chunk)
        if not m:
            continue
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0 and done <= total:
            return done, total
    return None


def tail_progress(path, nbytes=None, max_bytes=MAX_TAIL_BYTES):
    """(done, total) from the tail of a task log, or None if not determinable.

    The window grows 8x at a time until a counter turns up, because how deep
    the newest frame is buried depends on how noisy the subprocess is -- a
    quiet log needs 16 KB, an ASV21-DF log needs 200 KB of librosa warnings
    skipped first. The window that worked is remembered per path, so the search
    is paid once per task rather than once per redraw.
    """
    window = nbytes or _WINDOW.get(path, TAIL_BYTES)
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    while True:
        try:
            with open(path, "rb") as f:
                f.seek(max(0, size - window))
                blob = f.read(window)
        except OSError:
            return None
        got = parse_progress(blob.decode("utf-8", "replace"))
        if got:
            _WINDOW[path] = window
            return got
        if nbytes or window >= max_bytes or window >= size:
            return None
        window *= 8


def fmt_hms(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:   # NaN-safe
        return "--:--:--"
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _bar(fraction, width=BAR_WIDTH):
    fraction = 0.0 if fraction is None else max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


class NullReporter:
    """The --progress none case, so call sites never branch."""

    def start_task(self, slot, name, log_file, expect_lines=None): pass

    def set_phase(self, slot, phase): pass

    def finish_task(self, slot, status): pass

    def write(self, msg): print(msg, flush=True)

    def start(self): pass

    def stop(self): pass


class ProgressReporter:
    """Outer task counter plus per-worker inner progress.

    Every line the orchestrator prints goes through :meth:`write` so that the
    redrawn block and the scrolling completion messages cannot interleave --
    one lock owns the stream.
    """

    def __init__(self, total, title="", stream=None, style="bar", interval=None):
        self.total = total
        self.title = title
        self.stream = stream or sys.stdout
        self.style = style                       # "bar" | "plain"
        self.interval = interval if interval is not None else (1.0 if style == "bar" else 60.0)
        self.t0 = time.time()
        self.done = 0
        self.failed = 0
        self.slots = {}                          # slot -> dict(name, log, t0, expect)
        self._lock = threading.RLock()
        self._lines = 0                          # lines the last redraw left on screen
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._clear()

    def _loop(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                self._render()

    # -- task state ---------------------------------------------------------

    def start_task(self, slot, name, log_file, expect_lines=None):
        with self._lock:
            self.slots[slot] = {"name": name, "log": log_file, "t0": time.time(),
                                "expect": expect_lines, "phase": None}
            self._on_change()

    def set_phase(self, slot, phase):
        """Name what a slot is doing when it is no longer the scoring loop.

        Reading back and verifying a 2M-row score file takes minutes. Without
        this the slot would sit at 100% for that whole time, which reads as a
        hang rather than as the work it is.
        """
        with self._lock:
            if slot in self.slots:
                self.slots[slot]["phase"] = phase
                self._on_change()

    def finish_task(self, slot, status):
        """Idempotent: counts only a slot that is still registered.

        That lets the worker call it again in a ``finally`` to catch a task
        that raised before reporting, without double-counting the normal path.
        """
        with self._lock:
            if self.slots.pop(slot, None) is None:
                return
            self.done += 1
            if status not in ("ok", "skipped"):
                self.failed += 1
            self._on_change()

    def write(self, msg):
        with self._lock:
            self._clear()
            print(msg, file=self.stream, flush=True)
            self._on_change()

    # -- rendering ----------------------------------------------------------

    def _on_change(self):
        """Redraw on a state change -- but only where redrawing is free.

        In bar style the frame is overwritten in place, so reacting instantly
        costs nothing. In plain style every render is another line in a log
        file, and a sweep that starts and finishes 312 tasks would triple the
        status lines it was asked for. There, the timer is the only writer.
        """
        if self.style == "bar":
            self._render()

    def _clear(self):
        if self.style != "bar" or not self._lines:
            return
        # Up one line and erase, for each line the previous frame wrote.
        self.stream.write("\r" + "\x1b[1A\x1b[2K" * self._lines)
        self.stream.flush()
        self._lines = 0

    def _snapshot(self):
        """(rows, effective_done) -- rows are (slot, name, fraction, expect, elapsed).

        ``fraction`` comes from the subprocess, ``expect`` from the protocol.
        They are deliberately not multiplied together for display: the torch
        loops count batches, not trials, so a raw counter of 450/994 next to a
        31,779-trial dataset would look like a different quantity -- which it
        is. The fraction is the part that transfers; the trial total is
        reported from the side that actually knows it.
        """
        rows, partial = [], 0.0
        for slot in sorted(self.slots):
            s = self.slots[slot]
            prog = tail_progress(s["log"]) if s["log"] else None
            fraction = None
            if prog:
                fraction = prog[0] / prog[1]
                partial += fraction
            rows.append((slot, s["name"], fraction, s["expect"],
                         time.time() - s["t0"], s.get("phase")))
        return rows, self.done + partial

    def _display_fraction(self, effective_done):
        """Overall completion, held below 1.0 until the last task really ends.

        The inner fraction reaches 1.0 when the subprocess's loop finishes, but
        the task is not done until its score file is read back and verified --
        seconds to minutes on a 2M-row column. Showing 100% there would report
        a finish that has not happened.
        """
        if not self.total:
            return 0.0
        f = effective_done / self.total
        return min(f, 0.999) if self.done < self.total else min(f, 1.0)

    def _eta(self, effective_done):
        elapsed = time.time() - self.t0
        if effective_done <= 0 or elapsed <= 0:
            return None
        return (self.total - effective_done) * (elapsed / effective_done)

    def _headline(self, effective_done):
        pct = 100.0 * self._display_fraction(effective_done)
        fail = f"  {self.failed} failed" if self.failed else ""
        return (f"[{self.title}] {self.done}/{self.total} tasks  {pct:5.1f}%"
                f"  elapsed {fmt_hms(time.time() - self.t0)}"
                f"  eta {fmt_hms(self._eta(effective_done))}{fail}")

    @staticmethod
    def _slot_line(slot, name, fraction, expect, elapsed, phase=None, prefix="  "):
        if phase:
            pct, of = f"{phase:>6.6s}", "".ljust(20)
        else:
            pct = "  ... " if fraction is None else f"{100.0 * fraction:5.1f}%"
            of = (f"of {expect:,} trials" if expect else "").ljust(20)
        return f"{prefix}{slot:<9} {name:<40.40s} {pct}  {of}{fmt_hms(elapsed)}"

    def _render(self):
        rows, effective_done = self._snapshot()
        if self.style == "bar":
            self._clear()
            lines = [self._headline(effective_done),
                     f"  [{_bar(self._display_fraction(effective_done))}]"]
            lines += [self._slot_line(*r) for r in rows]
            self.stream.write("\n".join(lines) + "\n")
            self.stream.flush()
            self._lines = len(lines)
        else:
            parts = [self._headline(effective_done)]
            for slot, name, fraction, _, _, phase in rows:
                pct = phase or ("..." if fraction is None
                                else f"{100.0 * fraction:.0f}%")
                parts.append(f"{slot} {name} {pct}")
            print(" | ".join(parts), file=self.stream, flush=True)


def make_reporter(mode, total, title="", stream=None):
    """Build the reporter named by --progress; ``auto`` resolves against the tty."""
    stream = stream or sys.stdout
    if mode == "none":
        return NullReporter()
    if mode == "auto":
        mode = "bar" if stream.isatty() else "plain"
    return ProgressReporter(total, title=title, stream=stream, style=mode)
