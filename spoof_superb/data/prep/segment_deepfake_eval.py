"""
segment_deepfake_eval.py
------------------------
Build a segmented version of Deepfake-Eval 2024 and its protocol.

Deepfake-Eval 2024 ships as whole recordings of wildly varying length (seconds
to minutes). The benchmark's models read a fixed 4.0375 s window
(CROP = 64,600 samples at 16 kHz), so a long recording contributes exactly one
scored window and the rest of its audio is never seen. Segmenting first turns
each recording into several trials and makes the whole corpus reachable.

The segmentation is OUR artifact -- it does not ship with the dataset -- so
this script regenerates it from the two things that do:

    {root}/audio-data/                    the original recordings
    {root}/audio-metadata-publish.csv     Filename, Ground Truth, ...

and writes, by default:

    {root}/segmented/wav/{stem}_seg{N}.wav
    {root}/segmented/protocol.txt

Nothing under {root}/data/ is read or written. That tree is a separate,
pre-existing local artifact with its own train/test and duration splits; this
one is flat by design.

Why wav and not mp3
-------------------
91% of the sources are already mp3. Re-encoding mp3 -> mp3 adds a second lossy
generation, and codec compression is one of the degradation conditions this
benchmark measures -- so re-encoding would inject exactly the artifact under
study into the clean condition. Segments are therefore decoded once and written
as 16 kHz mono PCM. Pass --format mp3 if you need the smaller files and accept
that.

Why ffmpeg
----------
The corpus contains 4 files with a `.dat` extension that are really MP4/M4A
containers. librosa cannot open them, which is why the published score file has
1,976 rows rather than 1,980. ffmpeg reads them, so segmenting recovers all
1,980 recordings.

Usage
-----
    python -m spoof_superb.data.prep.segment_deepfake_eval --dry-run
    python -m spoof_superb.data.prep.segment_deepfake_eval --jobs 16
    python -m spoof_superb.data.prep.segment_deepfake_eval --limit 20   # smoke test
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from spoof_superb.config import cfg

DEFAULT_ROOT = os.path.join(cfg.data_root, "Deepfake_Eval_2024")

SEGMENT_SECONDS = 4.0     # = CROP (64,600 samples) / 16,000 Hz, rounded down
MIN_SECONDS = 1.0         # trailing fragments shorter than this are discarded
SAMPLE_RATE = 16000

# Ground Truth in the metadata -> the benchmark's key vocabulary.
LABELS = {"real": "bonafide", "fake": "spoof"}

PROTOCOL_HEADER = ["segment_id", "source_file", "label", "start_s", "duration_s"]


def probe_duration(path):
    """Seconds of audio, or None if ffprobe cannot read the file."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def segment_one(task):
    """Cut one recording into fixed-length segments.

    Returns (rows, error). `rows` are protocol rows; `error` is a string when
    the recording could not be processed at all.
    """
    (path, stem, label, out_dir, seconds, min_seconds, fmt, sample_rate, force) = task

    duration = probe_duration(path)
    if duration is None or duration <= 0:
        return [], f"{stem}: unreadable ({os.path.basename(path)})"

    n_full = int(duration // seconds)
    tail = duration - n_full * seconds
    n_total = n_full + (1 if tail >= min_seconds else 0)
    if n_total == 0:
        # Shorter than min_seconds in total: keep it as a single short segment
        # rather than dropping the recording, since the model pads short input.
        n_total = 1

    expected = [os.path.join(out_dir, f"{stem}_seg{i + 1}.{fmt}") for i in range(n_total)]
    if not force and all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in expected):
        rows = []
        for i, p in enumerate(expected):
            start = i * seconds
            rows.append([f"{stem}_seg{i + 1}.{fmt}", os.path.basename(path), label,
                         f"{start:.3f}", f"{min(seconds, duration - start):.3f}"])
        return rows, None

    # ffmpeg's segment muxer decodes once and writes every piece, which is much
    # cheaper than one decode per segment on a long recording.
    tmp = tempfile.mkdtemp(prefix="dfeseg_", dir=out_dir)
    try:
        pattern = os.path.join(tmp, f"%d.{fmt}")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", path,
               "-ar", str(sample_rate), "-ac", "1",
               "-f", "segment", "-segment_time", str(seconds),
               "-reset_timestamps", "1", "-segment_start_number", "1"]
        if fmt == "wav":
            cmd += ["-c:a", "pcm_s16le"]
        cmd.append(pattern)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            return [], f"{stem}: ffmpeg failed -- {proc.stderr.strip()[:200]}"

        produced = sorted(os.listdir(tmp), key=lambda f: int(os.path.splitext(f)[0]))
        rows = []
        for f in produced:
            idx = int(os.path.splitext(f)[0])
            src = os.path.join(tmp, f)
            start = (idx - 1) * seconds
            seg_dur = probe_duration(src) or 0.0

            # Drop a trailing fragment that is too short to be a useful trial,
            # unless it is the only thing this recording produced.
            if seg_dur < min_seconds and len(produced) > 1:
                os.remove(src)
                continue

            dst = os.path.join(out_dir, f"{stem}_seg{idx}.{fmt}")
            shutil.move(src, dst)
            rows.append([f"{stem}_seg{idx}.{fmt}", os.path.basename(path), label,
                         f"{start:.3f}", f"{seg_dur:.3f}"])
        if not rows:
            return [], f"{stem}: produced no usable segments"
        return rows, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_metadata(csv_path, audio_dir):
    """[(path, stem, label)] for every recording the metadata lists."""
    by_stem = {}
    for fn in os.listdir(audio_dir):
        by_stem.setdefault(os.path.splitext(fn)[0], os.path.join(audio_dir, fn))

    tasks, missing, unlabelled = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("Filename") or "").strip()
            if not fn:
                continue
            stem = os.path.splitext(fn)[0]
            label = LABELS.get((row.get("Ground Truth") or "").strip().lower())
            if label is None:
                unlabelled.append(stem)
                continue
            path = by_stem.get(stem)
            if path is None:
                missing.append(stem)
                continue
            tasks.append((path, stem, label))
    return tasks, missing, unlabelled


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.data.prep.segment_deepfake_eval",
        description="Segment Deepfake-Eval 2024 and write its protocol")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="dataset root containing audio-data/ and the metadata CSV")
    ap.add_argument("--audio_dir", default=None, help="default {root}/audio-data")
    ap.add_argument("--metadata", default=None,
                    help="default {root}/audio-metadata-publish.csv")
    ap.add_argument("--out_dir", default=None, help="default {root}/segmented")
    ap.add_argument("--seconds", type=float, default=SEGMENT_SECONDS,
                    help=f"segment length (default {SEGMENT_SECONDS}, the model's crop)")
    ap.add_argument("--min_seconds", type=float, default=MIN_SECONDS,
                    help="discard a trailing fragment shorter than this")
    ap.add_argument("--format", default="wav", choices=["wav", "mp3"],
                    help="wav keeps the audio lossless; mp3 re-encodes (see module docstring)")
    ap.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N recordings")
    ap.add_argument("--force", action="store_true", help="re-cut even if segments exist")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="report what would be done and write nothing")
    args = ap.parse_args(argv)

    audio_dir = args.audio_dir or os.path.join(args.root, "audio-data")
    metadata = args.metadata or os.path.join(args.root, "audio-metadata-publish.csv")
    out_base = args.out_dir or os.path.join(args.root, "segmented")
    out_dir = os.path.join(out_base, args.format)
    protocol = os.path.join(out_base, "protocol.txt")

    for p in (audio_dir, metadata):
        if not os.path.exists(p):
            print(f"[ERROR] not found: {p}")
            return 2
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[ERROR] ffmpeg and ffprobe are required")
        return 2

    tasks, missing, unlabelled = read_metadata(metadata, audio_dir)
    print(f"metadata rows with audio on disk : {len(tasks)}")
    if missing:
        print(f"  [WARN] {len(missing)} metadata rows have no audio file (skipped)")
    if unlabelled:
        print(f"  [WARN] {len(unlabelled)} rows have no usable Ground Truth (skipped)")
    if args.limit:
        tasks = tasks[:args.limit]
        print(f"  limited to first {len(tasks)}")

    if args.dry_run:
        print(f"\nwould write segments to : {out_dir}")
        print(f"would write protocol to : {protocol}")
        print(f"segment length          : {args.seconds}s "
              f"(discard trailing < {args.min_seconds}s)")
        print(f"format                  : {args.format} @ {args.sample_rate} Hz mono")
        n_bona = sum(1 for _, _, l in tasks if l == "bonafide")
        print(f"recordings              : {len(tasks)} "
              f"({n_bona} bonafide, {len(tasks) - n_bona} spoof)")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    payload = [(p, s, l, out_dir, args.seconds, args.min_seconds,
                args.format, args.sample_rate, args.force) for p, s, l in tasks]

    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(segment_one, t): t[1] for t in payload}
        done = 0
        for fut in as_completed(futures):
            got, err = fut.result()
            rows.extend(got)
            if err:
                errors.append(err)
            done += 1
            if done % 200 == 0 or done == len(payload):
                print(f"  {done}/{len(payload)} recordings -> {len(rows)} segments",
                      flush=True)

    rows.sort(key=lambda r: (r[1], int(r[0].rsplit("_seg", 1)[1].split(".")[0])))

    tmp = protocol + ".part"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(PROTOCOL_HEADER)
        w.writerows(rows)
    os.replace(tmp, protocol)

    n_bona = sum(1 for r in rows if r[2] == "bonafide")
    print(f"\nsegments written : {len(rows)} ({n_bona} bonafide, {len(rows) - n_bona} spoof)")
    print(f"audio            : {out_dir}")
    print(f"protocol         : {protocol}")
    if errors:
        print(f"\n[WARN] {len(errors)} recordings failed:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
