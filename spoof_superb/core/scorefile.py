"""Reading and writing the benchmark's canonical score-file format.

The format is four space-separated columns:

    {utt_id} - {key} {score}

Every scoring driver in this repo emitted that format with its own copy of the
reader and the writer. This module is the single implementation.

Two properties are load-bearing and easy to get wrong:

1. Fields are peeled from the RIGHT (``rsplit``), never split on whitespace.
   utt_ids legitimately contain spaces -- MLAAD v10 has 39,000 rows with TTS
   system directories like "Cartesia.ai (Sonic-3)", and Famous Figures ids are
   absolute paths. A left-split silently yields the wrong utt_id and reads "-"
   as the label for every one of them.

2. Writes are atomic (``.part`` then ``os.replace``). A multi-hour scoring run
   killed midway must not leave a truncated file that looks complete to the
   next reader.
"""

import os

import numpy as np

__all__ = ["read_reference", "read_scored", "write_scores", "report_eer"]


def read_scored(paths):
    """Read one or more score files -> (utt_ids, labels, scores) as arrays.

    `read_reference` answers "what should have been scored"; this answers "what
    was scored, and to what". Every analysis module needs the latter and each
    had grown its own reader, with three incompatible ideas of the format:

      * 4-column space-separated `utt_id - key score`   (canonical)
      * 4-column tab-separated, no header               (the .tsv twin)
      * 3-column tab-separated with a `utt_id label score` header
        (the legacy MLAAD v10 tsv only)

    All three appear on disk, so all three are read here rather than in four
    places. Which one a file is, is decided per line by its separator and field
    count -- not by its extension, because the legacy and v3 .tsv files share an
    extension and differ in both.

    Space-separated lines are peeled from the RIGHT for the reason given in the
    module docstring. Tab-separated lines are split on tabs, which is the whole
    point of the twin: tabs never occur inside an utt_id.

    Passing several paths concatenates them in order, which is how a column the
    benchmark defines as a pool of corpora (MLAAD + M-AILABS) is assembled.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    utts, labels, scores = [], [], []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    parts = line.split("\t")
                    if len(parts) == 4:            # utt_id, -, key, score
                        utt, key, score = parts[0], parts[2], parts[3]
                    elif len(parts) == 3:          # utt_id, label, score
                        utt, key, score = parts
                    else:
                        continue
                else:
                    parts = line.rsplit(" ", 3)
                    if len(parts) != 4:
                        continue
                    utt, key, score = parts[0], parts[2], parts[3]
                if key == "label":                 # the legacy tsv header
                    continue
                utts.append(utt)
                labels.append(key)
                scores.append(float(score))

    return (np.asarray(utts, dtype=object),
            np.asarray(labels, dtype=object),
            np.asarray(scores, dtype=np.float64))


def read_reference(paths):
    """Read one or more 4-column score files -> ([utt_id], {utt_id: key}).

    Accepts a list so a benchmark column the paper defines as the pool of
    several score files (ASVLD) is assembled exactly as published, in order.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    utts, keys = [], {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.rsplit(" ", 3)
                if len(parts) != 4:
                    continue
                utt, key = parts[0], parts[2]
                utts.append(utt)
                keys[utt] = key
    return utts, keys


def read_utt_ids(path):
    """utt_ids only (column 0), for --restrict_to against a reference file."""
    utts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            utts.append(line.rsplit(" ", 3)[0] if line.count(" ") >= 3 else line.split()[0])
    return utts


def write_scores(output_file, scored, keys):
    """Write scored rows atomically; add a .tsv twin when utt_ids contain spaces.

    Space-separated is the canonical format, but ``np.genfromtxt`` -- and so
    ``core.metrics.calculate_EER`` -- cannot parse ids containing spaces. The
    repo's own answer is a tab-separated copy (see linear_head_MLAAD_v10/tsv/),
    so emit one whenever it is needed.

    Returns the path of the .tsv twin, or None.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    tmp = output_file + ".part"
    with open(tmp, "w") as fh:
        for utt, score in scored:
            fh.write("{} - {} {}\n".format(utt, keys[utt], score))
    os.replace(tmp, output_file)
    print(f"  scores saved -> {output_file}  ({len(scored)} lines)", flush=True)

    if not any(" " in utt for utt, _ in scored):
        return None

    tsv = os.path.splitext(output_file)[0] + ".tsv"
    tmp = tsv + ".part"
    with open(tmp, "w") as fh:
        for utt, score in scored:
            fh.write("{}\t-\t{}\t{}\n".format(utt, keys[utt], score))
    os.replace(tmp, tsv)
    print(f"  tab-separated copy -> {tsv} (utt_ids contain spaces)", flush=True)
    return tsv


def report_eer(scored, keys):
    """Print the inline EER for a freshly scored set. Diagnostic only.

    Computed from the in-memory arrays rather than via calculate_EER, which
    cannot read score files whose utt_ids contain spaces.
    """
    try:
        bona = np.array([s for u, s in scored if keys[u] == "bonafide"])
        spoof = np.array([s for u, s in scored if keys[u] == "spoof"])
        if len(bona) and len(spoof):
            from spoof_superb.core.metrics import compute_eer
            print(f"  EER = {compute_eer(bona, spoof)[0] * 100:.4f} %", flush=True)
        else:
            print(f"  [WARN] single-class output ({len(bona)} bona / {len(spoof)} spoof)")
    except Exception as exc:
        print(f"  [WARN] could not compute EER inline: {type(exc).__name__}: {exc}")
