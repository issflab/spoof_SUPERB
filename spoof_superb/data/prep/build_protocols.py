"""
build_protocols.py
------------------
Build protocol files for the corpora that do not ship a usable one.

Every dataset should define its trial list through a protocol file. Most
corpora ship one; two do not, and this writes them once so scoring never has to
enumerate a directory at run time.

Why that matters beyond tidiness:

  * A directory walk has no record. Two runs months apart can enumerate
    different sets -- a corpus re-download, an added file, a filesystem that
    orders differently -- and nothing in the output says so. A protocol file is
    a reviewable artifact you can diff.
  * The trial list stops depending on machine state. The same protocol scores
    the same trials anywhere.
  * For ASVLD it also removes a structural problem: the layout wants one score
    file per (system, dataset, frontend), but the conditions were scored one
    per run. A combined protocol makes that a single run producing a single
    file, with the condition carried per row.

Both outputs land beside their corpus, in the same tab-separated shape:

    utt_id <TAB> label <TAB> [extra columns]

Usage
-----
    python -m spoof_superb.data.prep.build_protocols mailabs --dry-run
    python -m spoof_superb.data.prep.build_protocols mailabs
    python -m spoof_superb.data.prep.build_protocols asvld
"""

import argparse
import os

from spoof_superb.config import cfg
from spoof_superb.scoring.datasets import ASVLD_CONDITIONS, ASVLD_PROTOCOL_TEMPLATE

MAILABS_HEADER = ["utt_id", "label", "language"]
ASVLD_HEADER = ["utt_id", "label", "condition", "variant"]


def _write(path, header, rows, dry_run=False):
    if dry_run:
        print(f"  would write {len(rows)} rows -> {path}")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    os.replace(tmp, path)
    print(f"  wrote {len(rows)} rows -> {path}")
    return 0


def build_mailabs(args):
    """M-AILABS ships no protocol. It is entirely bonafide speech.

    utt_ids are written relative to data_root, matching the convention MLAAD
    already uses, so the two can be pooled without rewriting ids.
    """
    root = args.root or os.path.join(cfg.data_root, "MAILabs")
    out = args.out or os.path.join(root, "protocol.txt")
    if not os.path.isdir(root):
        print(f"[ERROR] not found: {root}")
        return 2

    rows, skipped = [], 0
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".wav"):
                continue
            # macOS AppleDouble sidecars are 176-245 byte metadata stubs, not
            # audio. M-AILABS carries 6 and they are absent from the published
            # score files.
            if fn.startswith("._"):
                skipped += 1
                continue
            full = os.path.join(dirpath, fn)
            utt = os.path.relpath(full, cfg.data_root)
            # .../MAILabs/{lang}/by_book/... -> {lang}
            rel = os.path.relpath(full, root).split(os.sep)
            language = rel[0] if len(rel) > 1 else "unknown"
            rows.append((utt, "bonafide", language))

    rows.sort()
    if skipped:
        print(f"  skipped {skipped} AppleDouble sidecar(s)")
    print(f"  {len(rows)} utterances, all bonafide")
    return _write(out, MAILABS_HEADER, rows, args.dry_run)


def build_asvld(args):
    """One protocol pooling the ASVLD laundering conditions.

    The five condition protocols are mutually disjoint -- 2,065,873 rows,
    2,065,873 distinct utt_ids, zero collisions -- because the condition is
    encoded in the utt_id. Pooling therefore loses nothing, and the published
    ASVLD score file is already a pooled file of exactly this kind.

    The condition is kept as a column so per-condition analysis stays possible
    without re-deriving it from the id.
    """
    proto_dir = args.protocols_dir or os.path.join(
        cfg.data_root, "ASVSpoofLaunderedDatabase", "ASVspoofLD", "protocols")
    out = args.out or os.path.join(os.path.dirname(proto_dir), "protocol.txt")
    conditions = args.conditions or ASVLD_CONDITIONS

    rows, seen = [], set()
    for cond in conditions:
        path = os.path.join(proto_dir, ASVLD_PROTOCOL_TEMPLATE.format(condition=cond))
        if not os.path.isfile(path):
            print(f"  [WARN] missing protocol for {cond}: {path}")
            continue
        n = 0
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) < 4:
                    continue
                utt, label = p[1], p[3]
                if utt in seen:
                    # Would silently drop rows on pooling; refuse rather than
                    # produce a protocol that is quietly short.
                    print(f"[ERROR] duplicate utt_id across conditions: {utt}")
                    return 1
                seen.add(utt)
                variant = p[5] if len(p) > 5 else "-"
                rows.append((utt, label, cond, variant))
                n += 1
        print(f"  {cond:16s} {n} rows")

    if not rows:
        print("[ERROR] no protocol rows found")
        return 1
    n_bona = sum(1 for r in rows if r[1] == "bonafide")
    print(f"  {len(rows)} utterances ({n_bona} bonafide, {len(rows) - n_bona} spoof)")
    return _write(out, ASVLD_HEADER, rows, args.dry_run)


BUILDERS = {"mailabs": build_mailabs, "asvld": build_asvld}


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.data.prep.build_protocols",
        description="Write protocol files for corpora that ship none")
    ap.add_argument("dataset", choices=sorted(BUILDERS))
    ap.add_argument("--root", default=None, help="mailabs: corpus root")
    ap.add_argument("--protocols_dir", default=None, help="asvld: condition protocols")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="asvld: which conditions to pool (default: all five)")
    ap.add_argument("--out", default=None, help="output protocol path")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args(argv)
    return BUILDERS[args.dataset](args)


if __name__ == "__main__":
    raise SystemExit(main())
