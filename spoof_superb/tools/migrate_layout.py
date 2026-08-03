"""Move a score tree from the v2 layout to v3.

v2 wrote the non-SSL systems as ``raw/{system}/{dataset}/none.txt``; v3 writes
``raw/non_ssl/{dataset}/{system}.txt``. `linear_head` does not move. See
`spoof_superb.core.scorepath` for why.

Destinations are computed by calling ``score_path(..., layout="v3")``, never by
string surgery, so this script cannot drift from the layout rule it implements.

Safety, in the order it matters:

* **Dry run is the default.** ``--apply`` is required to touch anything.
* **Copy, verify, then delete.** Each file is copied, both ends are hashed, and
  the source is removed only if the digests match and only under
  ``--delete-source``. A score file here can represent 15+ minutes of scoring,
  so nothing is moved in a way that loses the original on a partial write.
* **Never silently overwrite.** An existing destination is an error unless
  ``--force``, and even then it is re-verified after copying.
* **Idempotent.** Re-running after a completed migration reports nothing to do.

Run:
    python -m spoof_superb.tools.migrate_layout                  # dry run
    python -m spoof_superb.tools.migrate_layout --apply
    python -m spoof_superb.tools.migrate_layout --apply --delete-source
"""

import argparse
import hashlib
import os
import shutil
import sys

from spoof_superb.config import cfg
from spoof_superb.core.scorepath import (
    DATASET_DIRS,
    NON_SSL_SYSTEMS,
    canonical_dataset,
    score_path,
)

#: Reverse of DATASET_DIRS: on-disk directory -> registry key. score_path takes
#: the registry key, and the tree on disk carries the directory name.
_DIR_TO_KEY = {v: k for k, v in DATASET_DIRS.items()}


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def plan(scores_root):
    """[(src, dst, nbytes)] for every v2 non-SSL score file that must move.

    Files already at their v3 destination are skipped, which is what makes a
    second run a no-op.
    """
    raw = os.path.join(scores_root, "raw")
    moves = []
    for system in NON_SSL_SYSTEMS:
        sysdir = os.path.join(raw, system)
        if not os.path.isdir(sysdir):
            continue
        for dsdir in sorted(os.listdir(sysdir)):
            key = _DIR_TO_KEY.get(dsdir)
            if key is None:
                raise KeyError(
                    f"{os.path.join(sysdir, dsdir)}: directory {dsdir!r} is not "
                    f"a known dataset. Refusing to guess where it belongs; add "
                    f"it to scorepath.DATASET_DIRS or move it aside.")
            src_dir = os.path.join(sysdir, dsdir)
            if not os.path.isdir(src_dir):
                continue
            for name in sorted(os.listdir(src_dir)):
                src = os.path.join(src_dir, name)
                if not os.path.isfile(src):
                    continue
                ext = os.path.splitext(name)[1] or ".txt"
                dst = score_path(system, key, scores_root=scores_root,
                                 layout="v3", ext=ext)
                if os.path.abspath(src) == os.path.abspath(dst):
                    continue
                moves.append((src, dst, os.path.getsize(src)))
    return moves


def classify(scores_root):
    """[(src, dst, nbytes, state)] with state in {new, copied, clash}.

    `copied` means the destination already holds a byte-identical file. That is
    the normal state after a copy-only pass, which is the recommended way to
    run this: copy first, inspect, then delete sources later. Treating it as a
    clash would make that two-phase workflow impossible without --force, and
    --force is the wrong habit to build for it -- it would also overwrite a
    destination that genuinely differs.
    """
    out = []
    for src, dst, n in plan(scores_root):
        if not os.path.exists(dst):
            state = "new"
        elif os.path.getsize(src) == os.path.getsize(dst) and \
                _sha256(src) == _sha256(dst):
            state = "copied"
        else:
            state = "clash"
        out.append((src, dst, n, state))
    return out


def migrate(scores_root, apply=False, delete_source=False, force=False,
            out=sys.stdout):
    moves = classify(scores_root)
    if not moves:
        print("nothing to do: no v2 non-SSL score files found under "
              f"{os.path.join(scores_root, 'raw')}", file=out)
        return 0

    MARK = {"new": "  ", "copied": "==", "clash": "!!"}
    total = sum(n for _, _, n, _ in moves)
    print(f"{len(moves)} file(s), {total / 1e9:.2f} GB", file=out)
    print(f"{'':2s} {'source':58s} -> destination", file=out)
    for src, dst, n, state in moves:
        print(f"{MARK[state]} {os.path.relpath(src, scores_root):58s} -> "
              f"{os.path.relpath(dst, scores_root)}   ({n / 1e6:.1f} MB)",
              file=out)

    n_new = sum(1 for *_, s in moves if s == "new")
    n_copied = sum(1 for *_, s in moves if s == "copied")
    clashes = [m for m in moves if m[3] == "clash"]
    if n_copied:
        print(f"\n== {n_copied} destination(s) already hold an identical copy "
              f"(nothing to re-copy)", file=out)
    if clashes and not force:
        print(f"\nABORT: {len(clashes)} destination(s) exist and DIFFER from "
              f"their source. Inspect them, then re-run with --force to "
              f"overwrite.", file=out)
        return 2

    if not apply:
        todo = []
        if n_new or clashes:
            todo.append(f"copy {n_new + len(clashes)}")
        if delete_source:
            todo.append(f"remove {len(moves)} source(s)")
        print(f"\ndry run: nothing was written. Re-run with --apply to "
              f"{' and '.join(todo) if todo else 'do nothing'}.", file=out)
        return 0

    copied = removed = 0
    for src, dst, _, state in moves:
        if state != "copied":
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            d_src, d_dst = _sha256(src), _sha256(dst)
            if d_src != d_dst:
                print(f"\nFAILED verification, source kept: {src}\n"
                      f"  src {d_src}\n  dst {d_dst}", file=out)
                return 3
            copied += 1
        if delete_source:
            os.remove(src)
            removed += 1
    parts = [f"copied and verified {copied} file(s)"] if copied else []
    parts.append(f"removed {removed} source(s)" if removed else "sources kept")
    print("\n" + "; ".join(parts), file=out)

    if delete_source:
        for system in NON_SSL_SYSTEMS:
            sysdir = os.path.join(scores_root, "raw", system)
            # topdown=False yields children before parents, which is what makes
            # a parent removable once its last child is gone. Do not sort this:
            # alphabetical order puts the parent first, it is then still
            # non-empty, and the emptied system directory is left behind.
            for dirpath, _, _ in os.walk(sysdir, topdown=False):
                if os.path.isdir(dirpath) and not os.listdir(dirpath):
                    os.rmdir(dirpath)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.migrate_layout",
        description="Move non-SSL score files from the v2 layout to v3.")
    ap.add_argument("--scores_root", default=None,
                    help="defaults to cfg.scores_root")
    ap.add_argument("--apply", action="store_true",
                    help="actually copy; without this the run is a dry run")
    ap.add_argument("--delete-source", action="store_true",
                    help="remove each source after its copy verifies")
    ap.add_argument("--force", action="store_true",
                    help="overwrite destinations that already exist")
    args = ap.parse_args(argv)
    root = args.scores_root or cfg.scores_root
    return migrate(root, apply=args.apply, delete_source=args.delete_source,
                   force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
