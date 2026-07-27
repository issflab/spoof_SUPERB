"""
append_mailabs.py
-----------------
Append the staged M-AILABS (bonafide) scores onto each MLAAD v10 score file,
producing one combined file per model:
    linear_head_MLAAD_v10_<ssl>.txt  =  456,000 spoof  +  584,012 bonafide

Safety properties:
  - Idempotent: a target that already contains MAILabs/ rows is skipped, so
    re-running never double-appends.
  - Atomic: the combined file is built as a .part and renamed into place, so an
    interrupted run cannot leave a truncated or half-appended score file.
  - Non-destructive: the staged mailabs/ files are kept, so the MLAAD-only view
    can always be recovered (grep -v '^MAILabs/') and the append re-done.

Run with --dry-run first to see what would change.
"""
import argparse
import glob
import os
import sys

BASE = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10"
STAGE = os.path.join(BASE, "mailabs")
EXPECT_MLAAD = 456000
EXPECT_MAILABS = 584006


def first_field(line):
    return line.rsplit(" ", 3)[0] if line.count(" ") >= 3 else line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    staged = sorted(glob.glob(os.path.join(STAGE, "linear_head_MAILABS_*.txt")))
    if not staged:
        print("no staged M-AILABS files found")
        return 1

    n_done = n_skip = n_bad = 0
    for src in staged:
        ssl = os.path.basename(src)[len("linear_head_MAILABS_"): -len(".txt")]
        dst = os.path.join(BASE, f"linear_head_MLAAD_v10_{ssl}.txt")
        if not os.path.isfile(dst):
            print(f"{ssl:<40} SKIP (no MLAAD file)")
            n_bad += 1
            continue

        n_src = sum(1 for _ in open(src))
        if n_src != EXPECT_MAILABS:
            print(f"{ssl:<40} SKIP (staged has {n_src} lines, expected {EXPECT_MAILABS})")
            n_bad += 1
            continue

        # Idempotency guard: does the target already contain bonafide rows?
        has_mailabs = False
        n_dst = 0
        with open(dst) as f:
            for line in f:
                n_dst += 1
                if line.startswith("MAILabs/"):
                    has_mailabs = True
                    break
        if has_mailabs:
            print(f"{ssl:<40} SKIP (already contains MAILabs rows)")
            n_skip += 1
            continue

        n_dst = sum(1 for _ in open(dst))
        if n_dst != EXPECT_MLAAD:
            print(f"{ssl:<40} SKIP (MLAAD file has {n_dst} lines, expected {EXPECT_MLAAD})")
            n_bad += 1
            continue

        if args.dry_run:
            print(f"{ssl:<40} would append {n_src} -> total {n_dst + n_src}")
            n_done += 1
            continue

        tmp = dst + ".part"
        with open(tmp, "w") as fo:
            with open(dst) as fi:
                for line in fi:
                    fo.write(line)
            with open(src) as fi:
                for line in fi:
                    fo.write(line)
        total = sum(1 for _ in open(tmp))
        if total != n_dst + n_src:
            os.unlink(tmp)
            print(f"{ssl:<40} FAILED (combined {total} != {n_dst + n_src})")
            n_bad += 1
            continue
        os.replace(tmp, dst)
        print(f"{ssl:<40} appended {n_src} -> total {total}")
        n_done += 1

    print(f"\n{'DRY-RUN ' if args.dry_run else ''}appended={n_done} "
          f"skipped={n_skip} problems={n_bad}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
