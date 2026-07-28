"""
balance_mailabs.py
------------------
Produce a class-balanced variant of the combined score files:
    456,000 spoof (MLAAD v10)  +  456,000 bonafide (M-AILABS)

M-AILABS contributes 584,006 bonafide utterances, so 128,006 are dropped. The
subset is chosen by proportional stratified sampling over the 9 M-AILABS
languages (largest-remainder rounding to land on exactly 456,000), which keeps
the original language mix instead of truncating arbitrarily.

Two properties that matter:
  - The SAME bonafide utt_ids are kept for every model. The keep-set is derived
    once and reused, so per-model EERs stay comparable.
  - Deterministic: fixed seed, and the chosen ids are written to a manifest so
    the exact subset can be reproduced or audited later.

Non-destructive: the full (unbalanced) files are left untouched; balanced copies
are written to balanced/.
"""
import argparse
import glob
import os
import random
import sys
from collections import defaultdict

BASE = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10"
OUT = os.path.join(BASE, "balanced")
MANIFEST = os.path.join(OUT, "bonafide_keep_manifest.txt")
TARGET = 456000          # match the MLAAD v10 spoof count
SEED = 1234


def utt_of(line):
    return line.rsplit(" ", 3)[0]


def build_keep_set(src, target, seed):
    """Proportional stratified sample of bonafide utt_ids, by language."""
    by_lang = defaultdict(list)
    n_spoof = 0
    with open(src) as f:
        for line in f:
            if line.startswith("MAILabs/"):
                u = utt_of(line)
                by_lang[u.split("/")[1]].append(u)
            elif line.startswith("MLAAD/"):
                n_spoof += 1

    total = sum(len(v) for v in by_lang.values())
    print(f"  spoof={n_spoof} bonafide={total} target={target}")
    if n_spoof != target:
        print(f"  [WARN] spoof count {n_spoof} != target {target}")

    # Largest-remainder apportionment so the parts sum to exactly `target`.
    exact = {L: len(v) * target / total for L, v in by_lang.items()}
    alloc = {L: int(x) for L, x in exact.items()}
    short = target - sum(alloc.values())
    for L in sorted(exact, key=lambda L: exact[L] - alloc[L], reverse=True)[:short]:
        alloc[L] += 1

    rng = random.Random(seed)
    keep = set()
    for L in sorted(by_lang):
        pool = sorted(by_lang[L])          # sort first -> order independent of walk
        rng.shuffle(pool)
        keep.update(pool[: alloc[L]])
        print(f"    {L:<8} {len(by_lang[L]):>7} -> {alloc[L]:>7}")
    assert len(keep) == target, f"{len(keep)} != {target}"
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(BASE, "linear_head_MLAAD_v10_*.txt")))
    if not files:
        print("no combined score files found")
        return 1

    print(f"building keep-set from {os.path.basename(files[0])}")
    keep = build_keep_set(files[0], args.target, args.seed)

    if args.dry_run:
        print(f"\nDRY-RUN: would write {len(files)} balanced files "
              f"({args.target} spoof + {args.target} bonafide = {2*args.target} lines each)")
        return 0

    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST + ".part", "w") as fh:
        fh.write(f"# bonafide utt_ids kept for the balanced set\n")
        fh.write(f"# target={args.target} seed={args.seed} "
                 f"method=proportional-stratified-by-language\n")
        for u in sorted(keep):
            fh.write(u + "\n")
    os.replace(MANIFEST + ".part", MANIFEST)
    print(f"manifest -> {MANIFEST}")

    for src in files:
        base = os.path.basename(src).replace("linear_head_MLAAD_v10_",
                                             "linear_head_MLAAD_v10_balanced_")
        dst = os.path.join(OUT, base)
        n_sp = n_bf = 0
        with open(src) as fi, open(dst + ".part", "w") as fo:
            for line in fi:
                if line.startswith("MLAAD/"):
                    fo.write(line); n_sp += 1
                elif line.startswith("MAILabs/"):
                    if utt_of(line) in keep:
                        fo.write(line); n_bf += 1
        os.replace(dst + ".part", dst)
        flag = "" if (n_sp == args.target and n_bf == args.target) else "  <-- MISMATCH"
        print(f"{base}: {n_sp} spoof + {n_bf} bonafide = {n_sp + n_bf}{flag}", flush=True)

    print(f"\nwrote {len(files)} balanced files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
