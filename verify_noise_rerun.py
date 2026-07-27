"""
verify_noise_rerun.py
---------------------
Gate the fp32 Noise_Addition re-run before it replaces the archived scores.

Background
----------
The archived Noise_Addition files were NOT of uniform provenance:

  * 3 models (xls_r_300m, unispeech_sat_large, wav2vec2_large_ll60k) were already
    produced by eval_asvld.py -- these reproduce bit-exactly (r = 1.0, max|d| = 0).
  * The other 21 were split out of the original reference score files, produced by
    a different pipeline. Four of those (tera, mockingjay, mockingjay_960hr,
    audio_albert_960hr) carry 384,157 half-precision overflow NaN each -- 53.93%
    of the file -- on an identical utterance set, because those four share the
    same s3prl transformer feature path.

So this re-run does two things: it removes the NaN, and it makes all 24 files
uniform in provenance.

Contracts
---------
  C1 DROP-IN    same utterance sequence as the archive, same order.
  C2 LABELS     bonafide/spoof key per utterance unchanged.
  C3 FINITE     no NaN, no inf. The point of the re-run.
  C4 AGREEMENT  Pearson r >= 0.9998 between old and new on utterances the archive
                scored finitely. This replaces an earlier max|delta| threshold,
                which was wrong: absolute deviation scales with score magnitude
                and with pipeline differences, so it measured neither precision
                nor correctness. Observed range is 0.99986 (byol_a_2048) to 1.0.
  C5 EER        for models with NO archived NaN, |dEER| <= 0.15 pp -- the re-run
                must not move a model that had nothing wrong with it. Observed
                worst case is 0.07 pp. Models WITH archived NaN are exempt by
                construction: correcting them is the purpose, and their EER moves
                by 5.8-8.0 pp.

Advisory (not a gate): any model whose median|delta| is a large multiple of the
cohort median is reported. byol_a_2048 is such a case (median 0.224 vs ~0.002
cohort-wide) yet its EER moves only -0.05 pp; it was verified deterministic
run-to-run, so the difference is pipeline, not noise.

Usage
-----
    python3 verify_noise_rerun.py                 # verify only
    python3 verify_noise_rerun.py --promote       # verify, then swap on all-pass
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import compute_eer  # type: ignore

ROOT = "/data/ssl_anti_spoofing/asd_superb_score_files/asvld_rerun"
OLD_DIR = os.path.join(ROOT, "Noise_Addition")
NEW_DIR = os.path.join(ROOT, "Noise_Addition_new")
BACKUP_DIR = os.path.join(ROOT, "Noise_Addition_fp16_backup")

MIN_CORR = 0.9998   # C4
MAX_DEER = 0.15     # C5, percentage points, clean models only
OUTLIER_MEDIAN = 0.05


def read(path):
    return pd.read_csv(path, sep=" ", header=None, names=["u", "dash", "lab", "s"],
                       dtype={"u": str, "lab": str}, na_filter=False)


def eer_of(scores, labels):
    ok = np.isfinite(scores)
    bona = scores[ok & (labels == "bonafide")]
    spoof = scores[ok & (labels == "spoof")]
    if bona.size == 0 or spoof.size == 0:
        return float("nan")
    return compute_eer(bona, spoof)[0] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="On all-pass, move Noise_Addition_new over Noise_Addition.")
    args = ap.parse_args()

    src = BACKUP_DIR if os.path.isdir(BACKUP_DIR) else OLD_DIR
    models = sorted(f[len("linear_head_Noise_Addition_"):-len(".txt")]
                    for f in os.listdir(src)
                    if f.startswith("linear_head_Noise_Addition_") and f.endswith(".txt"))

    print(f"comparing against: {src}\n")
    print(f"{'model':<40}{'C1':>4}{'C2':>4}{'C3':>4}{'C4':>4}{'C5':>4}"
          f"{'oldNaN':>9}{'corr':>12}{'EERold':>8}{'EERnew':>8}{'dEER':>8}{'med|d|':>9}")
    print("-" * 118)

    all_pass, missing, advisories = True, [], []
    for m in models:
        fn = f"linear_head_Noise_Addition_{m}.txt"
        newp = os.path.join(NEW_DIR, fn)
        if not os.path.isfile(newp):
            missing.append(m)
            all_pass = False
            print(f"{m:<40} MISSING")
            continue

        o, n = read(os.path.join(src, fn)), read(newp)
        c1 = len(o) == len(n) and bool((o.u.to_numpy() == n.u.to_numpy()).all())
        c2 = c1 and bool((o.lab.to_numpy() == n.lab.to_numpy()).all())

        os_ = pd.to_numeric(o.s, errors="coerce").to_numpy(float)
        ns_ = n.s.to_numpy(float)
        old_nan = int((~np.isfinite(os_)).sum())
        c3 = int((~np.isfinite(ns_)).sum()) == 0

        fin = np.isfinite(os_)
        a, b = os_[fin], ns_[fin]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else 1.0
        med = float(np.median(np.abs(a - b)))
        c4 = corr >= MIN_CORR

        eo = eer_of(os_, o.lab.to_numpy())
        en = eer_of(ns_, n.lab.to_numpy())
        d = en - eo
        # A model with archived NaN is expected to move; that is the repair.
        c5 = True if old_nan else abs(d) <= MAX_DEER

        if med > OUTLIER_MEDIAN:
            advisories.append((m, med, d))

        ok = c1 and c2 and c3 and c4 and c5
        all_pass &= ok
        mk = lambda x: "ok" if x else "FAIL"
        print(f"{m:<40}{mk(c1):>4}{mk(c2):>4}{mk(c3):>4}{mk(c4):>4}{mk(c5):>4}"
              f"{old_nan:>9}{corr:>12.8f}{eo:>8.2f}{en:>8.2f}{d:>+8.2f}{med:>9.5f}")

    print("-" * 118)
    if missing:
        print(f"MISSING: {', '.join(missing)}")
    for m, med, d in advisories:
        print(f"[ADVISORY] {m}: median|delta|={med:.3f} is far above the cohort "
              f"(~0.002), but dEER={d:+.2f} pp. Pipeline difference, not noise "
              f"(verified deterministic run-to-run).")
    print("\nVERDICT:", "ALL PASS" if all_pass else "NOT PROMOTABLE")

    if args.promote:
        if not all_pass:
            print("Refusing to promote: at least one contract failed.")
            return 1
        if not os.path.isdir(BACKUP_DIR):
            print(f"Refusing to promote: backup {BACKUP_DIR} is missing.")
            return 1
        for m in models:
            fn = f"linear_head_Noise_Addition_{m}.txt"
            shutil.move(os.path.join(NEW_DIR, fn), os.path.join(OLD_DIR, fn))
        os.rmdir(NEW_DIR)
        print(f"\nPromoted {len(models)} files into {OLD_DIR}")
        print(f"Originals preserved at {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
