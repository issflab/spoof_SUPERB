"""Score-file verification entry point.

    python -m spoof_superb.verification.driver --check spoofceleb \
        --new out.txt --ref reference.txt

Replaces verify_mlaad.py and verify_spoofceleb.py, which shared their parser
and their statistics and differed only in the verdict -- now a policy in
policies.py.

verify_asvld.py and verify_noise_rerun.py are NOT folded in here: the first is
descriptive (a table, no pass/fail) and the second is a promotion gate with a
--promote side effect that moves directories. They keep their own entry points.

Two lines of output. Coverage first, then the graded comparison:

    [coverage] scored 611829  reference 152955  overlap 152955
               reference-not-scored 0  scored-not-in-reference 458874
    [verify] new=.. ref=.. shared=.. r=.. spearman=.. sign@0=.. -> PASS

Coverage is REPORTED, never enforced. Scoring the full protocol where the
published column used a subset is a deliberate, legitimate difference; the
point is that it appears as a line of output instead of silently changing what
gets compared. The grade is computed on the overlap.
"""

import argparse
import json
import os

from spoof_superb import REPO_ROOT
from spoof_superb.verification.policies import grade, POLICIES
from spoof_superb.verification.stats import compare

DEFAULT_MANIFEST = os.path.join(str(REPO_ROOT), "reference", "manifest.json")


def coverage_line(c):
    return (f"[coverage] scored {c.n_new}  reference {c.n_ref}  "
            f"overlap {c.n_shared}\n"
            f"           reference-not-scored {c.n_ref - c.n_shared}  "
            f"scored-not-in-reference {c.n_new - c.n_shared}")


def expected_from_manifest(path, dataset, model):
    """The published entry for this (dataset, model), or None."""
    if not (path and os.path.isfile(path) and dataset and model):
        return None
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    entry = manifest.get("datasets", {}).get(dataset, {}).get(model)
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spoof_superb.verification.driver")
    ap.add_argument("--check", choices=sorted(POLICIES), required=True,
                    help="grade policy to apply")
    ap.add_argument("--new", required=True, help="freshly produced score file")
    ap.add_argument("--ref", required=True,
                    help="reference score file (fetch one with bin/fetch_scores.sh)")
    ap.add_argument("--dataset", default=None,
                    help="with --manifest: report the published EER for context")
    ap.add_argument("--model", default=None, help="with --manifest")
    ap.add_argument("--manifest", nargs="?", const=DEFAULT_MANIFEST, default=None,
                    help=f"release manifest to read the expected EER from "
                         f"(default {DEFAULT_MANIFEST})")
    args = ap.parse_args(argv)

    c = compare(args.new, args.ref)
    if c.n_shared == 0:
        print(f"[verify] {args.new}: NO SHARED utt_ids "
              f"(new={c.n_new}, ref={c.n_ref})")
        return 1

    print(coverage_line(c))

    expected = expected_from_manifest(args.manifest, args.dataset, args.model)
    if expected and expected.get("eer_percent") is not None:
        print(f"[expected] published EER = {expected['eer_percent']:.4f}% "
              f"over {expected['n_rows']} rows")

    v = grade(args.check, c)
    print(f"[verify] {c.line()} -> {v.status}"
          f"{f' ({v.reason})' if v.status != 'PASS' and v.reason else ''}")
    return v.returncode


if __name__ == "__main__":
    raise SystemExit(main())
