"""Score-file verification entry point.

    python -m spoof_superb.verification.driver --check spoofceleb \
        --new out.txt --ref reference.txt

Replaces verify_mlaad.py and verify_spoofceleb.py, which shared their parser
and their statistics and differed only in the verdict -- now a policy in
policies.py.

verify_asvld.py and verify_noise_rerun.py are NOT folded in here: the first is
descriptive (a table, no pass/fail) and the second is a promotion gate with a
--promote side effect that moves directories. They keep their own entry points.

Output is one line, in the same shape the orchestrators parse:

    [verify] new=.. ref=.. shared=.. r=.. spearman=.. sign@0=.. -> PASS
"""

import argparse

from spoof_superb.verification.policies import POLICIES, grade
from spoof_superb.verification.stats import compare


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m spoof_superb.verification.driver")
    ap.add_argument("--check", choices=sorted(POLICIES), required=True,
                    help="grade policy to apply")
    ap.add_argument("--new", required=True, help="freshly produced score file")
    ap.add_argument("--ref", required=True, help="reference score file")
    args = ap.parse_args(argv)

    c = compare(args.new, args.ref)
    if c.n_shared == 0:
        print(f"[verify] {args.new}: NO SHARED utt_ids "
              f"(new={c.n_new}, ref={c.n_ref})")
        return 1

    v = grade(args.check, c)
    print(f"[verify] {c.line()} -> {v.status}"
          f"{f' ({v.reason})' if v.status != 'PASS' and v.reason else ''}")
    return v.returncode


if __name__ == "__main__":
    raise SystemExit(main())
