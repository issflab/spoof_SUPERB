"""
build_reference.py
------------------
Publisher-side tool: freeze a finished analysis run as the reference tables.

    python -m spoof_superb.tools.build_reference --from outputs

The reference for level-2 verification is NOT the paper's LaTeX. It is the set
of tables this pipeline produces from the published score tree, which is a
deliberate choice and the one that makes the benchmark reproducible:

  * The LaTeX carries numbers whose source no longer exists. Two published
    columns (ASV19 LA and ITW) do not regenerate from any score file in either
    tree, on identical trials with zero label disagreement. A reference that
    nobody -- including us -- can reproduce is not a reference, it is a target
    that trains people to ignore the check.
  * The tables here were computed by code that ships in this repo, from score
    files whose sha256 is published in `reference/manifest.json`. Every number
    has a path back to bytes anyone can download.

So this is the contract offered to a future user: run the three analyses, and
these are the tables you should get.

The CSVs are small (tens of KB) and go into git. The score files behind them do
not; they are indexed by `build_release_manifest` and fetched on demand.

What is copied is exactly what `verification.analysis.TABLES` verifies -- one
list, so the published set and the checked set cannot drift apart.
"""

import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

from spoof_superb import REPO_ROOT
from spoof_superb.config import cfg
from spoof_superb.verification.analysis import TABLES

DEFAULT_OUT = REPO_ROOT / "reference" / "analysis"


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.build_reference",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", default=None,
                    help="outputs root holding main_results/, degradation/, "
                         "tts/ (default: the configured analysis_root)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--scores_root", default=None,
                    help="recorded as the provenance of these tables "
                         "(default: the configured scores_root)")
    ap.add_argument("--note", default="",
                    help="free text recorded in REFERENCE.md")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src or (cfg.analysis_dir))
    out = Path(args.out)

    found, missing = [], []
    for sub, name, _opts in TABLES:
        p = src / sub / name
        (found if p.is_file() else missing).append(p)

    for p in found:
        print(f"  found   {p}")
    for p in missing:
        print(f"  MISSING {p}")

    if missing:
        print(f"\n{len(missing)} table(s) missing. A partial reference would "
              f"silently exempt those tables from verification, so nothing was "
              f"written. Run the analyses that produce them first.")
        return 2

    if args.dry_run:
        print(f"\nwould copy {len(found)} table(s) to {out}")
        return 0

    for sub, name, _opts in TABLES:
        dest = out / sub / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / sub / name, dest)

    provenance = {
        "generated": date.today().isoformat(),
        "built_from": str(src),
        "scores_root": args.scores_root or cfg.scores_root,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_describe": _git("describe", "--always", "--dirty"),
        "tables": [f"{sub}/{name}" for sub, name, _ in TABLES],
        "note": args.note,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    readme = [
        "# Reference analysis tables",
        "",
        "These are the tables `spoof_superb.verification analysis` checks a "
        "reproduction against.",
        "",
        f"- generated: {provenance['generated']}",
        f"- from score tree: `{provenance['scores_root']}`",
        f"- built from outputs: `{provenance['built_from']}`",
        f"- repo commit: `{provenance['git_commit'] or 'unknown'}`",
        "",
        "Reproduce them with:",
        "",
        "```bash",
        "python -m spoof_superb.analysis.recompute_main_results",
        "python -m spoof_superb.analysis.acoustic_degradation",
        "python -m spoof_superb.analysis.tts_systems",
        "python -m spoof_superb.verification analysis",
        "```",
        "",
        "The score files behind them are indexed in `../manifest.json` with a "
        "sha256 each, and fetched with `bin/fetch_release.sh`.",
        "",
    ]
    if args.note:
        readme += [f"> {args.note}", ""]
    (out / "REFERENCE.md").write_text("\n".join(readme))

    print(f"\nwrote {len(found)} table(s) + provenance.json + REFERENCE.md "
          f"to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
