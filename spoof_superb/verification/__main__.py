"""Verification entry point -- one command, two levels.

    python -m spoof_superb.verification all
    python -m spoof_superb.verification scores   [--ref-root TREE]
    python -m spoof_superb.verification analysis [--candidate DIR]

Verification is a SEPARATE STEP. Scoring never reads a score file it did not
write, and the analyses never compare against anything -- so a from-scratch
build cannot inherit an older tree's coverage, and a number in a table cannot
be quietly graded against a number the same script was trying to reproduce.

Level 1 compares score files; level 2 compares the analysis tables computed
from them. Level 1 answers "did the pipeline produce the same scores"; level 2
answers "do the same conclusions come out". They fail independently and both
are worth knowing: identical scores with a changed table means the analysis
code moved, and drifting scores with an intact table means the finding is
robust to the drift.
"""

import argparse
import sys
from pathlib import Path

from spoof_superb import REPO_ROOT
from spoof_superb.config import cfg
from spoof_superb.verification.analysis import verify_analysis
from spoof_superb.verification.report import (analysis_markdown, score_markdown,
                                              tally, write_reports)
from spoof_superb.verification.scores import verify_scores
from spoof_superb.verification.verdicts import IS_FAILURE

DEFAULT_MANIFEST = REPO_ROOT / "reference" / "manifest.json"
DEFAULT_REFERENCE_ANALYSIS = REPO_ROOT / "reference" / "analysis"


def _default_out(name):
    root = getattr(cfg, "outputs_root", "") or str(REPO_ROOT / "outputs")
    return Path(root) / "verification" / name


def _parse_rules(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            sys.exit(f"--*-id-rewrite expects OLD=NEW, got {p!r}")
        old, new = p.split("=", 1)
        out[old] = new
    return out


def _add_scores_args(ap):
    ap.add_argument("--candidate", default=None,
                    help="score tree to verify (default: the configured one)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="reference manifest for offline verification")
    ap.add_argument("--ref-root", default=None,
                    help="reference SCORE TREE; enables the full "
                         "per-utterance comparison instead of manifest mode")
    ap.add_argument("--models", nargs="*", default=None,
                    help="score-file slugs (default: the paper roster)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="benchmark columns (default: all)")
    ap.add_argument("--ref-id-rewrite", nargs="*", default=None, metavar="OLD=NEW",
                    help="rename whole path components in the REFERENCE utt_ids")
    ap.add_argument("--candidate-id-rewrite", nargs="*", default=None,
                    metavar="OLD=NEW", help="the same, for the candidate")


def _add_analysis_args(ap):
    ap.add_argument("--candidate", default=None,
                    help="outputs root holding main_results/, degradation/, "
                         "tts/ (default: the configured outputs_root)")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE_ANALYSIS),
                    help="reference analysis tables")
    ap.add_argument("--tables", nargs="*", default=None,
                    help="check only these: main_results, degradation, tts "
                         "(default: all six). For a caller that ran one "
                         "analysis and should not be told the other five are "
                         "missing.")


def run_scores(args):
    from spoof_superb.scoring.models import paper_models

    candidate = args.candidate or cfg.scores_root
    models = args.models or sorted(paper_models())

    if not args.ref_root and not Path(args.manifest).is_file():
        sys.exit(f"FATAL: no reference manifest at {args.manifest}\n"
                 f"       build one from a finished tree with\n"
                 f"         python -m spoof_superb.tools.build_release_manifest\n"
                 f"       or point --ref-root at a reference score tree.")

    print("=" * 78)
    print("LEVEL 1 -- score files")
    print("=" * 78)
    print(f"candidate : {candidate}")
    print(f"reference : {args.ref_root or args.manifest}"
          f"{'' if args.ref_root else '  (manifest)'}")
    print(flush=True)

    rows = verify_scores(
        candidate, models, datasets=args.datasets, ref_root=args.ref_root,
        manifest_path=args.manifest,
        rewrite_ref=_parse_rules(args.ref_id_rewrite),
        rewrite_cand=_parse_rules(args.candidate_id_rewrite))

    meta = {"candidate_root": candidate, "ref_root": args.ref_root,
            "manifest": args.manifest, "n_models": len(models),
            "n_datasets": len(rows) // max(len(models), 1)}
    out_dir = _default_out("scores")
    md, js = write_reports(out_dir, "score_verification",
                           score_markdown(rows, meta),
                           {"meta": meta, "cells": rows,
                            "verdicts": tally(rows)})
    _summarise(rows, md, js)
    return 1 if any(r["verdict"] in IS_FAILURE for r in rows) else 0


def run_analysis(args):
    candidate = args.candidate or (getattr(cfg, "outputs_root", "")
                                   or str(REPO_ROOT / "outputs"))

    if not Path(args.reference).is_dir():
        sys.exit(f"FATAL: no reference analysis tables at {args.reference}\n"
                 f"       publish a set with\n"
                 f"         python -m spoof_superb.tools.build_reference "
                 f"--from {candidate}")

    print("=" * 78)
    print("LEVEL 2 -- analysis tables")
    print("=" * 78)
    print(f"candidate : {candidate}")
    print(f"reference : {args.reference}")
    print(flush=True)

    entries = verify_analysis(candidate, args.reference,
                              tables=getattr(args, "tables", None))
    meta = {"candidate_root": candidate, "reference_root": args.reference,
            "tables": getattr(args, "tables", None) or "all"}
    out_dir = _default_out("analysis")
    md, js = write_reports(out_dir, "analysis_verification",
                           analysis_markdown(entries, meta),
                           {"meta": meta, "tables": entries,
                            "verdicts": tally(entries)})
    _summarise(entries, md, js)
    return 1 if any(e["verdict"] in IS_FAILURE for e in entries) else 0


def _summarise(rows, md, js):
    counts = tally(rows)
    print("\n=== VERDICTS ===")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}{'   (fails the run)' if k in IS_FAILURE else ''}")
    print(f"\nWrote {md}")
    print(f"Wrote {js}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.verification",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="level", required=True)

    s = sub.add_parser("scores", help="level 1: score files vs the reference")
    _add_scores_args(s)

    a = sub.add_parser("analysis", help="level 2: analysis tables vs the reference")
    _add_analysis_args(a)

    b = sub.add_parser("all", help="both levels; exits non-zero if either fails")
    _add_scores_args(b)
    b.add_argument("--analysis-candidate", default=None,
                   help="outputs root (default: the configured outputs_root)")
    b.add_argument("--reference-analysis", default=str(DEFAULT_REFERENCE_ANALYSIS))
    b.add_argument("--tables", nargs="*", default=None)

    args = ap.parse_args(argv)

    if args.level == "scores":
        return run_scores(args)
    if args.level == "analysis":
        return run_analysis(args)

    rc = run_scores(args)
    print()
    args.candidate = args.analysis_candidate
    args.reference = args.reference_analysis
    return max(rc, run_analysis(args))


if __name__ == "__main__":
    raise SystemExit(main())
