"""
build_reference_pack.py
-----------------------
Build the small, shippable reference pack from the full score-file tree.

The full score files are ~6 GB (more once everything is scored at full
protocol), which is too large to version in git. But almost nothing a user
needs requires the full files:

  * to know WHAT to score          -> the trial list
  * to know WHAT ANSWER to expect  -> the per-file EER and counts
  * to know whether their scores RANK the same way -> a subsample

The verification policies grade on Spearman / Pearson / sign-agreement, and a
2,000-row subsample reproduces the full-file Spearman to within ~3e-6 against a
0.99 threshold. So the pack is a few MB and lets someone clone the repo, score
one model, and verify it immediately -- with no large download.

The full score files still belong somewhere; they just belong in a release
asset or a DOI archive with a checksum manifest, not in git history.

Reads the score tree; writes only into --out_dir. Nothing is modified.

Output
------
    trials/published/{dataset}.tsv.gz        utt_id, label
    reference/summary.json                   per (dataset, model): counts, EER,
                                             score quantiles, sha256
    reference/subsample/{dataset}/{model}.tsv.gz   utt_id, label, score

The subsample draws the SAME utt_ids for every model of a given dataset, from a
seed derived from the dataset name, so files are comparable across models and
regenerating the pack is deterministic.

Usage
-----
    python -m spoof_superb.tools.build_reference_pack --dry-run
    python -m spoof_superb.tools.build_reference_pack --datasets wild --limit 3
    python -m spoof_superb.tools.build_reference_pack
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
from datetime import date

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.core.metrics import compute_eer
from spoof_superb.scoring.datasets import DATASETS, reference_paths

SUBSAMPLE_N = 2000
SEED_BASE = 1234
QUANTILES = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def discover_models(dataset, reference_ssl_placeholder="\x00"):
    """SSL model names that have a published score file for this dataset.

    Derived by splitting the registry's own filename template on the {ssl}
    slot and globbing, so no filename parsing is needed -- model names contain
    underscores and cannot be split out reliably any other way.
    """
    template = reference_paths(dataset, reference_ssl_placeholder)[0]
    prefix, suffix = template.split(reference_ssl_placeholder)
    out = []
    for path in sorted(glob.glob(prefix + "*" + suffix)):
        name = path[len(prefix):len(path) - len(suffix)] if suffix else path[len(prefix):]
        if name:
            out.append(name)
    return out


def read_score_file(path):
    """([utt_id], {utt_id: label}, {utt_id: score}) from a 4-column score file."""
    utts, labels, scores = [], {}, {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 3)
            if len(parts) != 4:
                continue
            utt, _, label, raw = parts
            try:
                value = float(raw)
            except ValueError:
                value = float("nan")
            utts.append(utt)
            labels[utt] = label
            scores[utt] = value
    return utts, labels, scores


def file_stats(labels, scores):
    values = np.array([scores[u] for u in scores], dtype=float)
    finite = np.isfinite(values)
    bona = np.array([scores[u] for u in scores if labels[u] == "bonafide"], dtype=float)
    spoof = np.array([scores[u] for u in scores if labels[u] == "spoof"], dtype=float)
    bona, spoof = bona[np.isfinite(bona)], spoof[np.isfinite(spoof)]

    eer = None
    if len(bona) and len(spoof):
        eer = float(compute_eer(bona, spoof)[0] * 100)

    q = {}
    if finite.any():
        for p, v in zip(QUANTILES, np.quantile(values[finite], QUANTILES)):
            q[str(p)] = round(float(v), 6)

    return {
        "n_rows": len(scores),
        "n_bonafide": int((np.array([labels[u] for u in scores]) == "bonafide").sum()),
        "n_spoof": int((np.array([labels[u] for u in scores]) == "spoof").sum()),
        "n_nonfinite": int((~finite).sum()),
        "eer_percent": None if eer is None else round(eer, 6),
        "score_quantiles": q,
    }


def write_tsv_gz(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with gzip.open(tmp, "wt", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    os.replace(tmp, path)
    return os.path.getsize(path)


def pick_subsample(utts, labels, n, seed):
    """Deterministic class-stratified sample of utt_ids."""
    bona = sorted(u for u in utts if labels[u] == "bonafide")
    spoof = sorted(u for u in utts if labels[u] == "spoof")
    rng = random.Random(seed)

    if not bona or not spoof:
        pool = sorted(set(utts))
        return sorted(rng.sample(pool, min(n, len(pool))))

    share = len(bona) / (len(bona) + len(spoof))
    n_bona = max(1, min(len(bona), round(n * share)))
    n_spoof = max(1, min(len(spoof), n - n_bona))
    return sorted(rng.sample(bona, n_bona) + rng.sample(spoof, n_spoof))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.tools.build_reference_pack",
        description="Build the shippable trial lists, summaries and subsamples")
    ap.add_argument("--out_dir", default=str(REPO_ROOT),
                    help="pack root; writes trials/ and reference/ under it")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these datasets (default: all in the registry)")
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE_N)
    ap.add_argument("--seed", type=int, default=SEED_BASE)
    ap.add_argument("--trial_list_from", default=None,
                    help="SSL model whose score file defines each trial list "
                         "(default: the config's reference_ssl)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N models per dataset (smoke test)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args(argv)

    from spoof_superb.config import cfg
    trial_ssl = args.trial_list_from or cfg.reference_ssl

    datasets = args.datasets or list(DATASETS)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        print(f"[ERROR] unknown dataset(s): {', '.join(unknown)}")
        return 2

    plan = {}
    for ds in datasets:
        models = discover_models(ds)
        if args.limit:
            models = models[:args.limit]
        plan[ds] = models
        print(f"  {ds:22s} {len(models):3d} model(s) with a published score file")

    if args.dry_run:
        total = sum(len(m) for m in plan.values())
        print(f"\nwould write to {args.out_dir}")
        print(f"  trials/published/*.tsv.gz         {len(plan)} files")
        print(f"  reference/subsample/*/*.tsv.gz    {total} files "
              f"({args.subsample} rows each)")
        print(f"  reference/summary.json            {total} entries")
        return 0

    summary = {
        "generated": date.today().isoformat(),
        "subsample_rows": args.subsample,
        "seed_base": args.seed,
        "trial_list_from": trial_ssl,
        "datasets": {},
    }
    written = 0

    for ds, models in plan.items():
        if not models:
            print(f"[{ds}] no score files found; skipped")
            continue

        # --- trial list, from one designated model's file -------------------
        trial_src = trial_ssl if trial_ssl in models else models[0]
        paths = reference_paths(ds, trial_src)
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            print(f"[{ds}] trial-list source missing: {missing[0]}")
            continue

        utts, labels, _ = read_score_file(paths[0])
        for extra in paths[1:]:
            u2, l2, _ = read_score_file(extra)
            utts += u2
            labels.update(l2)

        trials_path = os.path.join(args.out_dir, "trials", "published", f"{ds}.tsv.gz")
        size = write_tsv_gz(trials_path, ["utt_id", "label"],
                            ((u, labels[u]) for u in utts))
        print(f"[{ds}] trials {len(utts)} rows -> {size / 1024:.0f} KB "
              f"(from {trial_src})")

        sample_ids = pick_subsample(utts, labels, args.subsample,
                                    args.seed + (abs(hash(ds)) % 100000))

        summary["datasets"][ds] = {
            "trial_list_source": os.path.basename(paths[0]),
            "trial_list_rows": len(utts),
            "subsample_rows": len(sample_ids),
            "models": {},
        }

        for model in models:
            mpaths = reference_paths(ds, model)
            if any(not os.path.isfile(p) for p in mpaths):
                continue
            m_utts, m_labels, m_scores = read_score_file(mpaths[0])
            for extra in mpaths[1:]:
                u2, l2, s2 = read_score_file(extra)
                m_utts += u2
                m_labels.update(l2)
                m_scores.update(s2)

            stats = file_stats(m_labels, m_scores)
            stats["sha256"] = sha256(mpaths[0])
            stats["source"] = os.path.basename(mpaths[0])
            summary["datasets"][ds]["models"][model] = stats

            rows = [(u, m_labels[u], repr(m_scores[u]))
                    for u in sample_ids if u in m_scores]
            sub_path = os.path.join(args.out_dir, "reference", "subsample", ds,
                                    f"{model}.tsv.gz")
            write_tsv_gz(sub_path, ["utt_id", "label", "score"], rows)
            written += 1
            print(f"    {model:32s} n={stats['n_rows']:>9} "
                  f"EER={stats['eer_percent'] if stats['eer_percent'] is None else round(stats['eer_percent'], 3)!s:>8} "
                  f"sub={len(rows)}", flush=True)

    summary_path = os.path.join(args.out_dir, "reference", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    tmp = summary_path + ".part"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    os.replace(tmp, summary_path)

    print(f"\nsummary  -> {summary_path} ({os.path.getsize(summary_path) / 1024:.0f} KB)")
    print(f"subsample files written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
