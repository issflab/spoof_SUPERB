"""Verify a score file against the shipped reference pack.

The pack (`trials/` + `reference/`) is what makes verification work for someone
who has just cloned the repo: a few MB, versioned, no 6 GB download. It answers
the three questions that matter without the full score files.

  coverage  did you score the trials the benchmark defines?
  EER       is your headline number where it should be?
  ranking   do your scores order the utterances the same way?

Ranking is graded on a 2,000-row subsample. That is not an approximation worth
worrying about: a 2,000-row sample reproduces the full-file Spearman to within
about 3e-6, against a pass threshold of 0.99.

Coverage is REPORTED, never enforced. Scoring the full protocol while the
published column used a subset is a legitimate, deliberate difference -- the
point is that it shows up as a line of output instead of silently changing what
gets compared.
"""

import gzip
import json
import os

from spoof_superb.verification.stats import Comparison, compare, load_scores

__all__ = ["PackPaths", "load_trials", "load_subsample", "coverage", "check_against_pack"]


class PackPaths:
    def __init__(self, root):
        self.root = root
        self.trials_dir = os.path.join(root, "trials", "published")
        self.subsample_dir = os.path.join(root, "reference", "subsample")
        self.summary = os.path.join(root, "reference", "summary.json")

    def trials(self, dataset):
        return os.path.join(self.trials_dir, f"{dataset}.tsv.gz")

    def subsample(self, dataset, model):
        return os.path.join(self.subsample_dir, dataset, f"{model}.tsv.gz")


def _read_tsv_gz(path):
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) == len(header):
                yield dict(zip(header, row))


def load_trials(path):
    """{utt_id: label} for the trial list the benchmark published."""
    return {r["utt_id"]: r["label"] for r in _read_tsv_gz(path)}


def load_subsample(path):
    """{utt_id: score} for the reference subsample."""
    out = {}
    for r in _read_tsv_gz(path):
        try:
            out[r["utt_id"]] = float(r["score"])
        except (KeyError, ValueError):
            continue
    return out


def coverage(scored_ids, trial_ids):
    """How your trial set relates to the published one."""
    scored, trials = set(scored_ids), set(trial_ids)
    return {
        "n_scored": len(scored),
        "n_published": len(trials),
        "n_overlap": len(scored & trials),
        "published_not_scored": len(trials - scored),
        "scored_not_published": len(scored - trials),
    }


def check_against_pack(new_path, dataset, model, pack_root, policy=None):
    """Compare a new score file to the pack. Returns (coverage, Comparison, expected).

    `expected` is the summary entry for this (dataset, model), or None when the
    pack does not carry one.
    """
    paths = PackPaths(pack_root)

    trials_path = paths.trials(dataset)
    sub_path = paths.subsample(dataset, model)
    for p in (trials_path, sub_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    new_scores = load_scores(new_path)
    trials = load_trials(trials_path)
    cov = coverage(new_scores, trials)

    # Grade ranking on the subsample only, restricted to ids you actually have.
    ref_sub = load_subsample(sub_path)
    shared = [u for u in ref_sub if u in new_scores]

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "new.txt")
        b = os.path.join(td, "ref.txt")
        with open(a, "w") as fa, open(b, "w") as fb:
            for u in shared:
                label = trials.get(u, "spoof")
                fa.write(f"{u} - {label} {new_scores[u]}\n")
                fb.write(f"{u} - {label} {ref_sub[u]}\n")
        cmp_ = compare(a, b) if shared else Comparison(len(new_scores), len(ref_sub),
                                                       0, 0, 0, 0)

    expected = None
    if os.path.isfile(paths.summary):
        with open(paths.summary) as f:
            summary = json.load(f)
        expected = (summary.get("datasets", {}).get(dataset, {})
                    .get("models", {}).get(model))

    return cov, cmp_, expected


def format_report(dataset, model, cov, cmp_, expected, verdict=None):
    lines = [
        f"[coverage] {dataset}/{model}  scored {cov['n_scored']}  "
        f"published {cov['n_published']}  overlap {cov['n_overlap']}",
        f"           published-not-scored {cov['published_not_scored']}  "
        f"scored-not-published {cov['scored_not_published']}",
        f"[ranking ] on {cmp_.n_both_finite} subsample rows: {cmp_.line()}",
    ]
    if expected and expected.get("eer_percent") is not None:
        lines.append(f"[expected] published EER = {expected['eer_percent']:.4f}% "
                     f"over {expected['n_rows']} rows")
    if verdict is not None:
        lines.append(f"[verdict ] {verdict.status}"
                     f"{f' ({verdict.reason})' if verdict.reason else ''}")
    return "\n".join(lines)
