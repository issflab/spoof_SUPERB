"""Where a score file lives.

One function decides every score-file path, so the directory layout is a
property of the code rather than a convention each script remembers
differently. Switching layouts is then a config change, not a refactor.

The v2 layout
-------------
    {scores_root}/raw/{system}/{dataset}/{frontend}.txt

    system    linear_head | aasist_raw | lfcc_gmm     what produced the score
    dataset   canonical name, version included        what was scored
    frontend  the s3prl upstream, or 'none'           which encoder

Four properties this buys:

* **Nothing is parsed.** The old names could not be split reliably --
  `linear_head_Noise_Addition_wavlm_large_babble_10.txt` has no regex, because
  model names contain underscores. Directory levels have no such problem.
* **One glob per question.** `raw/*/*/xls_r_300m.txt` is every score file for
  one upstream; `raw/linear_head/mlaad_v10/*.txt` is every model on one set.
* **Versions are in the name.** `mlaad_v10` and `mlaad_legacy` are different
  datasets, not two directories you have to know to tell apart. That confusion
  is what put the wrong MLAAD column in an earlier draft.
* **No condition level.** ASVLD conditions live as rows inside one file per
  (system, dataset, frontend). Verified: the five ASVLD condition protocols are
  mutually disjoint (2,065,873 rows, 2,065,873 distinct utt_ids, 0 collisions)
  and the condition is recoverable from the utt_id suffix, so splitting by
  directory would add a level that carries no information. Pooling also forces
  a file to have one provenance instead of hiding a mixture.

The legacy layout
-----------------
Reproduces the paths the code wrote before the reorganisation, so an existing
tree stays readable and writable during a migration. It covers the paths the
orchestrator jobs actually produce; it is not a general inverse of the old
naming, because the old naming was not general.

Select with `score_layout` in configs/paths.yaml.
"""

import os

from spoof_superb.config import cfg

__all__ = ["LAYOUTS", "DATASET_DIRS", "score_path", "canonical_dataset"]

LAYOUTS = ("v2", "legacy")

#: Registry key -> canonical directory name.
#:
#: The registry keys are the CLI vocabulary (`--dataset wild`) and are left
#: alone; these are the on-disk names. They are lowercase, snake_case, and
#: carry the corpus version where one exists.
DATASET_DIRS = {
    "eval_2019":          "asvspoof2019_la_eval",
    "asvspoof2021_LA":    "asvspoof2021_la",
    "asvspoof2021_DF":    "asvspoof2021_df",
    "asvspoof5":          "asvspoof5",
    "deepfake_eval_2024": "deepfake_eval_2024",
    "wild":               "in_the_wild",
    "Famous_Figures":     "famous_figures",
    "spoofceleb":         "spoofceleb",
    "Multilingual":       "mlaad_v10",
    "asvspoofLD":         "asvspoof_ld",
    # Not a benchmark column: the bonafide counterpart to MLAAD, scored
    # separately and merged in afterwards.
    "MAILABS":            "mailabs",
}

#: Legacy write paths, as `(subdirectory, filename template)`.
#: Only the combinations the code actually wrote are listed; anything else in
#: legacy mode is an error rather than a guess.
_LEGACY_LINEAR_HEAD = {
    "Multilingual": ("linear_head_MLAAD_v10", "linear_head_MLAAD_v10_{frontend}.txt"),
    "MAILABS":      (os.path.join("linear_head_MLAAD_v10", "mailabs"),
                     "linear_head_MAILABS_{frontend}.txt"),
    "spoofceleb":   ("linear_head_SpoofCeleb", "linear_head_SpoofCeleb_{frontend}.txt"),
}


def canonical_dataset(dataset):
    """Canonical on-disk directory name for a registry key."""
    try:
        return DATASET_DIRS[dataset]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset!r}; add it to "
            f"spoof_superb.core.scorepath.DATASET_DIRS. Known: "
            f"{', '.join(sorted(DATASET_DIRS))}") from None


def score_path(system, dataset, frontend="none", scores_root=None,
               layout=None, ext=".txt"):
    """Absolute path of the score file for one (system, dataset, frontend).

    `frontend` is the s3prl upstream for `linear_head`, and 'none' for the
    baselines, which take no upstream -- recording a default upstream in their
    path would claim something untrue about them.
    """
    root = scores_root or cfg.scores_root
    layout = layout or getattr(cfg, "score_layout", "legacy")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown score_layout {layout!r}; expected one of "
                         f"{', '.join(LAYOUTS)}")

    if layout == "v2":
        return os.path.join(root, "raw", system, canonical_dataset(dataset),
                            f"{frontend}{ext}")

    # legacy
    if system in ("aasist_raw", "lfcc_gmm"):
        return os.path.join(root, "baselines", system, f"{system}_{dataset}{ext}")
    if system == "linear_head":
        spec = _LEGACY_LINEAR_HEAD.get(dataset)
        if spec is None:
            raise KeyError(
                f"no legacy write path for linear_head/{dataset}. The old tree "
                f"had no single convention for it; use layout='v2', or add the "
                f"mapping to scorepath._LEGACY_LINEAR_HEAD.")
        subdir, template = spec
        return os.path.join(root, subdir, template.format(frontend=frontend))
    raise KeyError(f"no legacy write path for system {system!r}")


def score_dir(system, dataset, scores_root=None, layout=None):
    """Directory holding every frontend's score file for one (system, dataset)."""
    return os.path.dirname(score_path(system, dataset, "x", scores_root, layout))


def main(argv=None):
    """Print the score-file path for one (system, dataset, frontend).

    Lets the shell scripts put their output in the configured layout without
    duplicating the rule:

        OUTPUT_FILE=$(python -m spoof_superb.core.scorepath \
            --system linear_head --dataset wild --frontend xls_r_300m)
    """
    import argparse
    ap = argparse.ArgumentParser(prog="python -m spoof_superb.core.scorepath")
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frontend", default="none")
    ap.add_argument("--layout", default=None, choices=[None, *LAYOUTS])
    ap.add_argument("--scores_root", default=None)
    args = ap.parse_args(argv)
    print(score_path(args.system, args.dataset, args.frontend,
                     scores_root=args.scores_root, layout=args.layout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
