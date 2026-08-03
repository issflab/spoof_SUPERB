"""Where a score file lives.

One function decides every score-file path, so the directory layout is a
property of the code rather than a convention each script remembers
differently. Switching layouts is then a config change, not a refactor.

The v3 layout (current)
-----------------------
    {scores_root}/raw/{method}/{dataset}/{varies}.txt

    method    linear_head | non_ssl                   what family produced it
    dataset   canonical name, version included        what was scored
    varies    the s3prl upstream, or the system       what differs in this dir

For `linear_head` the third level is the s3prl upstream, exactly as in v2. For
the non-SSL reference systems it is the system itself:

    raw/linear_head/mlaad_v10/xls_r_300m.txt
    raw/non_ssl/mlaad_v10/lfcc_gmm.txt
    raw/non_ssl/mlaad_v10/aasist_raw.txt

v2 wrote the baselines as raw/{system}/{dataset}/none.txt. That was honest --
they take no upstream, so no upstream was named -- but it spent a directory
level on a system already named one level up, and left every baseline score
file on disk with the same filename. `none.txt` identifies nothing once it is
out of its directory: in a log line, an attachment, or a copy, it is
unrecoverable. Naming the file after the system fixes that, and once the system
is in the filename the per-system directory is redundant, which is why the two
collapse into `non_ssl/`.

The glob property is preserved and gains one case:

    raw/*/*/xls_r_300m.txt        one upstream, everywhere
    raw/linear_head/mlaad_v10/*.txt   every SSL model on one set
    raw/non_ssl/*/lfcc_gmm.txt    one baseline, everywhere
    raw/non_ssl/mlaad_v10/*.txt   both baselines on one set  <- new

`non_ssl` rather than `baselines` because "baseline" is a role in an argument,
which can change if a system is added or promoted, while "not self-supervised"
is a property of the systems themselves, which cannot. Same reason the layout
refuses to write a default upstream for them.

The v2 layout
-------------
    {scores_root}/raw/{system}/{dataset}/{frontend}.txt

    system    linear_head | aasist_raw | lfcc_gmm     what produced the score
    dataset   canonical name, version included        what was scored
    frontend  the s3prl upstream, or 'none'           which encoder

Retained for reading, not writing. Every run_status.json records absolute
paths, so dropping v2 would make existing run histories uninterpretable.

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

__all__ = ["LAYOUTS", "NON_SSL_SYSTEMS", "NON_SSL_DIR", "DATASET_DIRS",
           "score_path", "score_dir", "canonical_dataset", "mlaad_pool_paths",
           "available_frontends"]

LAYOUTS = ("v3", "v2", "legacy")

#: Systems with no self-supervised front-end. They share one directory in v3
#: and are the only systems the legacy layout knows how to write.
NON_SSL_SYSTEMS = ("aasist_raw", "lfcc_gmm")

#: Directory holding the non-SSL systems in v3.
NON_SSL_DIR = "non_ssl"

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
    "deepfake_eval_2024_segmented": "deepfake_eval_2024_segmented",
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

    `frontend` is the s3prl upstream. The non-SSL systems take no upstream: in
    v3 their path has no frontend component at all, and the argument is ignored;
    in v2 and legacy it is recorded as 'none'.
    """
    root = scores_root or cfg.scores_root
    layout = layout or getattr(cfg, "score_layout", "legacy")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown score_layout {layout!r}; expected one of "
                         f"{', '.join(LAYOUTS)}")

    if layout == "v3":
        if system in NON_SSL_SYSTEMS:
            return os.path.join(root, "raw", NON_SSL_DIR,
                                canonical_dataset(dataset), f"{system}{ext}")
        return os.path.join(root, "raw", system, canonical_dataset(dataset),
                            f"{frontend}{ext}")

    if layout == "v2":
        return os.path.join(root, "raw", system, canonical_dataset(dataset),
                            f"{frontend}{ext}")

    # legacy
    if system in NON_SSL_SYSTEMS:
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


def mlaad_pool_paths(frontend, scores_root=None, layout=None,
                     system="linear_head"):
    """The score files that make up the MLAAD benchmark column, in pool order.

    MLAAD is the one benchmark column that is not a single score file. It is a
    two-class pool assembled from two single-class corpora:

        MLAAD v10   456,000 spoof rows      the deepfakes
        M-AILABS    584,006 bonafide rows   the real speech they were built from

    1,040,006 rows per model. M-AILABS is not a benchmark column of its own --
    it is single-class, so it has no EER -- which is why `DATASET_DIRS` carries
    the note that it is scored separately and merged in afterwards.

    The two layouts differ only in whether the pooling already happened on disk:

        legacy   one file. The v10 tsv was written with M-AILABS already
                 concatenated into it, so the pool is a single read.
        v2/v3    two files, one per corpus, pooled here at read time.

    v3 is the better arrangement and the reason is not aesthetic: a file that
    silently contains two corpora cannot be re-pooled at a different ratio,
    cannot be checked for either corpus's row count independently, and gives a
    misleading answer to "how many MLAAD utterances are there". Keeping them
    apart makes the pooling an explicit, inspectable step -- this function.

    Returns a list of paths. Callers concatenate them; nothing else about the
    column depends on which layout produced it.
    """
    layout = layout or getattr(cfg, "score_layout", "legacy")
    if layout not in LAYOUTS:
        raise ValueError(f"unknown score_layout {layout!r}; expected one of "
                         f"{', '.join(LAYOUTS)}")

    if layout == "legacy":
        if system != "linear_head":
            raise KeyError(
                f"legacy has no MLAAD pool for system {system!r}. The non-SSL "
                f"systems were scored on MLAAD only after the reorganisation; "
                f"read them from a v2 or v3 tree.")
        root = scores_root or cfg.scores_root
        return [os.path.join(root, "linear_head_MLAAD_v10", "tsv",
                             f"linear_head_MLAAD_v10_{frontend}.tsv")]

    # The tsv twin is what must be read: ~8.6% of MLAAD utt_ids contain spaces
    # (vendor directories like "Cartesia.ai (Sonic-3)"). M-AILABS ids never do,
    # so it has no tsv twin and is read from the canonical .txt.
    return [
        score_path(system, "Multilingual", frontend,
                   scores_root=scores_root, layout=layout, ext=".tsv"),
        score_path(system, "MAILABS", frontend,
                   scores_root=scores_root, layout=layout),
    ]


def available_frontends(system, dataset, scores_root=None, layout=None,
                        ext=".txt"):
    """Every frontend that has a score file for one (system, dataset), sorted.

    Analysis scripts need to enumerate "which models were scored on this set",
    and each had done it by globbing a directory and string-stripping a prefix
    off the filenames. That works until a model name contains the separator, so
    the enumeration is done here instead, by INVERTING the layout's own naming
    rule rather than by guessing at it.

    Under v2/v3 the frontend is a whole path component, so there is nothing to
    invert: the file stems are the answer. Under legacy the frontend is embedded
    in a filename, so the template that wrote it is split on its `{frontend}`
    placeholder and the surrounding literals are removed -- which is exact,
    where a regex over underscores is not.

    Returns [] if the directory does not exist: "nothing was scored" is a normal
    state to report, not an error.
    """
    layout = layout or getattr(cfg, "score_layout", "legacy")

    if layout == "legacy":
        if system in NON_SSL_SYSTEMS or system != "linear_head":
            raise KeyError(f"no legacy enumeration for system {system!r}")
        spec = _LEGACY_LINEAR_HEAD.get(dataset)
        if spec is None:
            raise KeyError(f"no legacy path convention for linear_head/{dataset}")
        subdir, template = spec
        # The tsv twin lives in a sibling directory, not beside the .txt.
        directory = os.path.join(scores_root or cfg.scores_root, subdir)
        if ext == ".tsv":
            directory = os.path.join(directory, "tsv")
        prefix, suffix = template.split("{frontend}")
        # `suffix` is a bare extension like '.txt'. os.path.splitext is wrong
        # here: it reads a leading dot as a hidden-file NAME, so splitext('.txt')
        # is ('.txt', '') and the extension swap silently produced '.txt.tsv'.
        suffix = suffix[:suffix.rindex(".")] + ext if "." in suffix else suffix + ext
        if not os.path.isdir(directory):
            return []
        return sorted(
            name[len(prefix):-len(suffix)]
            for name in os.listdir(directory)
            if name.startswith(prefix) and name.endswith(suffix)
        )

    directory = score_dir(system, dataset, scores_root, layout)
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(directory)
        if name.endswith(ext)
        and (system in NON_SSL_SYSTEMS) == (os.path.splitext(name)[0] in NON_SSL_SYSTEMS)
    )


def score_dir(system, dataset, scores_root=None, layout=None):
    """Directory holding the score files for one (system, dataset).

    Under v3 the non-SSL systems share a directory, so this returns a directory
    containing sibling systems' files as well; it is the containing directory,
    not an exclusive one.
    """
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
