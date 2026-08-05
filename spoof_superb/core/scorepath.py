"""Where a score file lives.

One function decides every score-file path, so the directory layout is a
property of the code rather than a convention each script remembers
differently.

    {scores_root}/raw/{method}/{dataset}/{varies}.txt

    method    linear_head | non_ssl                   what family produced it
    dataset   canonical name, version included        what was scored
    varies    the s3prl upstream, or the system       what differs in this dir

For `linear_head` the third level is the s3prl upstream. For the non-SSL
reference systems it is the system itself:

    raw/linear_head/mlaad_v10/xls_r_300m.txt
    raw/non_ssl/mlaad_v10/lfcc_gmm.txt
    raw/non_ssl/mlaad_v10/aasist_raw.txt

`non_ssl` rather than `baselines` because "baseline" is a role in an argument,
which can change if a system is added or promoted, while "not self-supervised"
is a property of the systems themselves, which cannot. Same reason the layout
refuses to write a default upstream for them.

Four properties this buys:

* **Nothing is parsed.** The old names could not be split reliably --
  `linear_head_Noise_Addition_wavlm_large_babble_10.txt` has no regex, because
  model names contain underscores. Directory levels have no such problem.
* **One glob per question.** `raw/*/*/xls_r_300m.txt` is every score file for
  one upstream; `raw/linear_head/mlaad_v10/*.txt` is every model on one set;
  `raw/non_ssl/mlaad_v10/*.txt` is both baselines on one set.
* **Versions are in the name.** `mlaad_v10` and `mlaad_legacy` are different
  datasets, not two directories you have to know to tell apart. That confusion
  is what put the wrong MLAAD column in an earlier draft.
* **No condition level.** ASVLD conditions live as rows inside one file per
  (system, dataset, frontend). Verified: the five ASVLD condition protocols are
  mutually disjoint (2,065,873 rows, 2,065,873 distinct utt_ids, 0 collisions)
  and the condition is recoverable from the utt_id suffix, so splitting by
  directory would add a level that carries no information. Pooling also forces
  a file to have one provenance instead of hiding a mixture.

The layouts that used to be here
--------------------------------
Two older conventions -- `legacy` (the pre-reorganisation tree) and `v2` (an
intermediate that wrote the non-SSL systems as `raw/{system}/{dataset}/none.txt`)
-- were supported for reading until 2026-08-05. They are gone, along with the
`--layout` flag on thirteen commands and the `score_layout` setting.

Nothing a user of this benchmark can obtain is in either: the published tree is
v3, `reference/manifest.json` indexes it, and the score files are fetched
already in this layout. The legacy tree was also established not to be
authoritative -- two of its published columns do not regenerate from any score
file in it (see `docs/internal/PLANNED_CHANGES.md`, P12) -- so the comparison
those layouts existed to enable has been done, its answer recorded, and there is
nothing further to learn by keeping the code that read them.

Note this is about PATHS, not about file contents. `core.scorefile` still reads
the three on-disk shapes -- 4-column space, 4-column tab, 3-column tab with a
header -- because the v3 tree carries a `.tsv` twin beside every `.txt`.
"""

import os

from spoof_superb.config import cfg

__all__ = ["NON_SSL_SYSTEMS", "NON_SSL_DIR", "DATASET_DIRS", "score_path",
           "score_dir", "canonical_dataset", "mlaad_pool_paths",
           "available_frontends", "COLUMN_KEYS", "column_key"]

#: Systems with no self-supervised front-end. They share one directory in v3
#: and are the only systems the legacy layout knows how to write.
NON_SSL_SYSTEMS = ("aasist_raw", "lfcc_gmm")

#: Directory holding the non-SSL systems in v3.
NON_SSL_DIR = "non_ssl"

#: Benchmark column -> the registry key it actually reads.
#:
#: Only DFEval24 differs from its own name, and it differs because the
#: MEASUREMENT changed. The retired legacy tree scored `deepfake_eval_2024`:
#: one 4 s window per recording, 1,980 trials, so all but the first four seconds
#: of a minutes-long file was never looked at. This tree scores
#: `deepfake_eval_2024_segmented`: every 4 s window of every recording, 56,481
#: trials.
#:
#: Per-segment trials weight long recordings far more heavily, so those two EERs
#: are DIFFERENT QUANTITIES -- not a corrected value. That distinction outlives
#: the layout it came from: an earlier draft of the paper printed the
#: unsegmented column (1,976 trials, four files it could not decode), and anyone
#: comparing a number here against that draft is comparing two measurements.
#:
#: This lives here, once, because both the module that PRODUCES the paper's
#: table and the module that VERIFIES it resolve through it. They each carried
#: their own copy; had those drifted, verification would have compared two
#: different measurements and passed -- blind to exactly the class of error it
#: exists to catch.
COLUMN_KEYS = {
    "deepfake_eval_2024": "deepfake_eval_2024_segmented",
}


def column_key(dataset):
    """The registry key a benchmark column reads."""
    return COLUMN_KEYS.get(dataset, dataset)


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
               ext=".txt"):
    """Absolute path of the score file for one (system, dataset, frontend).

    `frontend` is the s3prl upstream. The non-SSL systems take no upstream, so
    their path has no frontend component at all and the argument is ignored.
    """
    root = scores_root or cfg.scores_root
    if system in NON_SSL_SYSTEMS:
        return os.path.join(root, "raw", NON_SSL_DIR,
                            canonical_dataset(dataset), f"{system}{ext}")
    return os.path.join(root, "raw", system, canonical_dataset(dataset),
                        f"{frontend}{ext}")


def mlaad_pool_paths(frontend, scores_root=None, system="linear_head"):
    """The score files that make up the MLAAD benchmark column, in pool order.

    MLAAD is the one benchmark column that is not a single score file. It is a
    two-class pool assembled from two single-class corpora:

        MLAAD v10   456,000 spoof rows      the deepfakes
        M-AILABS    584,006 bonafide rows   the real speech they were built from

    1,040,006 rows per model. M-AILABS is not a benchmark column of its own --
    it is single-class, so it has no EER -- which is why `DATASET_DIRS` carries
    the note that it is scored separately and merged in afterwards.

    They are kept as two files and pooled here, at read time. The retired legacy
    tree wrote them pre-concatenated into one, which is the worse arrangement
    and not for aesthetic reasons: a file that silently contains two corpora
    cannot be re-pooled at a different ratio, cannot be checked for either
    corpus's row count independently, and gives a misleading answer to "how many
    MLAAD utterances are there". Keeping them apart makes the pooling an
    explicit, inspectable step -- this function.

    Returns a list of paths. Callers concatenate them.
    """
    # The tsv twin is what must be read: ~8.6% of MLAAD utt_ids contain spaces
    # (vendor directories like "Cartesia.ai (Sonic-3)"). M-AILABS ids never do,
    # so it has no tsv twin and is read from the canonical .txt.
    return [
        score_path(system, "Multilingual", frontend,
                   scores_root=scores_root, ext=".tsv"),
        score_path(system, "MAILABS", frontend, scores_root=scores_root),
    ]


def available_frontends(system, dataset, scores_root=None, ext=".txt"):
    """Every frontend that has a score file for one (system, dataset), sorted.

    Analysis scripts need to enumerate "which models were scored on this set",
    and each had done it by globbing a directory and string-stripping a prefix
    off the filenames. That works until a model name contains the separator, so
    the enumeration is done here instead, by inverting the layout's own naming
    rule rather than by guessing at it. The frontend is a whole path component,
    so there is nothing to invert: the file stems are the answer.

    Returns [] if the directory does not exist: "nothing was scored" is a normal
    state to report, not an error.
    """
    directory = score_dir(system, dataset, scores_root)
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(directory)
        if name.endswith(ext)
        and (system in NON_SSL_SYSTEMS) == (os.path.splitext(name)[0] in NON_SSL_SYSTEMS)
    )


def score_dir(system, dataset, scores_root=None):
    """Directory holding the score files for one (system, dataset).

    The non-SSL systems share a directory, so this returns a directory
    containing sibling systems' files as well; it is the containing directory,
    not an exclusive one.
    """
    return os.path.dirname(score_path(system, dataset, "x", scores_root))


def main(argv=None):
    """Print the score-file path for one (system, dataset, frontend).

    Lets the shell scripts place their output without duplicating the rule:

        OUTPUT_FILE=$(python -m spoof_superb.core.scorepath \
            --system linear_head --dataset wild --frontend xls_r_300m)
    """
    import argparse
    ap = argparse.ArgumentParser(prog="python -m spoof_superb.core.scorepath")
    ap.add_argument("--system", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frontend", default="none")
    ap.add_argument("--scores_root", default=None)
    args = ap.parse_args(argv)
    print(score_path(args.system, args.dataset, args.frontend,
                     scores_root=args.scores_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
