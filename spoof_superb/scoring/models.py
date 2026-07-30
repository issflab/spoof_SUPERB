"""Which SSL upstreams the paper reports, and where that list comes from.

24 trained linear heads exist on disk. Table 5 reports 21. The other three --
audio_albert_960hr, byol_a_2048, modified_cpc -- were trained and scored but
never made the paper, so scoring them on every dataset spends 36 of 288 tasks on
columns nobody reads, including two of the most expensive.

The roster is READ FROM `tests/baseline_table5.json`, not written down here.
That file already maps each Table 5 row to its model slug and is what the
zero-tolerance regression gate compares against, which makes it the one place
that cannot silently disagree with the paper. A hand-maintained list of 21 names
would be a second copy of the same fact, free to drift -- the exact duplication
this reorganisation has been removing everywhere else.

Consequences of that choice, both deliberate:

  * add a row to Table 5 and it becomes scoreable with no code change
  * lose the baseline file and this raises, rather than quietly falling back to
    "score everything" and burning a day of GPU time

`paper_only` is a default, never a restriction: `--models` names any upstream
explicitly, in the paper or not.
"""

import functools
import json
import os

from spoof_superb import REPO_ROOT

__all__ = ["paper_models", "is_paper_model", "non_paper_models", "TABLE5_BASELINE"]

TABLE5_BASELINE = os.path.join(str(REPO_ROOT), "tests", "baseline_table5.json")


@functools.lru_cache(maxsize=None)
def paper_models(path=None):
    """frozenset of the SSL slugs Table 5 reports.

    Raises rather than guessing: a missing or malformed baseline must not
    silently widen a sweep back to every model on disk.
    """
    path = path or TABLE5_BASELINE
    try:
        with open(path) as f:
            results = json.load(f)["results"]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"cannot determine the paper's model roster: {path} is missing. "
            f"Pass explicit --models, or restore the Table 5 baseline.")
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable Table 5 baseline: {exc}")

    slugs = {row["slug"] for row in results.values() if row.get("slug")}
    if not slugs:
        raise ValueError(f"{path} declares no model slugs")
    return frozenset(slugs)


def is_paper_model(ssl, path=None):
    return ssl in paper_models(path)


def non_paper_models(available, path=None):
    """The slugs in ``available`` that Table 5 does not report."""
    return sorted(set(available) - paper_models(path))
