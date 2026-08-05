"""Analysis: tables and figures computed from score files.

Locating the analysis metadata
------------------------------
Two CSVs describe the MLAAD v10 TTS systems rather than any model's scores:

    mlaad_v10_dir_to_system.csv              corpus directory -> canonical system
    mlaad_v10_tts_architecture_groups.csv    canonical system -> AR/NAR, vocoder

Both are curated facts about the corpus. They are the same kind of thing, but
they were stored in different places: the first in the repo beside the modules
that read it, the second inside the score tree. That second choice is wrong and
the layout port is what exposed it -- point the tools at a different score tree
and the architecture groups silently disappear, even though nothing about TTS
architectures depends on which tree you are reading.

`metadata_csv` resolves either name: the repo copy wins, and the configured
score root is searched as a fallback so existing setups keep working. When the
file is finally moved into the repo, the fallback becomes dead and can go.
"""

import os

__all__ = ["metadata_csv", "raw_score_path"]

_HERE = os.path.dirname(os.path.abspath(__file__))


def raw_score_path(dataset, frontend, linear_head_dir=None, scores_root=None,
):
    """The raw linear_head score file for one (dataset, frontend).

    Several analysis tools take a `--linear_head_dir` and build the old flat
    filename inside it. That is kept working -- an explicit directory always
    wins, so existing invocations are unaffected -- but when it is absent the
    path is resolved through the layout instead, which is what lets the same
    tool read a v3 tree.

    Two ways to say the same thing, with the explicit one taking precedence, is
    the smallest change that ports these tools without breaking their callers.
    """
    if linear_head_dir:
        return os.path.join(linear_head_dir,
                            f"linear_head_{dataset}_{frontend}.txt")
    from spoof_superb.core.scorepath import score_path
    return score_path("linear_head", dataset, frontend,
                      scores_root=scores_root)


def metadata_csv(name, scores_root=None):
    """Absolute path to an analysis metadata CSV.

    Searched in the repo first, then the score tree. Raises with both locations
    named rather than returning a path that does not exist, because the failure
    it prevents -- a tool reporting "no such file" for a path the user never
    chose -- is exactly what the split storage caused.
    """
    repo_copy = os.path.join(_HERE, name)
    if os.path.isfile(repo_copy):
        return repo_copy

    if scores_root is None:
        from spoof_superb.config import cfg
        scores_root = cfg.scores_root
    in_tree = os.path.join(scores_root, name)
    if os.path.isfile(in_tree):
        return in_tree

    raise FileNotFoundError(
        f"{name} not found. Looked in the repo ({repo_copy}) and in the score "
        f"tree ({in_tree}). This file is corpus metadata, not score data; if "
        f"it lives in a different tree, pass its path explicitly.")
