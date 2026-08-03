"""Analysis views: ways of looking at the raw score files, declared once.

A view groups the rows of a raw score file by something recoverable from the
utt_id -- the TTS system that produced an utterance, its language, the acoustic
condition applied to it. The legacy tree materialised eight such groupings as
separate directory trees totalling 23.1 GB against 7.5 GB of raw, and two of
those eight were the same view built twice, with the documentation already
warning which copy to trust.

The alternative is not to write less to disk. It is to make the GROUPING the
artefact rather than the copy: a view is defined by its source and a key
function, so it can be recomputed, checked against raw, and materialised when a
human wants to browse it -- but no analysis has to wait for that to happen.

Shape (approved as P11 D1-D7, D9)
---------------------------------
    {scores_root}/views/{view}/{group}/[{subgroup}/]{frontend}.txt
    {scores_root}/views/{view}/_bonafide/{frontend}.txt
    {scores_root}/views/{view}/_manifest.json

`views/` is a sibling of `raw/`, so `raw/*/*/model.txt` stays exact. The
frontend is always the LAST component, so `views/*/*/xls_r_300m.txt` finds one
model everywhere -- the property the legacy degradation tree lost by naming its
files `APC.txt` under four conditions and `linear_head_resamp_apc.txt` under
the fifth, which is the entire reason `compute_eer_matrix` carries three
hand-maintained stem dictionaries.

`_bonafide` takes an underscore because the legacy `bonafide/` was
indistinguishable from a TTS system of that name and sorted into the middle of
the real ones.

What is NOT here
----------------
No `acoustic_degradation` view reproducing the legacy Section 5.2 population.
That tree mixes ASVspoof2019 LA, ASVspoof2021 DF and ASVspoof5 utterances, and
the v3 tree holds those three corpora CLEAN only -- none of their utt_ids carry
a condition suffix, so the degraded audio was never scored into it. Rebuilding
that population is a scoring job. `asvld_conditions` is a different and
self-consistent measurement over one corpus, and is named so it cannot be
mistaken for the published one.

No normalised-score views: they are not used (P11 D8, dropped).
"""

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from spoof_superb.analysis import metadata_csv
from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import mlaad_pool_paths, score_path

__all__ = ["VIEW_SPECS", "ViewSpec", "load_view", "view_dir"]


@dataclass(frozen=True)
class ViewSpec:
    """How one view is derived from raw score files.

    `key` maps a utt_id to the path components it belongs under, or to None to
    drop the row. Returning a tuple is what allows a view to be one or two
    levels deep (P11 D3) without the depth being special-cased anywhere.
    """
    name: str
    dataset: str
    key: Callable[[str], Optional[tuple]]
    doc: str
    #: Dataset supplying a shared bonafide reference, when the source corpus is
    #: spoof-only. MLAAD's bonafide counterpart is M-AILABS, pooled once per
    #: view rather than per group -- a single shared reference is the whole
    #: point of restricting the analysis to MLAAD.
    bonafide_dataset: Optional[str] = None
    #: Read the tab-separated twin. Required when utt_ids contain spaces.
    ext: str = ".txt"


# --- key functions ----------------------------------------------------------

#: ASVLD condition suffix -> the family it belongs to. The families are the ones
#: the paper groups by; the condition is the leaf, because the condition is
#: recoverable and the family is a function of it, never the other way round.
_ASVLD_FAMILY = [
    (re.compile(r"^(babble|cafe|street|volvo|white)_\d+$"), "Additive_Noise"),
    (re.compile(r"^RT_\d+_\d+$"),                           "Reverberation"),
    (re.compile(r"^resample_\d+$"),                         "Resampling"),
    (re.compile(r"^recompression_\d+k$"),                   "Codec_Compression"),
    (re.compile(r"^lpf_\d+$"),                              "Channel_Distortions"),
]

_ASVLD_ID = re.compile(r"^LA_E_\d+_(?P<cond>.+)$")


def _asvld_key(utt):
    """`LA_E_2834763_babble_0` -> ('Additive_Noise', 'babble_0')."""
    m = _ASVLD_ID.match(utt)
    if m is None:
        return None
    cond = m.group("cond")
    for pattern, family in _ASVLD_FAMILY:
        if pattern.match(cond):
            return (family, cond)
    raise KeyError(
        f"utt_id {utt!r} has condition {cond!r}, which matches no ASVLD family. "
        f"Add it to views._ASVLD_FAMILY -- silently dropping a condition would "
        f"shrink a benchmark column without saying so.")


def _mlaad_parts(utt):
    """MLAAD ids are `MLAAD/fake/{language}/{raw_dir}/{file}.wav`."""
    parts = utt.split("/")
    if len(parts) != 5:
        raise ValueError(f"MLAAD utt_id has {len(parts)} segments, expected 5: {utt!r}")
    return parts[2], parts[3]          # language, raw_dir


def _mlaad_language_key(utt):
    language, _raw_dir = _mlaad_parts(utt)
    return (language,)


class _MlaadTtsKey:
    """raw directory -> (AR|NAR|closed_undisclosed, canonical system).

    A class rather than a closure so the two CSVs it needs are read once, at
    first use, instead of on import -- importing this module must not require a
    score tree or a corpus to be mounted.
    """

    BUCKET = {"AR": "AR", "NAR": "NAR", "unknown": "closed_undisclosed"}

    def __init__(self):
        self._system = None
        self._bucket = None

    def _load(self):
        import pandas as pd
        dir_map = pd.read_csv(metadata_csv("mlaad_v10_dir_to_system.csv"))
        self._system = dict(zip(dir_map["raw_dir"], dir_map["canonical_system"]))
        arch = pd.read_csv(metadata_csv("mlaad_v10_tts_architecture_groups.csv"))
        self._bucket = {r.tts_system: self.BUCKET[r.ar_nar]
                        for r in arch.itertuples(index=False)}

    def __call__(self, utt):
        if self._system is None:
            self._load()
        _language, raw_dir = _mlaad_parts(utt)
        try:
            system = self._system[raw_dir]
        except KeyError:
            raise KeyError(
                f"MLAAD directory {raw_dir!r} is in no dir map entry. Rebuild "
                f"the map with build_mlaad_dir_map rather than dropping it.") from None
        if system == "EXCLUDED":
            return None            # voice conversion, vocoders: not TTS systems
        return (self._bucket[system], system)


# --- the registry -----------------------------------------------------------

VIEW_SPECS = {
    "mlaad_tts": ViewSpec(
        name="mlaad_tts",
        dataset="Multilingual",
        bonafide_dataset="MAILABS",
        key=_MlaadTtsKey(),
        ext=".tsv",
        doc="MLAAD v10 spoof scores by TTS system, bucketed by generation mode. "
            "Every group is scored against the same pooled M-AILABS bonafide "
            "set, which is what makes systems comparable to each other rather "
            "than to their own source difficulty.",
    ),
    "mlaad_language": ViewSpec(
        name="mlaad_language",
        dataset="Multilingual",
        bonafide_dataset="MAILABS",
        key=_mlaad_language_key,
        ext=".tsv",
        doc="MLAAD v10 spoof scores by language, against the same pooled "
            "M-AILABS bonafide set.",
    ),
    "asvld_conditions": ViewSpec(
        name="asvld_conditions",
        dataset="asvspoofLD",
        key=_asvld_key,
        doc="ASVspoof-LD by acoustic condition, grouped into the five families "
            "the paper reports. NOT the legacy scores_by_acoustic_degradation "
            "population, which mixes three corpora and is mostly untagged; see "
            "P11. Each condition carries its own bonafide and spoof rows, so "
            "there is no shared reference pool.",
    ),
}


def view_dir(view, scores_root, ):
    """Root directory of one materialised view."""
    return os.path.join(scores_root, "views", view)


def load_view(spec, frontend, scores_root=None, layout=None):
    """Group one model's raw scores as the view defines.

    Returns (groups, bonafide):

        groups   {(level, ...): (utts, labels, scores)}
        bonafide (utts, labels, scores), or None when the source corpus carries
                 its own bonafide rows

    This is the whole of the view. Materialising writes it out; analysis can
    equally consume it here, which is P11 D9 -- a view is a query, and no number
    should wait on a directory being built.
    """
    if spec.bonafide_dataset:
        pool = mlaad_pool_paths(frontend, scores_root=scores_root, layout=layout)
        spoof_path, bona_path = pool[0], pool[1]
        utts, labels, scores = read_scored(spoof_path)
        bonafide = read_scored(bona_path)
    else:
        path = score_path("linear_head", spec.dataset, frontend,
                          scores_root=scores_root, layout=layout, ext=spec.ext)
        utts, labels, scores = read_scored(path)
        bonafide = None

    buckets = {}
    for i, utt in enumerate(utts.tolist()):
        key = spec.key(utt)
        if key is None:
            continue
        buckets.setdefault(key, []).append(i)

    groups = {}
    for key, idx in buckets.items():
        idx = np.asarray(idx, dtype=np.int64)
        groups[key] = (utts[idx], labels[idx], scores[idx])
    return groups, bonafide
