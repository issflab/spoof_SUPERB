"""Analysis views: the groupings the paper's analyses are computed over.

Two views, matching the two analyses beyond the main table:

    acoustic_degradation   Section 4.4.2, tab:acoustic_degradation
    tts_systems            Sections 4.4.3 and 3.2.3

They are built by their analysis entry points rather than left to a separate
step, so an analysis and the grouping it reports over cannot disagree.

Two shapes, because the two analyses group differently
------------------------------------------------------
A **partition** view splits ONE dataset by something in its utt_ids. Every row
lands in exactly one group and the groups are discovered from the data:

    tts_systems      raw/linear_head/mlaad_v10  ->  91 systems under AR / NAR /
                     closed_undisclosed, plus a pooled M-AILABS bonafide set

A **composite** view POOLS partitions of SEVERAL datasets into each group, and
the groups are named in advance because the paper names them. This is what the
degradation study needs and what a partition cannot express:

    acoustic_degradation   Baseline = ASV19 LA eval + ASV21 LA:C1
                                    + ASV21 DF:C1 + ASV5:C00

Both produce the same thing -- {group: (utts, labels, scores)} -- so everything
downstream, materialising included, is written once.

Layout (P11 D1-D7)
------------------
    {scores_root}/views/{view}/{group}/[{subgroup}/]{frontend}.txt
    {scores_root}/views/{view}/_bonafide/{frontend}.txt
    {scores_root}/views/{view}/_manifest.json

`views/` is a sibling of `raw/`, and the frontend is always the LAST component
so `views/*/*/xls_r_300m.txt` finds one model everywhere. Reserved names take an
underscore so no real group can collide with them.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from spoof_superb.analysis import metadata_csv
from spoof_superb.analysis.conditions import (CLEAN, asvld_family,
                                              condition_of)
from spoof_superb.core.scorefile import read_scored
from spoof_superb.core.scorepath import mlaad_pool_paths, score_path

__all__ = ["VIEW_SPECS", "PartitionView", "CompositeView", "Part",
           "load_view", "view_dir"]


# --- spec types -------------------------------------------------------------

@dataclass(frozen=True)
class Part:
    """One dataset's contribution to a composite group.

    Exactly one of `conditions` / `exclude` may be set. `exclude` exists because
    the paper defines most degraded partitions as "everything but the clean
    one" -- ASV21 DF C2-C9 is the eight codecs that are not `nocodec` -- and
    listing them positively would silently drop a condition if the corpus ever
    gained one.
    """
    dataset: str
    conditions: Optional[tuple] = None
    exclude: Optional[tuple] = None
    #: ASVLD is selected by degradation FAMILY rather than by condition, since
    #: the paper says "ASVLD (noise)", not fifteen condition names.
    family: Optional[str] = None

    def __post_init__(self):
        set_count = sum(x is not None for x in
                        (self.conditions, self.exclude, self.family))
        if set_count > 1:
            raise ValueError(f"{self.dataset}: give at most one of conditions, "
                             f"exclude, family")

    def selects(self, condition):
        if self.family is not None:
            return condition is not None and asvld_family(condition) == self.family
        if self.conditions is not None:
            return condition in self.conditions
        if self.exclude is not None:
            return condition not in self.exclude
        return True


@dataclass(frozen=True)
class CompositeView:
    """Groups named in advance, each pooling partitions of several datasets."""
    name: str
    groups: dict
    doc: str
    reference: Optional[str] = None      # group the others are compared against


@dataclass(frozen=True)
class PartitionView:
    """One dataset split by a key derived from its utt_ids."""
    name: str
    dataset: str
    key: Callable[[str], Optional[tuple]]
    doc: str
    bonafide_dataset: Optional[str] = None
    ext: str = ".txt"


# --- the acoustic degradation composition -----------------------------------
#
# Straight from tab:acoustic_degradation. Bold entries in that table -- the ones
# "retained unchanged from the baseline" -- are the same Part objects here, so
# the retained partitions cannot drift between conditions.

_LA_C1 = Part("asvspoof2021_LA", conditions=(CLEAN["asvspoof2021_LA"],))
_DF_C1 = Part("asvspoof2021_DF", conditions=(CLEAN["asvspoof2021_DF"],))
_A5_C00 = Part("asvspoof5", conditions=(CLEAN["asvspoof5"],))
_ASV19 = Part("eval_2019")

ACOUSTIC_DEGRADATION = CompositeView(
    name="acoustic_degradation",
    reference="Baseline",
    doc="Section 4.4.2 / tab:acoustic_degradation. Six conditions, one clean "
        "reference and five degraded, each composed from partitions of four "
        "corpora. Every degraded condition keeps the corpora it does not "
        "degrade, so its EER moves only for the degradation under study.",
    groups={
        # Reference: clean partitions of all four corpora.
        "Baseline": (_ASV19, _LA_C1, _DF_C1, _A5_C00),

        # ASVLD clean -> recompressed; DF C1 -> C2-C9; ASV5 C00 -> C01-C10.
        # ASV21 LA:C1 is retained as the clean complement.
        "Codec_Compression": (
            Part("asvspoofLD", family="Codec_Compression"),
            Part("asvspoof2021_DF", exclude=(CLEAN["asvspoof2021_DF"],)),
            Part("asvspoof5", exclude=(CLEAN["asvspoof5"], "C11")),
            _LA_C1,
        ),

        # ASVLD clean -> resampled. The other three retained.
        "Bandwidth": (
            Part("asvspoofLD", family="Bandwidth"),
            _LA_C1, _DF_C1, _A5_C00,
        ),

        # ASVLD clean -> noise-augmented. The other three retained.
        "Additive_Noise": (
            Part("asvspoofLD", family="Additive_Noise"),
            _LA_C1, _DF_C1, _A5_C00,
        ),

        # ASVLD clean -> reverberated. The other three retained.
        "Reverberation": (
            Part("asvspoofLD", family="Reverberation"),
            _LA_C1, _DF_C1, _A5_C00,
        ),

        # LA C1 -> C2-C7; ASV5 C00 -> C11. ASV19 and DF:C1 retained.
        # Note this condition does NOT use ASVLD's own lpf_* set.
        "Channel_Distortions": (
            Part("asvspoof2021_LA", exclude=(CLEAN["asvspoof2021_LA"],)),
            Part("asvspoof5", conditions=("C11",)),
            _DF_C1, _ASV19,
        ),
    },
)


# --- the TTS system partition -----------------------------------------------

def _mlaad_parts(utt):
    """MLAAD ids are `MLAAD/fake/{language}/{raw_dir}/{file}.wav`."""
    parts = utt.split("/")
    if len(parts) != 5:
        raise ValueError(f"MLAAD utt_id has {len(parts)} segments, expected 5: {utt!r}")
    return parts[2], parts[3]


class _MlaadTtsKey:
    """raw directory -> (generation mode, canonical system).

    Section 4.4.3: Dual-AR is merged into FishTTS and three non-TTS entries are
    excluded (griffin_lim, a phase-reconstruction vocoder; RVC, a voice
    conversion system; Voxtral, an audio-understanding model), removing 25,000
    utterances and leaving 431,000 across 91 systems. Both facts live in
    `mlaad_v10_dir_to_system.csv`; this reads them rather than restating them.

    A class rather than a closure so the CSVs are read at first use, not on
    import -- importing this module must not require a corpus to be mounted.
    """

    MODE = {"AR": "AR", "NAR": "NAR", "unknown": "closed_undisclosed"}

    def __init__(self):
        self._system = None
        self._mode = None

    def _load(self):
        import pandas as pd
        dir_map = pd.read_csv(metadata_csv("mlaad_v10_dir_to_system.csv"))
        self._system = dict(zip(dir_map["raw_dir"], dir_map["canonical_system"]))
        arch = pd.read_csv(metadata_csv("mlaad_v10_tts_architecture_groups.csv"))
        self._mode = {r.tts_system: self.MODE[r.ar_nar]
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
            return None
        return (self._mode[system], system)


TTS_SYSTEMS = PartitionView(
    name="tts_systems",
    dataset="Multilingual",
    bonafide_dataset="MAILABS",
    key=_MlaadTtsKey(),
    ext=".tsv",
    doc="Sections 4.4.3 and 3.2.3. MLAAD v10 spoof scores per TTS system, "
        "under the system's generation mode. Every system is scored against "
        "the same pooled M-AILABS bonafide reference (584,006 utterances), "
        "which is the reason this analysis is restricted to MLAAD: it measures "
        "the detectability of the synthesis system rather than the difficulty "
        "of its source corpus. Architecture group and vocoder family are "
        "functions of the system, so they are grouped up at analysis time "
        "rather than fixed into the tree.",
)


VIEW_SPECS = {
    "acoustic_degradation": ACOUSTIC_DEGRADATION,
    "tts_systems": TTS_SYSTEMS,
}


def view_dir(view, scores_root):
    """Root directory of one materialised view."""
    return os.path.join(scores_root, "views", view)


# --- loading ----------------------------------------------------------------

def _read_dataset(dataset, frontend, scores_root, ext=".txt"):
    path = score_path("linear_head", dataset, frontend,
                      scores_root=scores_root, ext=ext)
    return read_scored(path)


def _load_composite(spec, frontend, scores_root):
    groups = {}
    for group, parts in spec.groups.items():
        utts, labels, scores = [], [], []
        for part in parts:
            u, l, s = _read_dataset(part.dataset, frontend, scores_root)
            cond = condition_of(part.dataset)
            if cond is None:
                keep = np.ones(u.size, dtype=bool)
            else:
                keep = np.fromiter(
                    (part.selects(cond(x)) for x in u.tolist()),
                    dtype=bool, count=u.size)
            utts.append(u[keep])
            labels.append(l[keep])
            scores.append(s[keep])
        groups[(group,)] = (np.concatenate(utts), np.concatenate(labels),
                            np.concatenate(scores))
    return groups, None


def _load_partition(spec, frontend, scores_root):
    if spec.bonafide_dataset:
        pool = mlaad_pool_paths(frontend, scores_root=scores_root)
        utts, labels, scores = read_scored(pool[0])
        bonafide = read_scored(pool[1])
    else:
        utts, labels, scores = _read_dataset(spec.dataset, frontend, scores_root,
                                             spec.ext)
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


def load_view(spec, frontend, scores_root=None):
    """Group one model's raw scores as the view defines.

    Returns (groups, bonafide):

        groups   {(level, ...): (utts, labels, scores)}
        bonafide (utts, labels, scores), or None when the groups carry their own

    Composite groups are pools and may legitimately share no rows with each
    other; partition groups are disjoint and together reproduce their source.
    """
    if isinstance(spec, CompositeView):
        return _load_composite(spec, frontend, scores_root)
    return _load_partition(spec, frontend, scores_root)
