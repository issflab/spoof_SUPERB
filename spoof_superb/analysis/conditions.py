"""Which acoustic condition an evaluation utterance belongs to.

The paper's acoustic degradation study (Section 4.4.2, `tab:acoustic_degradation`)
composes each condition from partitions of four corpora. Those partitions are
identified in three different ways, and this module is where that difference is
absorbed so nothing downstream has to know about it:

    ASVspoof 2019 LA eval   the whole set is the clean reference; no condition
                            column exists and none is needed
    ASVspoof 2021 LA        the CODEC field of trial_metadata.txt.
                            none -> C1, and six codecs -> C2-C7
    ASVspoof 2021 DF        the CODEC field of trial_metadata.txt.
                            nocodec -> C1, and eight codecs -> C2-C9
    ASVspoof 5              an explicit condition column.
                            '-' -> C00, then C01-C11
    ASVspoof-LD             the utt_id's own suffix, e.g. LA_E_2834763_babble_0

Verified against the v3 tree -- every scored utterance resolves, none falls
outside its protocol:

    asvspoof2021_la    181,566 rows   7 conditions x 25,938
    asvspoof2021_df    611,829 rows   9 conditions x 67,981
    asvspoof5          680,774 rows   12 conditions ('-' = 171,602)
    asvspoof_ld      2,065,873 rows   29 conditions x 71,237

The C-numbering is the corpora's own, not ours. Only C1/C00 -- the clean
partition -- is load-bearing for the composition, because every other condition
enters as "all the rest"; the mapping is still spelled out so a reader can check
a row against the corpus documentation.
"""

import functools
import re

__all__ = ["condition_of", "CLEAN", "ASV21_LA_CODES", "ASV21_DF_CODES",
           "asvld_family"]

#: The clean partition of each corpus, as named by that corpus's own protocol.
#: This is the only assignment the composition depends on.
CLEAN = {
    "eval_2019":       None,        # whole set is clean; no partitioning
    "asvspoof2021_LA": "none",      # C1
    "asvspoof2021_DF": "nocodec",   # C1
    "asvspoof5":       "-",         # C00
}

#: ASVspoof 2021 LA: protocol codec token -> the corpus's condition code.
ASV21_LA_CODES = {
    "none": "C1", "alaw": "C2", "ulaw": "C3", "g722": "C4",
    "gsm": "C5", "opus": "C6", "pstn": "C7",
}

#: ASVspoof 2021 DF: protocol codec token -> the corpus's condition code.
ASV21_DF_CODES = {
    "nocodec": "C1", "low_mp3": "C2", "high_mp3": "C3", "low_m4a": "C4",
    "high_m4a": "C5", "low_ogg": "C6", "high_ogg": "C7", "mp3m4a": "C8",
    "oggm4a": "C9",
}

#: (utt_col, condition_col) in each protocol, zero-based over whitespace fields.
_PROTOCOL_COLS = {
    "asvspoof2021_LA": (1, 2),
    "asvspoof2021_DF": (1, 2),
    "asvspoof5":       (1, 3),
}

#: ASVLD condition suffix -> the family the paper groups it under. ASVLD's own
#: `lpf_*` is deliberately absent from the paper's Channel Distortions, which is
#: built from ASV21 LA C2-C7 and ASV5 C11 instead; it is still classified here
#: so `asvld_conditions` can report it.
_ASVLD_FAMILY = [
    (re.compile(r"^(babble|cafe|street|volvo|white)_\d+$"), "Additive_Noise"),
    (re.compile(r"^RT_\d+_\d+$"),                           "Reverberation"),
    (re.compile(r"^resample_\d+$"),                         "Bandwidth"),
    (re.compile(r"^recompression_\d+k$"),                   "Codec_Compression"),
    (re.compile(r"^lpf_\d+$"),                              "Channel_Distortions"),
]

_ASVLD_ID = re.compile(r"^LA_E_\d+_(?P<cond>.+)$")


def asvld_family(condition):
    """Which degradation family an ASVLD condition suffix belongs to."""
    for pattern, family in _ASVLD_FAMILY:
        if pattern.match(condition):
            return family
    raise KeyError(
        f"ASVLD condition {condition!r} matches no family. Add it to "
        f"conditions._ASVLD_FAMILY -- silently dropping a condition would "
        f"shrink an evaluation set without saying so.")


@functools.lru_cache(maxsize=None)
def _protocol_conditions(dataset):
    """utt_id -> condition token, read once per dataset per process."""
    from spoof_superb.scoring.datasets import PROTOCOL_SPECS
    utt_col, cond_col = _PROTOCOL_COLS[dataset]
    path = PROTOCOL_SPECS[dataset]["protocol"]
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) > max(utt_col, cond_col):
                out[parts[utt_col]] = parts[cond_col]
    if not out:
        raise ValueError(f"{path}: no rows parsed for {dataset}")
    return out


def condition_of(dataset):
    """A function mapping one utt_id to its condition in `dataset`.

    Returns None for ASVspoof 2019 LA eval, whose whole set is the clean
    reference -- there is nothing to partition, and inventing a single-valued
    condition column would only imply otherwise.
    """
    if dataset == "eval_2019":
        return None

    if dataset == "asvspoofLD":
        def _asvld(utt):
            m = _ASVLD_ID.match(utt)
            if m is None:
                return None          # a clean ASVLD id carries no suffix
            return m.group("cond")
        return _asvld

    table = _protocol_conditions(dataset)

    def _lookup(utt):
        try:
            return table[utt]
        except KeyError:
            raise KeyError(
                f"{utt!r} is not in the {dataset} protocol, so its condition "
                f"is unknown. A scored utterance outside its own protocol means "
                f"the score file and the protocol disagree about the trial "
                f"list.") from None
    return _lookup
