"""Dataset registry: how a trial list, its labels, and its audio paths are found.

Every benchmark set differs in exactly three ways, and nothing else:

  * where the trial list comes from      (reference score file / protocol / walk)
  * where the ground-truth key comes from (a column / a constant / a CSV)
  * how a utt_id maps to a file on disk   (the ``resolve`` callables below)

Isolating those three makes one scoring driver sufficient for all of them.

Corpus roots come from spoof_superb.config; nothing here is hardcoded.
"""

import csv
import os
from functools import lru_cache

from spoof_superb.config import cfg

DATA = cfg.data_root
SCORES_ROOT = cfg.scores_root
REFERENCE_DIR = cfg.reference_dir
DEFAULT_REFERENCE_SSL = cfg.reference_ssl

CROP = 64600  # ~4 s at 16 kHz, as in data/datasets_ssl.py

ASVLD_ROOT = os.path.join(DATA, "ASVSpoofLaunderedDatabase", "ASVspoofLD")
ASVLD_CONDITIONS = ["Noise_Addition", "Reverberation", "Resampling",
                    "Recompression", "Filtering"]
# note the upstream misspelling "Launered"
ASVLD_PROTOCOL_TEMPLATE = "ASVspoofLauneredDatabase_{condition}.txt"

DFEVAL_ROOT = os.path.join(DATA, "Deepfake_Eval_2024")
DFEVAL_AUDIO = os.path.join(DFEVAL_ROOT, "audio-data")
DFEVAL_SEGMENTED = os.path.join(DFEVAL_ROOT, "segmented")
FF_NFS_PREFIX = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/"
FF_LOCAL_ROOT = os.path.join(DATA, "famousfigures")

MLAAD_ROOT = os.path.join(DATA, "MLAAD")
MAILABS_ROOT = os.path.join(DATA, "MAILabs")
SPOOFCELEB_PROTOCOL = os.path.join(DATA, "SpoofCeleb/metadata/evaluation.csv")
SPOOFCELEB_AUDIO = os.path.join(DATA, "SpoofCeleb/flac/evaluation")


# ===========================================================================
# utt_id -> audio path
# ===========================================================================

@lru_cache(maxsize=1)
def _asvld_condition_index():
    """utt_id -> condition, parsed from the 5 ASVLD protocols.

    Needed because ASVLD audio lives at {root}/{condition}/flac/{utt}.flac but
    the pooled reference score file carries no condition column.
    """
    index = {}
    proto_dir = os.path.join(ASVLD_ROOT, "protocols")
    for cond in ASVLD_CONDITIONS:
        path = os.path.join(proto_dir, ASVLD_PROTOCOL_TEMPLATE.format(condition=cond))
        if not os.path.isfile(path):
            print(f"  [WARN] ASVLD protocol missing: {path}")
            continue
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    index[parts[1]] = cond
    print(f"  ASVLD condition index: {len(index)} utt_ids")
    return index


@lru_cache(maxsize=1)
def _dfeval_stem_index():
    """basename-without-extension -> real path.

    DFEval24 score files write every id with a .wav extension, but on disk the
    files are .mp3 / .m4a / .mp4 / .wav. Match on the stem.
    """
    index = {}
    if not os.path.isdir(DFEVAL_AUDIO):
        print(f"  [WARN] DFEval24 audio dir missing: {DFEVAL_AUDIO}")
        return index
    for fn in os.listdir(DFEVAL_AUDIO):
        stem = os.path.splitext(fn)[0]
        index.setdefault(stem, os.path.join(DFEVAL_AUDIO, fn))
    print(f"  DFEval24 stem index: {len(index)} files")
    return index


def _r_asv19(utt):      # utt already carries '.flac'
    return os.path.join(DATA, "ASVSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac", utt)


def _r_asv21_la(utt):
    return os.path.join(DATA, "ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/flac", utt + ".flac")


def _r_asv21_df(utt):
    return os.path.join(DATA, "ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/flac", utt + ".flac")


def _r_asv5(utt):
    return os.path.join(DATA, "ASVSpoof5/No_Laundering_eval/flac", utt + ".flac")


def _r_itw(utt):
    return os.path.join(DATA, "ds_wild/release_in_the_wild", utt + ".wav")


def _r_dfeval(utt):
    return _dfeval_stem_index().get(os.path.splitext(utt)[0])


def _r_dfeval_segmented(utt):
    """Segment ids are filenames under segmented/wav/, written by us."""
    return os.path.join(DFEVAL_SEGMENTED, "wav", utt)


def _r_famous(utt):
    """Famous Figures: {root}/{Speaker}/{Source}/{name}.wav

    Two rewrites are needed, both verified against the full reference file:
      1. Reference ids are absolute paths under a stale NFS mount that does not
         exist on this host; the same tree is present under /data/Data.
      2. Bonafide rows carry the protocol's empty Source field as the literal
         directory '-', but on disk they live under 'Bonafide'. All 49,945
         '/-/' rows are key=bonafide and all 49,945 resolve after this remap;
         without it the dataset would score zero bonafide trials and its EER
         would be undefined.
    """
    if utt.startswith(FF_NFS_PREFIX):
        rel = utt[len(FF_NFS_PREFIX):]
    elif utt.startswith("/"):
        rel = os.path.relpath(utt, FF_LOCAL_ROOT)
    else:
        rel = utt

    parts = rel.split("/")
    if len(parts) >= 3 and parts[1] == "-":
        parts[1] = "Bonafide"
        rel = "/".join(parts)

    return os.path.join(FF_LOCAL_ROOT, rel)


def _r_spoofceleb(utt):
    return os.path.join(SPOOFCELEB_AUDIO, utt)


def _r_mlaad(utt):      # ids are relative to /data/Data and carry the extension
    return os.path.join(DATA, utt)


def _r_asvld(utt):
    cond = _asvld_condition_index().get(utt)
    if cond is None:
        return None
    return os.path.join(ASVLD_ROOT, cond, "flac", utt + ".flac")


def asvld_condition_resolver(audio_base_dir, condition):
    """Resolver for a single-condition ASVLD run (audio path is known upfront)."""
    flac_dir = os.path.join(audio_base_dir, condition, "flac")
    return lambda utt: os.path.join(flac_dir, utt + ".flac")


def relative_resolver(base_dir):
    """Resolver for id-is-a-relative-path datasets (MLAAD, M-AILABS, SpoofCeleb)."""
    return lambda utt: os.path.join(base_dir, utt)


# ===========================================================================
# Benchmark registry -- reference-score-file driven (the 10 published columns)
# ===========================================================================

# `ref` is resolved relative to REFERENCE_DIR; `ref_abs` is an absolute template
# for columns whose published source lives outside linear_head/. A list means the
# paper's column is the POOL of those files, and it is assembled in that order.
#
# Sources here must match analysis/recompute_table5_mlaad_v10.py, which is the
# authority for what Table 5 actually reports. Two columns are NOT the obvious
# linear_head/ file:
#   MLAAD  -> the v10 re-run (1,040,006 rows), not legacy linear_head_Multilingual
#             (307,998). Different corpus scale entirely.
#   ASVLD  -> linear_head_asvspoofLD (1,207,509: noise x10, reverb x3, resample x4)
#             POOLED WITH asvld_rerun/Recompression (427,422 = 71,237 x 6 bitrates),
#             folded in by commit 6bf39a0. Reading only the first file silently
#             reproduces a pre-6bf39a0 column.
# SpoofCeleb legacy vs the linear_head_SpoofCeleb re-run were verified to have
# identical utt sets and labels, so either supplies the same trials; the legacy
# path is kept.
DATASETS = {
    "eval_2019":          dict(ref="linear_head_eval_2019_{ssl}.txt",          resolve=_r_asv19),
    "asvspoof2021_LA":    dict(ref="linear_head_asvspoof2021_LA_{ssl}.txt",    resolve=_r_asv21_la),
    "asvspoof2021_DF":    dict(ref="linear_head_asvspoof2021_DF_{ssl}.txt",    resolve=_r_asv21_df),
    "asvspoof5":          dict(ref="linear_head_asvspoof5_{ssl}.txt",          resolve=_r_asv5),
    "deepfake_eval_2024": dict(ref="linear_head_deepfake_eval_2024_{ssl}.txt", resolve=_r_dfeval),
    "wild":               dict(ref="linear_head_wild_{ssl}.txt",               resolve=_r_itw),
    "Famous_Figures":     dict(ref="linear_head_Famous_Figures_{ssl}.txt",     resolve=_r_famous),
    "spoofceleb":         dict(ref="linear_head_spoofceleb_{ssl}.txt",         resolve=_r_spoofceleb),
    "Multilingual":       dict(ref_abs=[os.path.join(
                                  SCORES_ROOT, "linear_head_MLAAD_v10",
                                  "linear_head_MLAAD_v10_{ssl}.txt")],
                               resolve=_r_mlaad),
    # Scoreable but NOT a published benchmark column: Deepfake-Eval cut into
    # fixed 4 s segments, so a long recording contributes more than the single
    # window the model would otherwise see. Built by
    # data.prep.segment_deepfake_eval; no `ref`, so --source benchmark is
    # refused for it.
    "deepfake_eval_2024_segmented": dict(resolve=_r_dfeval_segmented),

    # Scoreable but NOT a published benchmark column: the bonafide counterpart
    # to MLAAD, scored separately and merged in afterwards. No `ref`, so
    # has_reference() is False and --source benchmark is refused for it.
    "MAILABS":            dict(resolve=_r_mlaad),
    "asvspoofLD":         dict(ref_abs=[os.path.join(REFERENCE_DIR,
                                            "linear_head_asvspoofLD_{ssl}.txt"),
                                        os.path.join(
                                  SCORES_ROOT, "asvld_rerun", "Recompression",
                                  "linear_head_Recompression_{ssl}.txt")],
                               resolve=_r_asvld),
}


# ---------------------------------------------------------------------------
# Native trial sources
#
# A dataset's identity and its definition must be one input, not two. Before
# this, `--dataset spoofceleb --source protocol_csv` took the id list from a
# hardcoded SpoofCeleb path while the output was filed under whatever
# --dataset said -- nothing stopped those disagreeing, and a mismatch was
# silent. Now the dataset decides its own trial source and parameters, and
# --source is only an override.
#
# `native` is the source used when --source is not given. Corpus- or
# protocol-driven wherever one exists; `benchmark` (read the published score
# file) only where the trial list survives nowhere else. See RP-7.
#: How each protocol file is laid out. `protocol` is the file, `built_by` names
#: the tool that writes it when the corpus ships none, and the rest are
#: arguments to trials_from_protocol.
PROTOCOL_SPECS = {
    # Written by data.prep.segment_deepfake_eval:
    #   segment_id, source_file, label, start_s, duration_s
    "deepfake_eval_2024_segmented": dict(
        protocol=os.path.join(DFEVAL_SEGMENTED, "protocol.txt"),
        built_by="python -m spoof_superb.data.prep.segment_deepfake_eval",
        delimiter="\t", header=True, utt_col=0, label_col=2),
    "spoofceleb": dict(
        protocol=SPOOFCELEB_PROTOCOL,
        delimiter=",", header=True, utt_col=0, label_col=2, bonafide_when="a00"),
    "wild": dict(
        protocol=os.path.join(DATA, "ds_wild/protocols/meta.csv"),
        delimiter=",", header=True, utt_col=0, label_col=2, strip_ext=True),
    "Multilingual": dict(
        protocol=os.path.join(MLAAD_ROOT, "combined_meta_all.txt"),
        built_by="python -m spoof_superb.analysis.create_combined_mlaad_meta_all",
        delimiter="|", header=True, utt_col=1, label_const="spoof", rel_to=DATA),
    "MAILABS": dict(
        protocol=os.path.join(MAILABS_ROOT, "protocol.txt"),
        built_by="python -m spoof_superb.data.prep.build_protocols mailabs",
        delimiter="\t", header=True, utt_col=0, label_col=1),
    "asvspoofLD": dict(
        protocol=os.path.join(ASVLD_ROOT, "protocol.txt"),
        built_by="python -m spoof_superb.data.prep.build_protocols asvld",
        delimiter="\t", header=True, utt_col=0, label_col=1),

    # --- the columns that were reference-driven until now (RP-7) -------------
    # These read the FULL protocol, not the published subset. Where the two
    # differ, the coverage line in verification reports it; see docs/08.
    "eval_2019": dict(
        protocol=os.path.join(
            DATA, "ASVSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.eval.trl.txt"),
        delimiter=None, header=False, utt_col=1, label_col=4, add_ext=".flac"),
    "asvspoof2021_LA": dict(
        protocol=os.path.join(
            DATA, "ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/trial_metadata.txt"),
        delimiter=None, header=False, utt_col=1, label_col=5),
    "asvspoof2021_DF": dict(
        protocol=os.path.join(
            DATA, "ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/trial_metadata.txt"),
        delimiter=None, header=False, utt_col=1, label_col=5),
    # Named .tsv but space-delimited.
    "asvspoof5": dict(
        protocol=os.path.join(DATA, "ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv"),
        delimiter=None, header=False, utt_col=1, label_col=8),
    # Ground Truth is Real/Fake; LABEL_ALIASES maps it.
    "deepfake_eval_2024": dict(
        protocol=os.path.join(DATA, "Deepfake_Eval_2024/audio-metadata-publish.csv"),
        delimiter=",", header=True, utt_col=0, label_col=2, strip_ext=True),
    # AudioPath is absolute under the local root; ids are made relative to it,
    # so they no longer carry the retired NFS mount prefix the published files
    # were written with.
    "Famous_Figures": dict(
        protocol=os.path.join(DATA, "famousfigures/protocol.txt"),
        delimiter="\t", header=True, utt_col=4, label_col=3,
        rel_to=os.path.join(DATA, "famousfigures")),
}

NATIVE_TRIALS = {
    "spoofceleb":         dict(source="protocol"),
    "wild":               dict(source="protocol"),
    "Multilingual":       dict(source="protocol"),
    "MAILABS":            dict(source="protocol"),
    "asvspoofLD":         dict(source="protocol"),
    "eval_2019":          dict(source="protocol"),
    "asvspoof2021_LA":    dict(source="protocol"),
    "asvspoof2021_DF":    dict(source="protocol"),
    "asvspoof5":          dict(source="protocol"),
    "deepfake_eval_2024":  dict(source="protocol"),
    "deepfake_eval_2024_segmented": dict(source="protocol"),
    "Famous_Figures":     dict(source="protocol"),
}

#: Scoreable but not a published benchmark column: the bonafide counterpart to
#: MLAAD, scored separately and merged in afterwards.
NON_BENCHMARK = {"MAILABS", "deepfake_eval_2024_segmented"}

#: Every dataset that can be scored, benchmark column or not.
SCOREABLE = list(DATASETS)


def native_source(dataset):
    """The trial source a dataset uses when --source is not given."""
    spec = NATIVE_TRIALS.get(dataset)
    return spec["source"] if spec else "benchmark"


def native_params(dataset):
    """Trial-source parameters for a dataset (protocol path, walk root, ...)."""
    spec = NATIVE_TRIALS.get(dataset, {})
    return {k: v for k, v in spec.items() if k != "source"}


def has_reference(dataset):
    """True if a published score file defines this dataset's trial list."""
    spec = DATASETS.get(dataset)
    return bool(spec) and ("ref" in spec or "ref_abs" in spec)


def reference_paths(dataset, reference_ssl):
    """Absolute reference score file(s) defining a dataset's trial list."""
    spec = DATASETS.get(dataset)
    if spec is None:
        raise KeyError(f"unknown dataset {dataset!r}; known: "
                       f"{', '.join(SCOREABLE)}")
    if "ref_abs" in spec:
        return [t.format(ssl=reference_ssl) for t in spec["ref_abs"]]
    if "ref" not in spec:
        raise KeyError(
            f"{dataset} has no published reference score file; it is scored "
            f"from its own protocol (source={native_source(dataset)!r}). "
            f"--source benchmark does not apply to it.")
    return [os.path.join(REFERENCE_DIR, spec["ref"].format(ssl=reference_ssl))]


# ===========================================================================
# Trial-list sources
# ===========================================================================

def trials_from_asvld_protocol(protocols_dir, condition):
    """([utt_id], {utt_id: key}) from a 6-column ASVLD protocol.

    Columns: speaker utt_id attack_id key condition variant
    """
    path = os.path.join(protocols_dir, ASVLD_PROTOCOL_TEMPLATE.format(condition=condition))
    if not os.path.isfile(path):
        return None, None

    utts, keys = [], {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"[WARN] skipping malformed protocol line: {line.strip()!r}")
                continue
            utts.append(parts[1])
            keys[parts[1]] = parts[3]
    return utts, keys


def trials_from_walk(root, data_base, label):
    """([utt_id], {utt_id: label}) for every wav under `root`.

    Used for MLAAD (all spoof) and M-AILABS (all bonafide). utt_ids are
    relative to `data_base`, reproducing the id format of the reference score
    files.
    """
    utts = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # Skip macOS AppleDouble sidecars ('._name.wav'): 176-245 byte
            # metadata stubs, not audio. M-AILABS carries 6 of them and they are
            # absent from the reference score files, so they are not utterances.
            if fn.startswith("._"):
                continue
            if fn.endswith(".wav"):
                utts.append(os.path.relpath(os.path.join(dirpath, fn), data_base))
    utts.sort()
    return utts, {u: label for u in utts}


#: Corpus label vocabularies -> the repo's two labels.
#:
#: Corpora disagree on spelling: In-the-Wild writes "bona-fide", ASVspoof
#: writes "bonafide", Deepfake-Eval writes "Real"/"Fake". Everything
#: downstream filters on == "bonafide" / == "spoof", so an unmapped label does
#: not raise -- the trial simply disappears from the EER, and the number is
#: quietly wrong. Normalising here, and refusing anything unrecognised, is what
#: keeps that from happening.
LABEL_ALIASES = {
    "bonafide": "bonafide", "bona-fide": "bonafide", "bona fide": "bonafide",
    "real": "bonafide", "genuine": "bonafide", "human": "bonafide",
    "spoof": "spoof", "spoofed": "spoof", "fake": "spoof", "tts": "spoof",
}


def normalise_label(raw):
    """Map a corpus label onto the repo's vocabulary, or raise."""
    label = LABEL_ALIASES.get(raw.strip().lower())
    if label is None:
        raise ValueError(
            f"unrecognised label {raw!r}. Downstream code filters on "
            f"'bonafide'/'spoof', so an unmapped label silently drops the trial "
            f"from every EER. Add it to datasets.LABEL_ALIASES.")
    return label


def trials_from_protocol(path, delimiter="\t", header=True, utt_col=0,
                         label_col=1, label_const=None, bonafide_when=None,
                         strip_ext=False, add_ext=None, rel_to=None):
    """([utt_id], {utt_id: label}) from a delimited protocol file.

    One reader for every corpus that ships (or was given) a protocol, so the
    per-corpus differences are declared as parameters instead of written as
    separate functions.

      delimiter          the corpus's own separator; None splits on any
                         whitespace, which several "tsv" files actually use
      header             whether to skip a first line
      utt_col/label_col  which columns carry the id and the ground truth
      label_const        for corpora that are entirely one class
      bonafide_when      when the column holds an attack id rather than a
                         label: this value means bonafide, anything else spoof
                         (SpoofCeleb's 'a00')
      strip_ext          when the score-file id drops the extension
      add_ext            when the score-file id carries one the protocol omits
      rel_to             when the protocol stores an absolute path but the id
                         is relative to a root
    """
    utts, keys = [], {}
    with open(path, newline="") as f:
        if header:
            next(f, None)
        for line in f:
            line = line.rstrip("\r\n")     # Famous Figures ships CRLF
            if not line:
                continue
            parts = line.split() if delimiter is None else line.split(delimiter)
            if len(parts) <= utt_col:
                continue
            utt = parts[utt_col].strip()
            if not utt:
                continue
            if rel_to:
                utt = os.path.relpath(utt, rel_to)
            if strip_ext:
                utt = os.path.splitext(utt)[0]
            if add_ext and not utt.endswith(add_ext):
                utt += add_ext
            if label_const is not None:
                raw = label_const
            elif len(parts) > label_col:
                raw = parts[label_col]
            else:
                continue
            if bonafide_when is not None:
                raw = "bonafide" if raw.strip() == bonafide_when else "spoof"
            try:
                label = normalise_label(raw)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from None
            utts.append(utt)
            keys[utt] = label
    return utts, keys


def trials_from_protocol_csv(path, bonafide_attack="a00"):
    """([utt_id], {utt_id: key}) from a SpoofCeleb-style CSV (file,speaker,attack).

    The utt_id is the 'file' column VERBATIM -- it already carries the .flac
    extension and reproduces the reference ids. Unlike MLAAD (all spoof) and
    M-AILABS (all bonafide), the label is per-utterance: attack == 'a00' is
    bonafide, every other attack is spoof.
    """
    utts, keys = [], {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("file") or "").strip()
            if not fn:
                continue
            attack = (row.get("attack") or "").strip()
            utts.append(fn)
            keys[fn] = "bonafide" if attack == bonafide_attack else "spoof"
    return utts, keys
