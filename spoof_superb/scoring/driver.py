"""The single scoring entry point: (model, checkpoint, trial source) -> score file.

Replaces eval_asvld.py, eval_mlaad.py and eval_baselines.py, which were three
copies of the same program differing only in where the trial list came from.

    python -m spoof_superb.scoring.driver --list_datasets

    # a published benchmark column, trial list from the reference score file
    python -m spoof_superb.scoring.driver --model linear_head --ssl_model xls_r_300m \
        --model_path .../swa.pth --dataset wild --output_file out.txt

    # one ASVLD condition, trial list from the protocol
    python -m spoof_superb.scoring.driver --model linear_head --ssl_model xls_r_300m \
        --model_path .../swa.pth --source asvld --asvld_condition Noise_Addition \
        --protocols_dir .../protocols --audio_base_dir .../ASVspoofLD \
        --output_file out.txt

    # MLAAD / M-AILABS, trial list by walking the corpus
    python -m spoof_superb.scoring.driver --model linear_head --ssl_model xls_r_300m \
        --model_path .../swa.pth --source walk --walk_root /data/Data/MLAAD/fake \
        --label spoof --output_file out.txt

    # SpoofCeleb, trial list and per-utterance labels from the protocol CSV
    python -m spoof_superb.scoring.driver --model linear_head --ssl_model xls_r_300m \
        --model_path .../swa.pth --source protocol_csv --output_file out.txt

Decisions carried into this merge (see REORG_PLAN.md D1-D6):
  D1  fp32 is the default; --amp is opt-in. eval_mlaad and eval_baselines
      exposed fp16 autocast, eval_asvld had no AMP path at all, and fp16 is
      what produced the NaN documented in verify_noise_rerun.py.
  D2  A CUDA device that is unavailable is a hard error for every back-end.
      eval_mlaad already refused; eval_asvld silently fell back to CPU, which
      turns a 20-minute run into a 25-hour one that is easy not to notice.
  D3  Writes are atomic everywhere (core.scorefile.write_scores).
  D4  Missing audio is filtered before scoring; undecodable audio is dropped
      during scoring and counted. Both stages, not one.
  D5  Labels come from the trial source, which is the only thing that differs
      per dataset.
"""

import argparse
import os
import sys

import numpy as np
import torch

from spoof_superb.core.scorefile import read_reference, read_utt_ids, report_eer, write_scores
from spoof_superb.scoring import backends
from spoof_superb.scoring.datasets import (
    ASVLD_CONDITIONS,
    PROTOCOL_SPECS,
    SCOREABLE,
    native_params,
    native_source,
    ASVLD_ROOT,
    DATA,
    DATASETS,
    DEFAULT_REFERENCE_SSL,
    MAILABS_ROOT,
    MLAAD_ROOT,
    SPOOFCELEB_AUDIO,
    SPOOFCELEB_PROTOCOL,
    asvld_condition_resolver,
    reference_paths,
    relative_resolver,
    trials_from_asvld_protocol,
    trials_from_protocol,
    trials_from_protocol_csv,
    trials_from_walk,
)

MODELS = ("linear_head", "aasist_raw", "lfcc_gmm")
SOURCES = ("protocol", "benchmark", "asvld", "walk", "protocol_csv")

# Conditions no ASVLD run scores. Replaces the `.asvld_skip` sentinel file,
# which lived next to eval_asvld.py, was read relative to that script's own
# location, and silently changed behaviour if the script moved. Filtering is
# excluded from the published ASVLD column, so skipping it is the pre-existing
# behaviour -- now visible instead of hidden in an untracked dotfile.
DEFAULT_SKIP_CONDITIONS = ("Filtering",)

# Per-back-end batch defaults, preserved from the drivers they came from.
DEFAULT_BATCH = {"linear_head": 32, "aasist_raw": 64, "lfcc_gmm": 0}


def _apply_dataset_defaults(args):
    """Let the dataset decide its own trial source and parameters.

    The dataset is the single input: `--dataset spoofceleb` alone determines
    the trial list, the audio root and the output path. `--source` and the
    per-source flags stay available as overrides, but nothing has to be kept
    in agreement by hand.
    """
    if args.dataset and args.dataset not in SCOREABLE:
        print(f"[ERROR] unknown dataset {args.dataset!r}. Known: "
              f"{', '.join(SCOREABLE)}")
        return False

    if args.source is None:
        args.source = native_source(args.dataset) if args.dataset else "benchmark"

    # Only fill parameters the caller did not set explicitly.
    if args.dataset and args.source == native_source(args.dataset):
        for key, value in native_params(args.dataset).items():
            if getattr(args, key, None) in (None, "", 0):
                setattr(args, key, value)
    return True


def _resolve_trials(args):
    """-> (utts, keys, resolve) for the selected source, or (None, None, None)."""
    if args.source == "benchmark":
        if args.dataset not in DATASETS:
            print(f"[ERROR] {args.dataset!r} has no published reference score "
                  f"file. Benchmark columns: {', '.join(DATASETS)}")
            return None, None, None
        spec = DATASETS[args.dataset]
        paths = ([args.reference_file] if args.reference_file
                 else reference_paths(args.dataset, args.reference_ssl))
        for p in paths:
            if not os.path.isfile(p):
                print(f"[ERROR] reference score file not found: {p}")
                return None, None, None
        utts, keys = read_reference(paths)
        for p in paths:
            print(f"[{args.dataset}] reference {p}", flush=True)
        return utts, keys, spec["resolve"]

    if args.source == "protocol":
        spec = dict(PROTOCOL_SPECS.get(args.dataset, {}))
        path = args.protocol or spec.pop("protocol", None)
        spec.pop("protocol", None)
        if not path:
            print(f"[ERROR] no protocol declared for {args.dataset!r}; pass --protocol")
            return None, None, None
        if not os.path.isfile(path):
            print(f"[ERROR] protocol not found: {path}\n"
                  f"        some are built once -- see "
                  f"spoof_superb.data.prep.build_protocols")
            return None, None, None
        utts, keys = trials_from_protocol(path, **spec)
        n_bona = sum(1 for u in utts if keys[u] == "bonafide")
        print(f"[{args.dataset}] protocol {path}: {len(utts)} utts "
              f"({n_bona} bonafide, {len(utts) - n_bona} spoof)", flush=True)
        return utts, keys, DATASETS[args.dataset]["resolve"]

    if args.source == "asvld":
        if args.asvld_condition in args.skip_conditions:
            print(f"[{args.asvld_condition}] in skip_conditions -> nothing to score.")
            return [], {}, None
        utts, keys = trials_from_asvld_protocol(args.protocols_dir, args.asvld_condition)
        if utts is None:
            print(f"[WARN] protocol missing for {args.asvld_condition} "
                  f"under {args.protocols_dir} -- skipping condition")
            return [], {}, None
        print(f"[{args.asvld_condition}] protocol utt_ids: {len(utts)}", flush=True)
        return utts, keys, asvld_condition_resolver(args.audio_base_dir, args.asvld_condition)

    if args.source == "walk":
        utts, keys = trials_from_walk(args.walk_root, args.data_base, args.label)
        print(f"  enumerated {len(utts)} wavs under {args.walk_root}", flush=True)
        return utts, keys, relative_resolver(args.data_base)

    if args.source == "protocol_csv":
        utts, keys = trials_from_protocol_csv(args.protocol_csv)
        n_bona = sum(1 for u in utts if keys[u] == "bonafide")
        print(f"  protocol {args.protocol_csv}: {len(utts)} utts "
              f"({n_bona} bonafide, {len(utts) - n_bona} spoof)", flush=True)
        return utts, keys, relative_resolver(args.audio_base)

    print(f"[ERROR] unknown source {args.source!r}")
    return None, None, None


def _apply_restrict(utts, keys, restrict_to, restrict_prefix):
    """Keep only ids present in a reference score file, in that file's order.

    Used to reproduce or verify against an existing reference subset.
    """
    wanted = read_utt_ids(restrict_to)
    if restrict_prefix:
        wanted = [u for u in wanted if u.startswith(restrict_prefix)]
    kept, dropped, seen = [], 0, set()
    for u in wanted:
        if u in seen:
            continue
        seen.add(u)
        if u in keys:
            kept.append(u)
        else:
            dropped += 1
    if dropped:
        print(f"  [WARN] {dropped} restrict utt_ids not in the trial list (dropped)")
    print(f"  restricted to reference subset: {len(kept)}", flush=True)
    return kept


def run(args):
    utts, keys, resolve = _resolve_trials(args)
    if utts is None:
        return 2
    if not utts:
        return 0        # a skipped condition is a no-op, not a failure

    if args.restrict_to:
        utts = _apply_restrict(utts, keys, args.restrict_to, args.restrict_prefix)
    if args.limit:
        utts = utts[:args.limit]
        print(f"  limited to first {len(utts)}", flush=True)

    n_bona = sum(1 for u in utts if keys[u] == "bonafide")
    print(f"  {len(utts)} trials ({n_bona} bonafide)", flush=True)

    # D4 stage 1: drop ids with no audio on disk, so counts stay honest and the
    # DataLoader cannot die mid-run.
    items, missing = [], 0
    for u in utts:
        p = resolve(u)
        if p and os.path.isfile(p):
            items.append((u, p))
        else:
            missing += 1
    if missing:
        print(f"  [WARN] {missing}/{len(utts)} trials have no audio on disk (dropped)",
              flush=True)
    print(f"  scoring {len(items)} utterances", flush=True)
    if not items:
        print("  [ERROR] nothing to score; aborting.")
        return 1

    # D2: refuse the silent CPU fallback for every torch back-end.
    if args.model != "lfcc_gmm":
        if str(args.cuda_device).startswith("cuda") and not torch.cuda.is_available():
            print(f"[ERROR] {args.cuda_device} requested but CUDA is unavailable in this "
                  f"process; refusing to fall back to CPU.")
            return 2
    device = args.cuda_device if torch.cuda.is_available() else "cpu"

    batch_size = args.batch_size or DEFAULT_BATCH[args.model]
    if args.model == "linear_head":
        if not args.ssl_model:
            print("[ERROR] --ssl_model is required for --model linear_head")
            return 2
        scored, n_bad = backends.score_linear_head(
            items, args.model_path, device, args.ssl_model,
            batch_size=batch_size, num_workers=args.num_workers, amp=args.amp)
    elif args.model == "aasist_raw":
        scored, n_bad = backends.score_aasist_raw(
            items, args.model_path, device,
            batch_size=batch_size, num_workers=args.num_workers, amp=args.amp)
    else:
        scored, n_bad = backends.score_lfcc_gmm(items, args.model_path, n_jobs=args.n_jobs)

    if n_bad:
        print(f"  [WARN] {n_bad} unreadable files skipped (no score written)", flush=True)

    bad = [(u, s) for u, s in scored if not np.isfinite(s)]
    if bad:
        print(f"  [ERROR] {len(bad)} non-finite scores, e.g. {bad[:3]}")
        return 1

    write_scores(args.output_file, scored, keys)
    report_eer(scored, keys)
    return 0


def run_eval_from_main(args, cfg, device):
    """Entry point used by main.py when cfg.model_arch is a non-SSL baseline."""
    if not args.eval_dataset:
        print("[ERROR] --eval_dataset is required for the non-SSL baselines "
              f"(one of: {', '.join(DATASETS)})")
        return 2
    if not args.eval_output:
        print("[ERROR] --eval_output is required")
        return 2

    ns = build_parser().parse_args([
        "--model", cfg.model_arch,
        "--model_path", str(cfg.pretrained_checkpoint),
        "--dataset", args.eval_dataset,
        "--output_file", args.eval_output,
        "--cuda_device", str(device),
        "--batch_size", str(args.batch_size),
        "--n_jobs", str(getattr(args, "n_jobs", 16)),
    ])
    return run(ns)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.scoring.driver",
        description="Score a model on a benchmark set -> canonical 4-column score file",
    )
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--model_path", help="swa.pth (torch back-ends) or GMM dir (lfcc_gmm)")
    ap.add_argument("--ssl_model", default=None,
                    help="s3prl upstream name; required for --model linear_head")
    ap.add_argument("--output_file")

    ap.add_argument("--source", choices=SOURCES, default=None,
                    help="override where the trial list comes from; by default "
                         "the dataset decides")
    ap.add_argument("--dataset", help="one of " + ", ".join(SCOREABLE))
    ap.add_argument("--reference_ssl", default=DEFAULT_REFERENCE_SSL,
                    help="SSL model whose score file defines the trial list")
    ap.add_argument("--reference_file", default=None,
                    help="explicit reference score file (overrides --reference_ssl)")

    ap.add_argument("--asvld_condition", choices=ASVLD_CONDITIONS,
                    help="asvld source: which laundering condition to score")
    ap.add_argument("--protocols_dir", default=None)
    ap.add_argument("--audio_base_dir", default=None,
                    help="asvld source: dir containing {condition}/flac/*.flac")
    ap.add_argument("--skip_conditions", nargs="*", default=list(DEFAULT_SKIP_CONDITIONS),
                    help="ASVLD conditions to treat as a no-op")

    ap.add_argument("--walk_root", default=None,
                    help="walk source: dir to enumerate wavs under")
    ap.add_argument("--data_base", default=None,
                    help="walk source: base dir utt_ids are written relative to")
    ap.add_argument("--label", default=None,
                    help="walk source: key for every row")

    ap.add_argument("--protocol", default=None,
                    help="protocol source: override the dataset's protocol file")
    ap.add_argument("--protocol_csv", default=None,
                    help="protocol_csv source: CSV with columns file,speaker,attack")
    ap.add_argument("--audio_base", default=None,
                    help="protocol_csv source: dir the 'file' column is relative to")

    ap.add_argument("--restrict_to", default=None,
                    help="reference score file; score only utt_ids present there")
    ap.add_argument("--restrict_prefix", default=None,
                    help="with --restrict_to, keep only ids under this prefix")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (debug)")

    ap.add_argument("--cuda_device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="0 = the back-end's default (linear_head 32, aasist_raw 64)")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--n_jobs", type=int, default=16, help="LFCC-GMM worker processes")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast. OFF by default: fp16 overflow is what put "
                         "384,157 NaN per model into the masked-spectrogram front-ends.")
    ap.add_argument("--list_datasets", action="store_true")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_datasets:
        for k in SCOREABLE:
            src = native_source(k)
            if src == "benchmark":
                refs = reference_paths(k, args.reference_ssl)
                n = sum(sum(1 for _ in open(r)) for r in refs if os.path.isfile(r))
                names = " + ".join(os.path.basename(r) for r in refs)
                print(f"{k:22s} source={src:12s} trials={n:>9} ref={names}")
            else:
                print(f"{k:22s} source={src:12s} (from the corpus/protocol)")
        return 0

    for req in ("model", "model_path", "output_file"):
        if not getattr(args, req):
            ap.error(f"--{req} is required")
    if not _apply_dataset_defaults(args):
        return 2
    if args.source == "benchmark" and not args.dataset:
        ap.error("--dataset is required for --source benchmark")
    if args.source == "asvld" and not args.asvld_condition:
        ap.error("--asvld_condition is required for --source asvld")

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
