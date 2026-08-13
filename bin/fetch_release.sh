#!/usr/bin/env bash
# ============================================================================
# Fetch the published release -- score files and detector checkpoints -- into
# one directory.
#
#   {bench_root}/
#     scores/   the raw score files, in the layout scores_root expects
#     models/   the downstream detector weights
#
# The other two directories under bench_root, analysis/ and verification/, are
# generated locally; only these two halves are downloaded.
#
# Everything is verified on arrival: score files against the sha256 in
# reference/manifest.json, checkpoints against the SHA256SUMS published beside
# them. A file that is already present and already matches is skipped, so this
# is safe to re-run and safe to interrupt.
#
# Nothing here needs huggingface_hub; the repositories are public and are read
# over plain HTTPS.
#
# Edit the settings block below, then run:   bin/fetch_release.sh
#   bin/fetch_release.sh --list      # show what would be fetched, fetch nothing
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Where to put everything. The two subdirectories are created inside it.
# Defaults to bench_root in configs/paths.yaml, which falls back to the
# repo's own bench/ when that is unset.
DEST="$BENCH_DIR"

# What to fetch: "all", "scores" or "models".
WHAT="all"

# Narrow the score files. Leave empty to mean "all of them". Ignored for models.
DATASET=""
MODEL=""

# "yes" re-downloads even when the local file already matches its sha256.
FORCE="no"

# Where the two repositories live. Override if you mirror them elsewhere.
SCORES_REPO="${SPOOF_SUPERB_SCORES_REPO:-https://huggingface.co/datasets/issf/spoof-superb-scores}"
MODELS_REPO="${SPOOF_SUPERB_MODELS_REPO:-https://huggingface.co/issf/spoof-superb-models}"

# ------------------------------------------------------------ END SETTINGS --

# Flags override the settings above, so the common cases need no editing:
#   --scores | --models      fetch only that half
#   --dest DIR               download somewhere other than DEST
#   --dataset X | --model Y  narrow the score files
#   --force                  re-download even when the local copy verifies
#   --list                   show what would be fetched
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --scores)  WHAT="scores" ;;
        --models)  WHAT="models" ;;
        --all)     WHAT="all" ;;
        --dest)    DEST="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --model)   MODEL="$2"; shift ;;
        --force)   FORCE="yes" ;;
        --list)    ARGS+=("--list") ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)         echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

MANIFEST="$REPO/reference/manifest.json"
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 2; }

exec "$PY" - "$MANIFEST" "$DEST" "$WHAT" "$DATASET" "$MODEL" "$FORCE" \
             "$SCORES_REPO" "$MODELS_REPO" "$BENCH_DIR" "$SCORES_ROOT" \
             "$LINEAR_HEAD_PREFIX" "$@" <<'PYEOF'
import hashlib, json, os, sys, urllib.request, urllib.error

(manifest_path, dest, what, want_ds, want_model, force,
 scores_repo, models_repo, bench_dir, scores_root,
 linear_head_prefix) = sys.argv[1:12]
flags = sys.argv[12:]
listing = "--list" in flags

scores_dir = os.path.join(dest, "scores")
models_dir = os.path.join(dest, "models")


def resolve(repo, path):
    """A file's download URL on the Hub."""
    return f"{repo.rstrip('/')}/resolve/main/{path.lstrip('/')}"


def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest_path, sha, size_hint=None):
    """Download and verify. Returns 'ok', 'skipped' or 'failed'."""
    if force != "yes" and os.path.isfile(dest_path) and digest(dest_path) == sha:
        return "skipped"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        print(f"    FAILED {url}: {exc}", file=sys.stderr)
        return "failed"
    got = digest(tmp)
    if got != sha:
        os.remove(tmp)
        print(f"    CHECKSUM MISMATCH for {dest_path}\n"
              f"      expected {sha}\n      got      {got}", file=sys.stderr)
        return "failed"
    os.replace(tmp, dest_path)
    return "ok"


# --------------------------------------------------------------- the score files
def score_entries():
    """Every published score file, from both manifest blocks.

    ``files`` is the main results set; ``files_extra`` covers the MLAAD
    per-system scores and the non-SSL baselines, which the original manifest
    did not index.
    """
    manifest = json.load(open(manifest_path))
    for block in ("files", "files_extra"):
        for group, models in sorted(manifest.get(block, {}).items()):
            for model, e in sorted(models.items()):
                for item in (e if isinstance(e, list) else [e]):
                    if not isinstance(item, dict) or "path" not in item:
                        continue
                    if want_ds and want_ds not in (group, os.path.dirname(item["path"])):
                        continue
                    if want_model and model != want_model:
                        continue
                    yield group, model, item


# ------------------------------------------------------------- the checkpoints
#: The published repository is laid out for a human reading it: one flat
#: `{slug}.pth` per model. The repo expects the training layout instead --
#: `{prefix}{slug}/swa.pth`, and the baselines under their run directories --
#: and that is what `discover_linear_heads()` scans for and what the GMM
#: back-end opens. Rather than teach three call sites a second layout, the
#: files are put down here in the shape everything downstream already reads.
_BASELINE_DIR = "baselines"
_AASIST_RUN = "model_weighted_CCE_50_64_aasist_raw_ASV19_none"


def local_path(published):
    """Where a published checkpoint goes on disk."""
    if published.startswith("non_ssl/lfcc_gmm/gmm_"):
        cls = published.rsplit("gmm_", 1)[1].removesuffix(".pkl")   # bonafide | spoof
        return os.path.join(_BASELINE_DIR, "lfcc_gmm", cls, "gmm_final.pkl")
    if published.startswith("non_ssl/aasist_raw"):
        return os.path.join(_BASELINE_DIR, _AASIST_RUN, "swa.pth")
    if published.endswith(".pth") and "/" not in published:
        slug = published.removesuffix(".pth")
        return os.path.join(f"{linear_head_prefix}{slug}", "swa.pth")
    return published          # README.md, SHA256SUMS and anything unrecognised


def model_entries():
    """Every checkpoint, read from the SHA256SUMS published beside them."""
    url = resolve(models_repo, "SHA256SUMS")
    try:
        text = urllib.request.urlopen(url).read().decode()
    except Exception as exc:
        print(f"  cannot read {url}: {exc}", file=sys.stderr)
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, _, path = line.partition("  ")
        path = path.strip().lstrip("./")
        if path and path != "SHA256SUMS":
            yield sha.strip(), path, local_path(path)


# --------------------------------------------------------------------- listing
if listing:
    total = 0
    if what in ("all", "scores"):
        print("scores/")
        for group, model, e in score_entries():
            total += e.get("bytes", 0)
            print(f"  {group:22s} {model:34s} {e.get('bytes', 0)/1e6:8.1f} MB  {e['path']}")
    if what in ("all", "models"):
        print("models/")
        for sha, path, local in model_entries():
            arrow = f"  ->  {local}" if local != path else ""
            print(f"  {path}{arrow}")
    print(f"\n  {total/1e6:.1f} MB of score files"
          f"{' (checkpoints are ~19 MB in total)' if what in ('all','models') else ''}")
    print(f"  destination: {dest}")
    raise SystemExit(0)

# --------------------------------------------------------------------- fetching
counts = {"ok": 0, "skipped": 0, "failed": 0}

if what in ("all", "scores"):
    print(f"scores -> {scores_dir}")
    for group, model, e in score_entries():
        target = os.path.join(scores_dir, e["path"])
        if force != "yes" and os.path.isfile(target) and digest(target) == e["sha256"]:
            counts["skipped"] += 1
            continue
        print(f"  {group}/{model}  ({e.get('bytes', 0)/1e6:.1f} MB)", flush=True)
        counts[fetch(resolve(scores_repo, e["path"]), target, e["sha256"])] += 1

if what in ("all", "models"):
    print(f"models -> {models_dir}")
    for sha, path, local in model_entries():
        target = os.path.join(models_dir, local)
        if force != "yes" and os.path.isfile(target) and digest(target) == sha:
            counts["skipped"] += 1
            continue
        print(f"  {path}", flush=True)
        counts[fetch(resolve(models_repo, path), target, sha)] += 1

print(f"\n  fetched {counts['ok']}, already present {counts['skipped']}, "
      f"failed {counts['failed']}")
if counts["ok"] or counts["skipped"]:
    # Only say something if there is something to do. When the download went to
    # the configured bench_root and scores_root follows it, the benchmark is
    # already pointed at these files and any instruction here would be noise.
    if os.path.realpath(dest) != os.path.realpath(bench_dir):
        print(f"\n  This went to {dest}, which is not the configured bench_root")
        print(f"  ({bench_dir}). To read it, set in configs/paths.yaml:")
        print(f"    bench_root: {dest}")
    elif os.path.realpath(scores_root) != os.path.realpath(scores_dir):
        print(f"\n  scores_root is set to {scores_root}, so the analyses will NOT")
        print(f"  read what was just downloaded. Delete that key from")
        print(f"  configs/paths.yaml to follow bench_root, or set it to:")
        print(f"    scores_root: {scores_dir}")
    if what in ("all", "models"):
        print(f"\n  Checkpoints are in {models_dir}, laid out as the repo expects:")
        print(f"    {linear_head_prefix}{{ssl_model}}/swa.pth")
        print(f"    {_BASELINE_DIR}/  aasist_raw and lfcc_gmm")
        print(f"  So models_root can point straight at it and bin/orchestrate.sh")
        print(f"  will discover them.")
raise SystemExit(1 if counts["failed"] else 0)
PYEOF
