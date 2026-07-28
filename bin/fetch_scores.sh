#!/usr/bin/env bash
# ============================================================================
# Fetch published score files into scores_root.
#
# The score files are ~6 GB and are not in this repo. They are published as a
# per-file archive; reference/manifest.json says what exists, how big it is and
# what its sha256 should be. Files are fetched individually, so verifying one
# model on one dataset costs a megabyte, not a gigabyte.
#
# Edit the settings block below, then run:   bin/fetch_scores.sh
#   bin/fetch_scores.sh --list       # show what the manifest offers, fetch nothing
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---------------------------------------------------------------- SETTINGS --

# Which files to fetch. Leave DATASET or MODEL empty to mean "all of them".
DATASET="wild"
MODEL="xls_r_300m"

# Where the archive is published. Overrides the URL recorded in the manifest.
# Leave empty to use the manifest's own archive_url.
ARCHIVE_URL="${SPOOF_SUPERB_SCORES_URL:-}"

# "yes" re-downloads even when the local file already matches its sha256.
FORCE="no"

# ------------------------------------------------------------ END SETTINGS --

MANIFEST="$REPO/reference/manifest.json"
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST" >&2; exit 2; }

exec "$PY" - "$MANIFEST" "$SCORES_ROOT" "$DATASET" "$MODEL" "$ARCHIVE_URL" "$FORCE" "$@" <<'PYEOF'
import hashlib, json, os, sys, urllib.request

manifest_path, scores_root, want_ds, want_model, archive_url, force = sys.argv[1:7]
flags = sys.argv[7:]
manifest = json.load(open(manifest_path))
archive_url = archive_url or manifest.get("archive_url") or ""

def entries():
    for ds, models in sorted(manifest.get("datasets", {}).items()):
        if want_ds and ds != want_ds:
            continue
        for model, e in sorted(models.items()):
            if want_model and model != want_model:
                continue
            for item in (e if isinstance(e, list) else [e]):
                yield ds, model, item

if "--list" in flags:
    total = 0
    for ds, model, e in entries():
        total += e["bytes"]
        print(f"  {ds:22s} {model:32s} {e['bytes']/1e6:8.1f} MB  {e['path']}")
    print(f"\n  {total/1e6:.1f} MB total")
    raise SystemExit(0)

if not archive_url:
    print("No archive URL. Set ARCHIVE_URL in this script, export "
          "SPOOF_SUPERB_SCORES_URL, or rebuild the manifest with --archive_url.",
          file=sys.stderr)
    raise SystemExit(2)

def digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

ok = skipped = failed = 0
for ds, model, e in entries():
    dest = os.path.join(scores_root, e["path"])
    if force != "yes" and os.path.isfile(dest) and digest(dest) == e["sha256"]:
        skipped += 1
        continue
    url = archive_url.rstrip("/") + "/" + e["path"].lstrip("/")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"  fetching {ds}/{model}  ({e['bytes']/1e6:.1f} MB)", flush=True)
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        print(f"    FAILED {url}: {exc}", file=sys.stderr)
        failed += 1
        continue
    got = digest(tmp)
    if got != e["sha256"]:
        os.remove(tmp)
        print(f"    CHECKSUM MISMATCH for {e['path']}\n"
              f"      expected {e['sha256']}\n      got      {got}", file=sys.stderr)
        failed += 1
        continue
    os.replace(tmp, dest)
    ok += 1

print(f"\n  fetched {ok}, already present {skipped}, failed {failed}")
raise SystemExit(1 if failed else 0)
PYEOF
