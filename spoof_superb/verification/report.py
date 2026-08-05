"""Rendering: one Markdown report a human reads, one JSON a script reads.

Both levels write both. The Markdown is the deliverable -- it is what somebody
pastes into an issue when their reproduction disagrees with ours -- so it is
ordered by what that person needs, not by what is easiest to emit:

  1. the verdict, and whether the run passed
  2. the tally, so the shape of the disagreement is visible in one glance
  3. every cell that needs attention, WORST FIRST, each with the reason that
     placed it there
  4. the full matrix last, as reference

A report that lists 190 passing cells before the one that failed is a report
nobody reads to the bottom.
"""

import json
from pathlib import Path

from spoof_superb.verification.verdicts import (ANALYSIS_LADDER, IS_FAILURE,
                                                SCORE_LADDER, rank)

__all__ = ["write_reports", "score_markdown", "analysis_markdown", "tally"]


def tally(rows, key="verdict"):
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def _tally_table(counts, ladder):
    lines = ["| verdict | cells | |",
             "|---|---:|---|"]
    for v in sorted(counts, key=lambda v: rank(v, ladder)):
        flag = "**fails the run**" if v in IS_FAILURE else ""
        lines.append(f"| `{v}` | {counts[v]} | {flag} |")
    return "\n".join(lines)


def _fmt(v, spec=".3f"):
    if v is None:
        return "-"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def score_markdown(rows, meta):
    counts = tally(rows)
    failed = [r for r in rows if r["verdict"] in IS_FAILURE]
    status = "FAIL" if failed else "PASS"

    out = [f"# Score-file verification -- {status}", ""]
    out.append(f"- candidate: `{meta['candidate_root']}`")
    if meta.get("ref_root"):
        out.append(f"- reference: `{meta['ref_root']}`  "
                   f"(full per-utterance comparison)")
    else:
        out.append(f"- reference: `{meta['manifest']}`  (manifest mode -- "
                   f"trial sets compared by digest, scores by EER)")
    out.append(f"- {len(rows)} cells = {meta['n_datasets']} datasets x "
               f"{meta['n_models']} models")
    out.append("")
    out.append(_tally_table(counts, SCORE_LADDER))
    out.append("")

    if not meta.get("ref_root") and any(
            r["verdict"] == "SCORES_DIFFER" for r in rows):
        out += [
            "> Manifest mode cannot separate a genuine score difference from a",
            "> flat-DET `SENSITIVE` cell, because rank agreement is not in the",
            "> manifest. Re-run the cells above with `--ref-root` against the",
            "> reference score tree to decide.", ""]

    attention = sorted(
        (r for r in rows if r["verdict"] not in ("IDENTICAL", "EQUIVALENT")),
        key=lambda r: (-rank(r["verdict"], SCORE_LADDER),
                       -(r.get("d_eer") or 0.0)))
    if attention:
        out.append(f"## Cells needing attention ({len(attention)})")
        out.append("")
        out.append("| dataset | model | verdict | EER ref | EER cand | dEER pp | why |")
        out.append("|---|---|---|---:|---:|---:|---|")
        for r in attention:
            ref = r.get("eer_a_common", r.get("eer_a"))
            cand = r.get("eer_b_common", r.get("eer_b"))
            out.append(f"| {r['dataset']} | `{r['model']}` | `{r['verdict']}` | "
                       f"{_fmt(ref)} | {_fmt(cand)} | {_fmt(r.get('d_eer'), '.4f')} | "
                       f"{r.get('reason', '')} |")
        out.append("")

    out.append("## All cells")
    out.append("")
    out.append("| dataset | model | verdict | n ref | n cand | EER ref | EER cand | dEER pp |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        ref = r.get("eer_a_common", r.get("eer_a"))
        cand = r.get("eer_b_common", r.get("eer_b"))
        out.append(f"| {r['dataset']} | `{r['model']}` | `{r['verdict']}` | "
                   f"{_fmt(r.get('n_a'))} | {_fmt(r.get('n_b'))} | "
                   f"{_fmt(ref)} | {_fmt(cand)} | {_fmt(r.get('d_eer'), '.4f')} |")
    out.append("")
    return "\n".join(out)


def analysis_markdown(entries, meta):
    counts = tally(entries)
    failed = [e for e in entries if e["verdict"] in IS_FAILURE]
    status = "FAIL" if failed else "PASS"

    out = [f"# Analysis verification -- {status}", ""]
    out.append(f"- candidate: `{meta['candidate_root']}`")
    out.append(f"- reference: `{meta['reference_root']}`")
    out.append("")
    out.append(_tally_table(counts, ANALYSIS_LADDER))
    out.append("")
    out.append("A table is graded on whether the paper's CLAIMS survive, not on "
               "how far its cells moved. Cell deltas are reported beside each "
               "verdict as the diagnostic they are.")
    out.append("")

    for e in entries:
        rep = e.get("report", {})
        out.append(f"## `{e['table']}` -- {e['verdict']}")
        out.append("")
        if rep.get("note"):
            out += [rep["note"], ""]
            continue

        cells = rep.get("cells", {})
        out.append(f"- {rep.get('n_rows_shared')} models x "
                   f"{rep.get('n_cols_shared')} columns compared")
        if cells:
            out.append(f"- cell deltas: max **{_fmt(cells.get('max_delta_pp'), '.4f')} pp**, "
                       f"median {_fmt(cells.get('median_delta_pp'), '.4f')} pp, "
                       f"{cells.get('n_over_tol')}/{cells.get('n_compared')} over "
                       f"{cells.get('tol_pp')} pp")
        for label, key in (("models only in the reference", "rows_reference_only"),
                           ("models only in the candidate", "rows_candidate_only"),
                           ("columns only in the reference", "cols_reference_only"),
                           ("columns only in the candidate", "cols_candidate_only")):
            if rep.get(key):
                out.append(f"- {label}: {', '.join(rep[key])}")
        out.append("")

        broken = rep.get("broken_claims") or []
        if broken:
            out.append("**Claims that changed:**")
            out.append("")
            for b in broken:
                out.append(f"- {b}")
            out.append("")
        else:
            out.append("Every claim checked survives: best-in-column, the "
                       "column ordering by mean EER, and the emphasis markers "
                       "where the table carries them.")
            out.append("")

        soft = rep.get("soft_rank_columns") or []
        if soft:
            sp = rep.get("claims", {}).get("model_rank_correlation", {})
            out.append("Model ordering loosened (rank correlation below "
                       "threshold) in: "
                       + ", ".join(f"`{c}` ({_fmt(sp.get(c), '.4f')})" for c in soft))
            out.append("")

        worst = (cells or {}).get("worst") or []
        if worst:
            out.append("<details><summary>Largest cell differences</summary>")
            out.append("")
            out.append("| model | column | reference | candidate | delta pp |")
            out.append("|---|---|---:|---:|---:|")
            for w in worst:
                out.append(f"| {w['model']} | {w['column']} | "
                           f"{_fmt(w['reference'], '.4f')} | "
                           f"{_fmt(w['candidate'], '.4f')} | "
                           f"{_fmt(w['delta'], '.4f')} |")
            out += ["", "</details>", ""]

        one_sided = (cells or {}).get("one_sided") or []
        if one_sided:
            out.append(f"{cells.get('n_one_sided')} cell(s) present on one side "
                       f"only -- e.g. "
                       + ", ".join(f"{o['model']}/{o['column']}"
                                   for o in one_sided[:5]))
            out.append("")
    return "\n".join(out)


def write_reports(out_dir, stem, markdown, payload):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{stem}.md"
    js = out_dir / f"{stem}.json"
    md.write_text(markdown)
    js.write_text(json.dumps(payload, indent=2, default=str))
    return md, js
