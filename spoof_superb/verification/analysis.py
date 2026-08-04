"""Level 2 -- verify analysis tables against the published reference.

    python -m spoof_superb.verification analysis --candidate outputs

Level 1 asks whether the score files reproduce. This asks the question a reader
of the paper actually has: **do the same conclusions come out?**

Those are not the same question, and the difference runs in both directions.
A run can miss every cell by 0.3 pp and still support every sentence in the
paper. A run can miss one cell by 0.4 pp and change which model is best on a
column -- which is a sentence. So grading on `max |delta|` alone is wrong twice
over: it fails reproductions that succeeded, and it passes reproductions that
changed a finding.

What is reported, therefore, is three layers, in this order:

1. **Structure.** Which models and which columns each side has. A missing row is
   not a small number, it is an absent measurement, and no delta describes it.
2. **Cells.** Per-cell |delta| in percentage points, with the distribution and
   the worst offenders named. This is the diagnostic layer -- it tells you where
   to look, not whether you passed.
3. **Claims.** The things the paper asserts, checked one at a time:
     * which model is best in each column          (an argmin, per column)
     * the top-five set under Mean                 (the caption's bolding rule)
     * the ordering of the columns by their mean   ("which degradation hurts
       most", "which architecture group is hardest") -- these ARE the sentences
       in 4.4.2 and 4.4.3
     * the model ordering within each column       (rank correlation)
   Plus, where the CSV carries the paper's own `*` emphasis markers, a direct
   comparison of the marked cells. That checks the published claim as published,
   with no rule restated here to drift away from the one that produced it.

The verdict grades layer 3 and reports layer 2 beside it, so a user can always
see both "how far off am I" and "does it matter".
"""

import csv
import math
from pathlib import Path

from spoof_superb.verification.verdicts import CELL_TOL_PP, RANK_TOL

__all__ = ["TABLES", "Table", "load_table", "compare_table", "verify_analysis"]

#: The reference tables, as (analysis sub-directory, file name, claim options).
#:
#: `exclude` names rows that must not compete for "best in column". The paper's
#: caption is explicit: bold marks the best SSL MODEL, and the two non-SSL rows
#: are the thing being compared against, not competitors in the comparison.
TABLES = [
    ("main_results", "main_results_table.csv",
     {"exclude": ("LFCC-GMM", "AASIST"), "mean_col": "Mean"}),
    ("degradation", "eer_matrix.csv", {"reference_col": "Baseline"}),
    ("tts", "eer_by_tts_system.csv", {}),
    ("tts", "eer_by_architecture.csv", {}),
    ("tts", "eer_by_generation_mode.csv", {}),
    ("tts", "eer_by_vocoder_family.csv", {}),
]


class Table:
    """A models x columns matrix of EERs, plus any emphasis markers."""

    def __init__(self, index, columns, values, marked):
        self.index = index          # row names, in file order
        self.columns = columns      # column names, in file order
        self.values = values        # {(row, col): float or None}
        self.marked = marked        # {(row, col)} carrying a '*' in the CSV

    def col(self, name, rows=None):
        """(row, value) pairs for one column, skipping absent values."""
        rows = rows if rows is not None else self.index
        return [(r, self.values[(r, name)]) for r in rows
                if self.values.get((r, name)) is not None]


def load_table(path):
    """Read an analysis CSV. `*` marks emphasis and is stripped from the number."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path}: empty")
    header = rows[0]
    columns = header[1:]
    index, values, marked = [], {}, set()
    for raw in rows[1:]:
        if not raw or not raw[0].strip():
            continue
        name = raw[0].strip()
        index.append(name)
        for col, cell in zip(columns, raw[1:]):
            cell = (cell or "").strip()
            if cell.endswith("*"):
                marked.add((name, col))
                cell = cell[:-1].strip()
            try:
                values[(name, col)] = float(cell)
            except ValueError:
                values[(name, col)] = None      # 'TODO', '', 'n/a'
    return Table(index, columns, values, marked)


def _spearman(pairs_a, pairs_b):
    """Rank correlation of two orderings over the same names. None if degenerate."""
    names = [n for n, _ in pairs_a]
    b = dict(pairs_b)
    names = [n for n in names if n in b]
    if len(names) < 3:
        return None
    a = dict(pairs_a)
    ra = _ranks([a[n] for n in names])
    rb = _ranks([b[n] for n in names])
    n = len(names)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


def _column_order(table, columns, rows):
    """Columns sorted by their mean over the models -- the paper's orderings."""
    means = []
    for c in columns:
        vals = [v for _r, v in table.col(c, rows)]
        if vals:
            means.append((c, sum(vals) / len(vals)))
    return [c for c, _ in sorted(means, key=lambda t: t[1])]


def compare_table(ref, cand, exclude=(), mean_col=None, reference_col=None):
    """Structure, cells and claims for one table. Returns (verdict, report)."""
    rows_shared = [r for r in ref.index if r in set(cand.index)]
    cols_shared = [c for c in ref.columns if c in set(cand.columns)]

    rep = {
        "rows_reference_only": [r for r in ref.index if r not in set(cand.index)],
        "rows_candidate_only": [r for r in cand.index if r not in set(ref.index)],
        "cols_reference_only": [c for c in ref.columns if c not in set(cand.columns)],
        "cols_candidate_only": [c for c in cand.columns if c not in set(ref.columns)],
        "n_rows_shared": len(rows_shared),
        "n_cols_shared": len(cols_shared),
    }

    if not rows_shared or not cols_shared:
        return "STRUCTURE_DIFFERS", dict(
            rep, note="no shared rows or columns; nothing is comparable")

    # ---- layer 2: cells --------------------------------------------------
    deltas, absent = [], []
    for r in rows_shared:
        for c in cols_shared:
            a, b = ref.values.get((r, c)), cand.values.get((r, c))
            if a is None or b is None:
                if a is not b:
                    absent.append({"model": r, "column": c,
                                   "reference": a, "candidate": b})
                continue
            deltas.append({"model": r, "column": c, "reference": a,
                           "candidate": b, "delta": abs(a - b)})
    ordered = sorted(deltas, key=lambda d: -d["delta"])
    ds = [d["delta"] for d in ordered]
    rep["cells"] = {
        "n_compared": len(ds),
        "n_over_tol": sum(1 for d in ds if d > CELL_TOL_PP),
        "tol_pp": CELL_TOL_PP,
        "max_delta_pp": ds[0] if ds else None,
        "median_delta_pp": (sorted(ds)[len(ds) // 2] if ds else None),
        # Only cells actually over tolerance. Taking the top ten unconditionally
        # padded the list with exact matches, which reads as nine more problems
        # than there are.
        "worst": [d for d in ordered if d["delta"] > CELL_TOL_PP][:10],
        "one_sided": absent[:10],
        "n_one_sided": len(absent),
    }

    # ---- layer 3: claims -------------------------------------------------
    live = [r for r in rows_shared if r not in set(exclude)]
    claims, broken = {}, []

    best = {}
    for c in cols_shared:
        ra, rb = ref.col(c, live), cand.col(c, live)
        wa = min(ra, key=lambda t: t[1])[0] if ra else None
        wb = min(rb, key=lambda t: t[1])[0] if rb else None
        best[c] = {"reference": wa, "candidate": wb, "same": wa == wb}
        if wa != wb:
            broken.append(f"best on {c}: {wa} -> {wb}")
    claims["best_per_column"] = best

    if mean_col and mean_col in cols_shared:
        ta = [n for n, _ in sorted(ref.col(mean_col, live), key=lambda t: t[1])[:5]]
        tb = [n for n, _ in sorted(cand.col(mean_col, live), key=lambda t: t[1])[:5]]
        claims["mean_top5"] = {"reference": ta, "candidate": tb,
                               "same_set": set(ta) == set(tb),
                               "same_order": ta == tb}
        if set(ta) != set(tb):
            broken.append(f"top-5 under {mean_col}: "
                          f"{sorted(set(ta) ^ set(tb))} moved in or out")

    rank_cols = [c for c in cols_shared if c != mean_col]
    oa = _column_order(ref, rank_cols, live)
    ob = _column_order(cand, rank_cols, live)
    claims["column_order_by_mean"] = {"reference": oa, "candidate": ob,
                                      "same": oa == ob}
    if oa != ob:
        broken.append("the ordering of columns by mean EER changed: "
                      f"{' < '.join(oa)}  ->  {' < '.join(ob)}")

    if reference_col and reference_col in cols_shared:
        # Degradation reports relative change against a clean baseline. If the
        # sign of that change flips, the paper's claim about a condition -- that
        # it HURTS -- has reversed, however small the cells moved.
        flips = []
        for c in cols_shared:
            if c == reference_col:
                continue
            for r in live:
                base_a = ref.values.get((r, reference_col))
                base_b = cand.values.get((r, reference_col))
                a, b = ref.values.get((r, c)), cand.values.get((r, c))
                if None in (base_a, base_b, a, b):
                    continue
                if (a >= base_a) != (b >= base_b):
                    flips.append({"model": r, "column": c})
        claims["degradation_sign_flips"] = flips
        if flips:
            broken.append(f"{len(flips)} cell(s) changed the SIGN of their "
                          f"change against {reference_col}")

    spearman = {}
    for c in cols_shared:
        s = _spearman(ref.col(c, live), cand.col(c, live))
        spearman[c] = s
    claims["model_rank_correlation"] = spearman
    soft = [c for c, s in spearman.items() if s is not None and s < RANK_TOL]

    if ref.marked or cand.marked:
        shared_grid = {(r, c) for r in rows_shared for c in cols_shared}
        ma = ref.marked & shared_grid
        mb = cand.marked & shared_grid
        claims["emphasis_markers"] = {
            "same": ma == mb,
            "reference_only": sorted(f"{r}/{c}" for r, c in ma - mb),
            "candidate_only": sorted(f"{r}/{c}" for r, c in mb - ma),
        }
        if ma != mb:
            broken.append(f"{len(ma ^ mb)} emphasis marker(s) moved -- the "
                          f"published table would print different bolding")

    rep["claims"] = claims
    rep["broken_claims"] = broken

    # ---- verdict ---------------------------------------------------------
    if rep["rows_reference_only"] or rep["cols_reference_only"] or absent:
        verdict = "STRUCTURE_DIFFERS"
    elif broken:
        verdict = "CONCLUSIONS_DIFFER"
    elif not ds:
        verdict = "STRUCTURE_DIFFERS"
    elif rep["cells"]["n_over_tol"] == 0:
        verdict = "IDENTICAL"
    elif soft:
        verdict = "CONCLUSIONS_HOLD"
    else:
        verdict = "EQUIVALENT"
    rep["soft_rank_columns"] = soft
    return verdict, rep


def verify_analysis(candidate_root, reference_root, tables=None, progress=print):
    """One entry per reference table.

    `tables` restricts the check to some of them, by sub-directory or by
    `sub/name`. It exists so a script that computed ONE analysis can verify what
    it computed, instead of reporting the other five as MISSING failures. It is
    a narrowing of the QUESTION, not a tolerance: a table you asked for and did
    not produce is still MISSING, and the default remains all six.
    """
    wanted = _select(tables)
    out = []
    for sub, name, opts in wanted:
        ref_path = Path(reference_root) / sub / name
        cand_path = Path(candidate_root) / sub / name
        entry = {"table": f"{sub}/{name}", "reference": str(ref_path),
                 "candidate": str(cand_path)}
        if not ref_path.is_file():
            entry.update(verdict="MISSING",
                         report={"note": f"no reference table at {ref_path}"})
        elif not cand_path.is_file():
            entry.update(verdict="MISSING",
                         report={"note": f"no candidate table at {cand_path} -- "
                                         f"run the analysis that writes it"})
        else:
            try:
                verdict, rep = compare_table(load_table(ref_path),
                                             load_table(cand_path), **opts)
                entry.update(verdict=verdict, report=rep)
            except Exception as exc:
                entry.update(verdict="ERROR",
                             report={"note": f"{type(exc).__name__}: {exc}"})
        progress(f"  {entry['table']:<36} {entry['verdict']}")
        out.append(entry)
    return out


def _select(tables):
    """Resolve a --tables filter, or every table when none is given."""
    if not tables:
        return TABLES
    keep = set(tables)
    chosen = [t for t in TABLES if t[0] in keep or f"{t[0]}/{t[1]}" in keep]
    if not chosen:
        known = sorted({t[0] for t in TABLES})
        raise SystemExit(f"--tables matched nothing. Known: {', '.join(known)}")
    return chosen
