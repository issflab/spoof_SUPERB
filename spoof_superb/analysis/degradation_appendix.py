"""Appendix tables for the matched acoustic-degradation analysis.

    python -m spoof_superb.analysis.degradation_appendix

Reads `{outputs_root}/degradation_matched/` and writes three `.tex` fragments
beside it. They are pasted into access.tex, not `\\input` from it, so the paper
carries its own copy; regenerating here does not update the paper until they
are re-inlined. Table floats must sit at top level in access.tex and never
inside a `\\textcolor{red}{...}` group -- a float in a horizontal-mode group
raises "Not in outer par mode". Review markup goes on the captions instead.

Three tables, and why
---------------------
`per_variant_eer.csv` is 19 models x 54 variants = 1,026 EERs. Printing it in
full would run pages; as a heatmap the annotations land near 1.5 pt at column
width. These three carry what the main text leans on, and the full matrix
ships as the released CSV.

  A1  what each cell is made of. Single column, no numbers -- it is the key
      naming the variant that sits in each column of A3.
  A2  per-cell mean +- sd, 19 x 9. Recovers what was removed from Fig. 1.
  A3  per-variant EER averaged over the models, as a grid: condition and
      corpus down the side, variant slots across. This is the table showing a
      cell is not one thing -- ASV5's ten codecs run 20.5 to 32.2 against a
      17.75 Baseline, while ASVLD's six bitrates sit between 17.7 and 18.0.

A2 collapses the variant axis and A3 the model axis, so both margins of the
full matrix are in the paper; only the interior goes to the CSV.

Two earlier layouts were tried and rejected. A 54-row long table ran over a
page while repeating each cell's description 54 times. Folding the variants
into one wrapped paragraph per cell fitted in nine rows but left nothing
aligned, so reading "which bitrate gave that value" meant counting along two
lists in parallel. The grid below keeps each variant in a fixed column and
puts the naming in A1.

`VARIANT_ORDER` is the single source of the left-to-right ordering. A1 names
the variants in that order and A3 prints them in it, so the key and the grid
cannot drift apart.
"""

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from spoof_superb import REPO_ROOT
from spoof_superb.config import cfg
from spoof_superb.scoring.models import display_by_slug, paper_table_rows

#: Canonical left-to-right variant order per cell. A1 and A3 both read this.
VARIANT_ORDER = OrderedDict([
    ("Codec:ASVLD", [f"recompression_{b}k" for b in (16, 64, 128, 196, 256, 320)]),
    ("Codec:ASV21DF", ["low_mp3", "high_mp3", "low_m4a", "high_m4a",
                       "low_ogg", "high_ogg", "mp3m4a", "oggm4a"]),
    ("Codec:ASV5", [f"C{i:02d}" for i in range(1, 11)]),
    ("Noise:ASVLD", [f"{t}_{s}" for s in (0, 10, 20)
                     for t in ("babble", "cafe", "street", "volvo", "white")]),
    ("Resampling:ASVLD", [f"resample_{r}" for r in (8000, 11025, 22050, 44100)]),
    ("Reverb:ASVLD", ["RT_0_3", "RT_0_6", "RT_0_9"]),
    ("Channel:ASVLD", ["lpf_7000"]),
    ("Channel:ASV21LA", ["alaw", "ulaw", "g722", "gsm", "opus", "pstn"]),
    ("Channel:ASV5", ["C11"]),
])

#: Cell -> (condition, corpus) as printed. Blank condition continues the group.
NAMES = OrderedDict([
    ("Codec:ASVLD", ("Codec", "ASVLD")),
    ("Codec:ASV21DF", ("", "ASV21 DF")),
    ("Codec:ASV5", ("", "ASV5")),
    ("Noise:ASVLD", ("Noise", "ASVLD")),
    ("Resampling:ASVLD", ("Resamp.", "ASVLD")),
    ("Reverb:ASVLD", ("Reverb", "ASVLD")),
    ("Channel:ASVLD", ("Channel", "ASVLD")),
    ("Channel:ASV21LA", ("", "ASV21 LA")),
    ("Channel:ASV5", ("", "ASV5")),
])

#: Short description for A1, written in VARIANT_ORDER order.
DESCRIPTION = {
    "Codec:ASVLD": r"16, 64, 128, 196, 256, 320~kbps",
    "Codec:ASV21DF": r"mp3, m4a, ogg at low then high rate; mp3m4a; oggm4a",
    "Codec:ASV5": r"C01--C10, traditional and neural codecs",
    "Noise:ASVLD": r"babble, cafe, street, volvo, white; one A3 row per SNR",
    "Resampling:ASVLD": r"8, 11.025, 22.05, 44.1~kHz",
    "Reverb:ASVLD": r"$RT_{60}$ = 0.3, 0.6, 0.9~s",
    "Channel:ASVLD": r"7~kHz low-pass",
    "Channel:ASV21LA": r"a-law, $\mu$-law, G.722, GSM, Opus, PSTN",
    "Channel:ASV5": r"C11, telephony channel",
}


#: Printed column label per variant. A3 prints a name row above each value
#: row, so a reader never has to hold A1 open beside it.
VARIANT_LABEL = {
    **{f"recompression_{b}k": str(b) for b in (16, 64, 128, 196, 256, 320)},
    "low_mp3": r"mp3$_{\text{lo}}$", "high_mp3": r"mp3$_{\text{hi}}$",
    "low_m4a": r"m4a$_{\text{lo}}$", "high_m4a": r"m4a$_{\text{hi}}$",
    "low_ogg": r"ogg$_{\text{lo}}$", "high_ogg": r"ogg$_{\text{hi}}$",
    "mp3m4a": "mp3m4a", "oggm4a": "oggm4a",
    **{f"C{i:02d}": f"C{i:02d}" for i in range(1, 11)}, "C11": "C11",
    **{f"{t}_{s}": t for t in ("babble", "cafe", "street", "volvo", "white")
       for s in (0, 10, 20)},
    **{f"resample_{r}": lab for r, lab in
       ((8000, "8"), (11025, "11.025"), (22050, "22.05"), (44100, "44.1"))},
    "RT_0_3": "0.3", "RT_0_6": "0.6", "RT_0_9": "0.9",
    "lpf_7000": r"7\,kHz",
    "alaw": "a-law", "ulaw": r"$\mu$-law", "g722": "G.722",
    "gsm": "GSM", "opus": "Opus", "pstn": "PSTN",
}

#: Unit shown beside the variant names, where the names are bare numbers.
UNIT = {
    "Codec:ASVLD": "kbps", "Resampling:ASVLD": "kHz",
    "Reverb:ASVLD": r"$RT_{60}$, s",
}

#: What the degradation physically is, one per A3 row group.
DEGRADATION = {
    "Codec:ASVLD": "Re-compression",
    "Codec:ASV21DF": "Lossy transcode",
    "Codec:ASV5": "Traditional + neural codecs",
    "Noise:ASVLD": "Additive environmental noise",
    "Resampling:ASVLD": "Sample-rate reduction",
    "Reverb:ASVLD": "Simulated room impulse response",
    "Channel:ASVLD": "Low-pass filter",
    "Channel:ASV21LA": "Telephony codecs and PSTN",
    "Channel:ASV5": "Telephony channel",
}

#: A3 groups: one per corpus. The SNR list is set only for additive noise,
#: whose three levels share a single row of noise-type names.
A3_GROUPS = [
    ("Codec", "ASVLD", "Codec:ASVLD", None),
    ("", "ASV21 DF", "Codec:ASV21DF", None),
    ("", "ASV5", "Codec:ASV5", None),
    ("Noise", "ASVLD", "Noise:ASVLD", (0, 10, 20)),
    ("Resamp.", "ASVLD", "Resampling:ASVLD", None),
    ("Reverb", "ASVLD", "Reverb:ASVLD", None),
    ("Channel", "ASVLD", "Channel:ASVLD", None),
    ("", "ASV21 LA", "Channel:ASV21LA", None),
    ("", "ASV5", "Channel:ASV5", None),
]

NCOL = 10   # widest A3 row: ASV5's ten codecs


def _verify(per_variant):
    seen = {}
    for r in per_variant:
        seen.setdefault(r["column"], set()).add(r["variant"])
    for cell, order in VARIANT_ORDER.items():
        got, expect = seen.get(cell, set()), set(order)
        if got != expect:
            raise SystemExit(
                f"FATAL: {cell} variants changed.\n"
                f"  missing: {sorted(expect - got)}\n"
                f"  new:     {sorted(got - expect)}\n"
                f"Update VARIANT_ORDER and DESCRIPTION together.")


def _write(lines, out):
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def table_composition(out):
    """A1: the key to A3's columns. Single column, no numbers."""
    lines = [
        r"% Generated by spoof_superb.analysis.degradation_appendix -- do not edit.",
        r"\begin{table}[!t]", r"\centering",
        r"\caption{\textcolor{red}{Composition of the nine degradation cells in"
        r" Figs.~\ref{fig:abs_eer_heatmap} and~\ref{fig:rel_eer_heatmap}. Each cell"
        r" replaces the named corpus's clean partition with its $n$ degraded variants,"
        r" holding the other three corpora at their clean Baseline partitions. Variants"
        r" are listed in the order they occupy the columns of"
        r" Table~\ref{tab:degradation_variants}.}}",
        r"\label{tab:degradation_cells}", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llcp{4.1cm}@{}}", r"\toprule",
        r"\textbf{Condition} & \textbf{Corpus} & \textbf{$n$} & \textbf{Variants} \\",
        r"\midrule",
    ]
    first = True
    for cell, (cond, corpus) in NAMES.items():
        if cond and not first:
            lines.append(r"\addlinespace[2pt]")
        lines.append(f"{cond} & {corpus} & {len(VARIANT_ORDER[cell])} & "
                     f"{DESCRIPTION[cell]} \\\\")
        first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(lines, out)


def table_spread(spread, out):
    """A2: per-cell mean +- sd, one row per model."""
    by = {}
    for r in spread:
        by.setdefault(r["Model"], {})[r["column"]] = r
    order = [m for m in paper_table_rows() if m in by]
    cols = list(NAMES)
    head = " & ".join(r"\textbf{" + NAMES[c][1] + "}" for c in cols)
    lines = [
        r"% Generated by spoof_superb.analysis.degradation_appendix -- do not edit.",
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{\textcolor{red}{Per-cell EER (\%) as mean $\pm$ standard deviation"
        r" over the variants listed in Table~\ref{tab:degradation_cells}. The means are"
        r" the values plotted in Fig.~\ref{fig:abs_eer_heatmap}. Channel:ASVLD and"
        r" Channel:ASV5 are single-variant cells and carry no spread. A large spread"
        r" means the cell averages settings that behave very differently: ASV5's ten"
        r" codecs disagree by up to 7.7 points of EER, while ASV21 DF's eight transcodes"
        r" agree to within 0.9.}}",
        r"\label{tab:degradation_spread}", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Codec}} & \textbf{Noise} & \textbf{Resamp.}"
        r" & \textbf{Reverb} & \multicolumn{3}{c}{\textbf{Channel}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}"
        r"\cmidrule(lr){7-7}\cmidrule(lr){8-10}",
        r"\textbf{Model} & " + head + r" \\",
        r"\midrule",
    ]
    for m in order:
        cells = []
        for c in cols:
            r = by[m][c]
            mean, sd, n = float(r["mean"]), float(r["sd"]), int(r["n"])
            cells.append(f"{mean:.1f}" if n == 1
                         else f"{mean:.1f}\\,$\\pm$\\,{sd:.1f}")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write(lines, out)


def table_variants(per_variant, baselines, out):
    """A3: per-variant dEER, with names, a description, and ruled row groups.

    dEER rather than EER: against a 17.75 Baseline the homogeneous cells all
    read 17.4-18.0, and "no effect" is only visible if the reader keeps the
    Baseline in mind. In dEER it is zero. It also rescues rows like Channel on
    ASV21 LA, where EER 18.2/18.1/17.9/18.8/17.8/19.6 hides that PSTN costs six
    times what any other codec in the row does.

    A2 already carries the absolute EER of every cell, so the two tables
    complement rather than duplicate each other.
    """
    dee = {}
    for r in per_variant:
        b = baselines[r["Model"]]
        dee.setdefault(r["column"], {}).setdefault(r["variant"], []).append(
            (float(r["EER"]) - b) / b * 100.0)
    mean = {c: {v: float(np.mean(x)) for v, x in d.items()} for c, d in dee.items()}

    def pad(cells):
        return cells + [""] * (NCOL - len(cells))

    body = []
    for i, (cond, corpus, cell, snrs) in enumerate(A3_GROUPS):
        if i:
            body.append(r"\midrule")
        order = VARIANT_ORDER[cell]
        if snrs:
            order = [v for v in order if v.endswith("_0")]
        unit = UNIT.get(cell)
        head = corpus + (f" ({unit})" if unit else "")
        names = [r"\textit{" + VARIANT_LABEL[v] + "}" for v in order]
        body.append(" & ".join([cond, head, DEGRADATION[cell]] + pad(names)) + r" \\")
        if snrs:
            for snr in snrs:
                vals = [f"{mean[cell][v.replace('_0', f'_{snr}')]:+.1f}" for v in order]
                body.append(" & ".join(["", rf"\quad {snr}~dB SNR", ""] + pad(vals))
                            + r" \\")
        else:
            vals = [f"{mean[cell][v]:+.1f}" for v in order]
            body.append(" & ".join(["", r"\quad $\Delta$EER", ""] + pad(vals)) + r" \\")

    lines = [
        r"% Generated by spoof_superb.analysis.degradation_appendix -- do not edit.",
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{\textcolor{red}{Relative EER change ($\Delta$EER, \%) of each"
        r" individual degradation variant, averaged over the 19 SSL models. Each corpus"
        r" gives a row of variant names in italic with the $\Delta$EER beneath;"
        r" additive noise shares one row of noise types across its three SNRs."
        r" Positive is worse. The cells differ enormously: ASV5's ten codecs span 84.2"
        r" points while ASVLD's six bitrates span 2.0 and ASV21 DF's eight transcodes"
        r" 2.2, so a single ``codec compression'' number would describe none of them."
        r" Severity also tracks the physical parameter wherever there is one, falling"
        r" with SNR for every noise type and rising with $RT_{60}$. Absolute EER for"
        r" each cell is given in Table~\ref{tab:degradation_spread}.}}",
        r"\label{tab:degradation_variants}", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llp{2.6cm}|" + "r" * NCOL + "}", r"\toprule",
        r"\textbf{Condition} & \textbf{Corpus} & \textbf{Degradation} &"
        r" \multicolumn{" + str(NCOL) +
        r"}{c}{\textbf{Variants (italic) and their $\Delta$EER (\%)}} \\",
        r"\midrule",
    ] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write(lines, out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m spoof_superb.analysis.degradation_appendix",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", default=None)
    args = ap.parse_args(argv)

    root = getattr(cfg, "outputs_root", "") or str(REPO_ROOT / "outputs")
    d = Path(args.in_dir or os.path.join(root, "degradation_matched"))

    per_variant = list(csv.DictReader(open(d / "per_variant_eer.csv")))
    spread = list(csv.DictReader(open(d / "cell_spread.csv")))
    _verify(per_variant)

    display = display_by_slug()
    matrix = {r["Model"]: float(r["Baseline"])
              for r in csv.DictReader(open(d / "eer_matrix.csv"))}
    baselines = {}
    for slug in {r["Model"] for r in per_variant}:
        name = display.get(slug, slug)
        if name not in matrix:
            raise SystemExit(f"FATAL: no Baseline for {slug} ({name})")
        baselines[slug] = matrix[name]

    table_composition(d / "tab_degradation_cells.tex")
    table_spread(spread, d / "tab_degradation_spread.tex")
    table_variants(per_variant, baselines, d / "tab_degradation_variants.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
