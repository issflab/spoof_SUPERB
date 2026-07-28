"""
make_tsv_mlaad.py
-----------------
Emit a tab-delimited companion for each MLAAD v10 score file.

Why: the reference score format is '<utt_id> - <label> <score>', space-delimited.
MLAAD v10 introduces vendor directories whose names contain spaces
('Cartesia.ai (Sonic-3)', 'OpenAI TTS-1 HD', ...), so ~8.6% of v10 utt_ids
contain spaces. Any consumer that splits on whitespace -- e.g.
scripts/evaluate_score_file.py (pd.read_csv sep=r"\\s+") and
scripts/compute_eer_matrix.py (line.split()) -- mis-parses those rows.

The .txt files stay exactly as produced (reference-format compatible). This
writes a parallel .tsv with three tab-separated columns:
    utt_id <TAB> label <TAB> score
which pandas reads unambiguously with sep='\\t'. Purely additive; delete if
unwanted.
"""
import glob
import os
import sys

OUT_DIR = "/data/ssl_anti_spoofing/asd_superb_score_files/linear_head_MLAAD_v10"
TSV_DIR = os.path.join(OUT_DIR, "tsv")


def main():
    os.makedirs(TSV_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(OUT_DIR, "linear_head_MLAAD_v10_*.txt")))
    if not files:
        print("no score files found")
        return 1
    for src in files:
        base = os.path.basename(src)[: -len(".txt")]
        dst = os.path.join(TSV_DIR, base + ".tsv")
        n = n_space = 0
        tmp = dst + ".part"
        with open(src) as fi, open(tmp, "w") as fo:
            fo.write("utt_id\tlabel\tscore\n")
            for line in fi:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.rsplit(" ", 3)
                if len(parts) < 4:
                    continue
                utt, _dash, label, score = parts
                if " " in utt:
                    n_space += 1
                fo.write(f"{utt}\t{label}\t{score}\n")
                n += 1
        os.replace(tmp, dst)
        print(f"{base}: {n} rows ({n_space} with spaces in utt_id) -> {dst}", flush=True)
    print(f"\nwrote {len(files)} tsv files to {TSV_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
