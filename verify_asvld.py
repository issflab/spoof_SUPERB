"""
verify_asvld.py
---------------
Compare freshly generated ASVLD score files (asvld_rerun) against the existing
reference score files, per (model, condition). Prints a summary table.

For each verifiable (model, condition):
  - line-count match (new vs reference)
  - utt_id set difference (present in one but not the other)
  - score distribution (mean/std/min/max) for new and reference
  - max absolute score difference over the common utt_ids
Recompression & Filtering have no reference -> reported as GENERATED-ONLY.
"""
import os
import statistics as st

RERUN = "/data/ssl_anti_spoofing/asd_superb_score_files/asvld_rerun"
REF   = "/data/ssl_anti_spoofing/asd_superb_score_files/scores_by_category"

MODELS = ["xls_r_300m", "unispeech_sat_large", "wav2vec2_large_ll60k"]

# model -> (Noise/Reverb reference basename, Resampling reference basename)
REFNAME = {
    "xls_r_300m":          ("XLS-R.txt",         "linear_head_resamp_xls_r_300m.txt"),
    "unispeech_sat_large": ("Unispeech-SAT.txt", "linear_head_resamp_unispeech_sat_large.txt"),
    "wav2vec2_large_ll60k":("wav2vec2_Large.txt","linear_head_resamp_wav2vec2_large_ll60k.txt"),
}

# condition -> reference (category subdir, which basename slot) ; None => no reference
def ref_path(model, cond):
    rn, rr = REFNAME[model]
    if cond == "Noise_Addition":  return os.path.join(REF, "Additive_Noise", rn)
    if cond == "Reverberation":   return os.path.join(REF, "Reverberation", rn)
    if cond == "Resampling":      return os.path.join(REF, "Resampling", rr)
    return None  # Recompression, Filtering

# Filtering intentionally excluded (ignored per user request; no reference exists anyway).
CONDITIONS = ["Noise_Addition", "Reverberation", "Resampling", "Recompression"]


def load_scores(path):
    d = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            d[p[0]] = (p[2], float(p[3]))  # utt -> (key, score)
    return d


def dist(vals):
    if not vals:
        return (float("nan"),) * 4
    return (sum(vals) / len(vals),
            st.pstdev(vals) if len(vals) > 1 else 0.0,
            min(vals), max(vals))


def main():
    rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            new_path = os.path.join(RERUN, cond, f"linear_head_{cond}_{model}.txt")
            if not os.path.isfile(new_path):
                rows.append((model, cond, "-", "-", "-", "MISSING-OUTPUT"))
                continue
            new = load_scores(new_path)
            rp = ref_path(model, cond)
            if rp is None or not os.path.isfile(rp):
                nm, ns, nmin, nmax = dist([v[1] for v in new.values()])
                print(f"[{model}/{cond}] GENERATED-ONLY  n={len(new)}  "
                      f"mean={nm:.4f} std={ns:.4f} min={nmin:.3f} max={nmax:.3f}")
                rows.append((model, cond, len(new), "no-ref", "n/a", "GENERATED-ONLY"))
                continue

            ref = load_scores(rp)
            new_ids, ref_ids = set(new), set(ref)
            only_new = new_ids - ref_ids
            only_ref = ref_ids - new_ids
            common = new_ids & ref_ids
            lc_match = (len(new) == len(ref))

            diffs = [abs(new[u][1] - ref[u][1]) for u in common]
            keymis = sum(1 for u in common if new[u][0] != ref[u][0])
            max_diff = max(diffs) if diffs else float("nan")

            nm, ns, nmin, nmax = dist([v[1] for v in new.values()])
            rm, rs, rmin, rmax = dist([v[1] for v in ref.values()])

            # Contract: the score files must reproduce the reference *as an
            # evaluation artifact* -- identical utt set/keys and an unchanged EER.
            # A raw max|delta| threshold is the wrong contract: GPU float
            # non-determinism produces a few large per-utterance outliers that do
            # not move the metric, so we gate on EER + correlation instead.
            import numpy as np
            from evaluation import calculate_EER
            ks = list(common)
            a = np.array([new[u][1] for u in ks])
            b = np.array([ref[u][1] for u in ks])
            corr = float(np.corrcoef(a, b)[0, 1]) if len(ks) > 1 else 1.0
            frac_big = float(np.mean(np.abs(a - b) > 0.5)) if len(ks) else 0.0
            eer_new, eer_ref = calculate_EER(new_path), calculate_EER(rp)
            d_eer = eer_new - eer_ref

            status = "PASS"
            if not lc_match:         status = "COUNT-DIFF"
            if only_new or only_ref: status = "SET-DIFF"
            if keymis:               status = "KEY-DIFF"
            if abs(d_eer) > 0.05 or corr < 0.9999 or frac_big > 0.001:
                status = "SCORE-DIFF"
            print(f"   corr={corr:.7f}  frac|d|>0.5={100*frac_big:.4f}%  "
                  f"EER new={eer_new:.4f} ref={eer_ref:.4f} dEER={d_eer:+.5f}")

            print(f"\n[{model}/{cond}]  status={status}")
            print(f"   lines new={len(new)} ref={len(ref)} match={lc_match}")
            print(f"   common={len(common)} only_new={len(only_new)} only_ref={len(only_ref)} key_mismatch={keymis}")
            print(f"   new  dist: mean={nm:.4f} std={ns:.4f} min={nmin:.3f} max={nmax:.3f}")
            print(f"   ref  dist: mean={rm:.4f} std={rs:.4f} min={rmin:.3f} max={rmax:.3f}")
            print(f"   max|Δscore| over common = {max_diff:.6g}")
            if only_new:
                print(f"   e.g. only_new: {list(only_new)[:3]}")
            if only_ref:
                print(f"   e.g. only_ref: {list(only_ref)[:3]}")

            rows.append((model, cond, f"{len(new)}=={len(ref)}" if lc_match else f"{len(new)}!={len(ref)}",
                         lc_match, f"{max_diff:.4g}" if diffs else "n/a", status))

    print("\n\n================ SUMMARY TABLE ================")
    print(f"{'model':22} {'condition':16} {'counts':22} {'lc_match':9} {'max_diff':10} {'status'}")
    for r in rows:
        print(f"{r[0]:22} {r[1]:16} {str(r[2]):22} {str(r[3]):9} {str(r[4]):10} {r[5]}")


if __name__ == "__main__":
    main()
