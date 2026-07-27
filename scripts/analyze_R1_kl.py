"""R1 analysis: direct Sigma_pi (consecutive-policy KL) vs cosine-std vs Final50.

Answers yGKw Q1 / fXvf Q6: measure target-policy drift directly and correlate
it with the gradient-cosine diagnostic and late-training performance across
seeds and conditions.

Usage: python scripts/analyze_R1_kl.py [results_dir]
"""
import glob
import json
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results/R1_kl_instrumented"
CONDS = ["full", "no_aux", "stopgrad", "frozen"]
FINAL_WINDOW = 3200  # 50 iterations x 64 envs, matches the paper's Final50


def per_run(path):
    d = json.load(open(path))
    r = np.array(d["rewards"])
    f50 = r[-FINAL_WINDOW:].mean()
    kl = [x["policy_kl"] for x in d.get("policy_kl", [])]
    n = len(kl)
    kl_late = np.mean(kl[int(0.6 * n):]) if n else np.nan
    cos = [g["cosine"] for g in d.get("gradient_decomp", []) if np.isfinite(g["cosine"])]
    nc = len(cos)
    cos_std_late = np.std(cos[int(0.5 * nc):]) if nc > 4 else np.nan
    return kl_late, cos_std_late, f50


def main():
    summary = {}
    print(f"{'cond':10s} {'lateKL(e-3)':>16s} {'cos_std':>8s} {'Final50 (mean+/-std)':>22s} n")
    for c in CONDS:
        rows = [per_run(f) for f in sorted(glob.glob(f"{RES}/R1_{c}_seed*.json"))]
        if not rows:
            continue
        kls, cstds, f50s = map(np.array, zip(*rows))
        summary[c] = (kls, cstds, f50s)
        print(f"{c:10s} {1e3*np.nanmean(kls):9.3f}+/-{1e3*np.nanstd(kls):5.3f}"
              f" {np.nanmean(cstds):8.3f}"
              f" {np.mean(f50s):11.1f}+/-{np.std(f50s):5.1f}  {len(rows)}")

    aux_on = [c for c in ("full", "stopgrad", "frozen") if c in summary]
    allkl = np.concatenate([summary[c][0] for c in aux_on])
    allcs = np.concatenate([summary[c][1] for c in aux_on])
    allf = np.concatenate([summary[c][2] for c in aux_on])
    m = np.isfinite(allcs) & np.isfinite(allkl)
    print(f"\nAcross aux-ON runs (n={int(m.sum())}):")
    print(f"  corr(late KL drift, cosine-std): rho = {np.corrcoef(allkl[m], allcs[m])[0, 1]:.3f}")
    print(f"  corr(cosine-std, Final50):       rho = {np.corrcoef(allcs[m], allf[m])[0, 1]:.3f}")
    print(f"  corr(late KL drift, Final50):    rho = {np.corrcoef(allkl[m], allf[m])[0, 1]:.3f}")

    for c in CONDS:
        if c in summary:
            kls, cstds, f50s = summary[c]
            print(f"\n{c} per-seed Final50: " + " ".join(f"{x:.1f}" for x in f50s))


if __name__ == "__main__":
    main()
