"""R8 analysis against the pre-registration in VERIFIED_NUMBERS.md section 3e.

Pre-stated criteria (written before results):
  Supports: jpi_rel scales ~linearly in eps (ratio eps0.2/eps0.05 in [3, 5])
            AND stop-grad response < Full response.
  Refutes:  flat in eps, or stop-grad >= Full.
Metric: jpi_rel_eps* averaged over late-training log points, per seed.
Also: belief effective rank, Full vs stop-grad vs No-Aux (no pre-commitment,
descriptive).
"""
import glob
import json

import numpy as np

EPS = ["0.05", "0.1", "0.2"]


def late(vals):
    n = len(vals)
    return float(np.mean(vals[int(0.6 * n):])) if n else np.nan


def per_run(path):
    d = json.load(open(path))
    gd = d["gradient_decomp"]
    out = {}
    for e in EPS:
        k = f"jpi_rel_eps{e}"
        vals = [x[k] for x in gd if k in x and np.isfinite(x[k])]
        out[e] = late(vals)
    ranks = [x["belief_effective_rank"] for x in gd if "belief_effective_rank" in x]
    out["rank_late"] = late(ranks)
    r = np.array(d["rewards"])
    out["f50"] = float(r[-3200:].mean())
    return out


print(f"{'cond':10s} {'jpi@.05':>9s} {'jpi@.1':>8s} {'jpi@.2':>8s} {'lin .2/.05':>11s} {'rank':>7s} {'F50':>7s}")
summary = {}
for cond in ["full", "stopgrad", "no_aux"]:
    rows = [per_run(f) for f in sorted(glob.glob(f"results/R8_jpi_rank/R8_{cond}_seed*.json"))]
    agg = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    summary[cond] = agg
    j5, j1, j2 = (np.nanmean(agg[e]) for e in EPS)
    lin = j2 / j5 if j5 and np.isfinite(j5) and j5 > 0 else float("nan")
    print(f"{cond:10s} {j5:9.3f} {j1:8.3f} {j2:8.3f} {lin:11.2f} "
          f"{np.nanmean(agg['rank_late']):7.1f} {np.mean(agg['f50']):7.1f}")

print()
print("=== Verdicts against pre-registration (section 3e) ===")
full = summary["full"]
sg = summary["stopgrad"]
j5f = np.nanmean(full["0.05"])
linf = np.nanmean(full["0.2"]) / j5f if j5f > 0 else float("nan")
print(f"1. Linearity (Full): eps0.2/eps0.05 = {linf:.2f}  (pre-registered support band [3, 5])")
j_full = np.nanmean(full["0.1"])
j_sg = np.nanmean(sg["0.1"])
print(f"2. Pathway contrast at eps=0.1: Full {j_full:.3f} vs stop-grad {j_sg:.3f} "
      f"({'stop-grad SMALLER: supports' if j_sg < j_full else 'stop-grad NOT smaller: refutes/weakens'})")
print("   per-seed Full @0.1:", " ".join(f"{x:.3f}" for x in full["0.1"]))
print("   per-seed sg   @0.1:", " ".join(f"{x:.3f}" for x in sg["0.1"]))
print(f"3. Rank (descriptive): Full {np.nanmean(full['rank_late']):.1f}, "
      f"stop-grad {np.nanmean(sg['rank_late']):.1f}, "
      f"No-Aux {np.nanmean(summary['no_aux']['rank_late']):.1f}")
