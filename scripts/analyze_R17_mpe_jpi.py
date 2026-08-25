"""R17 analysis: J_pi sensitivity on MPE simple_spread, Full arm, 5 seeds.

Same metric definition as analyze_R8_jpi.py (Overcooked AA) so the two are
directly comparable: jpi_rel_eps* averaged over the late 40% of log points,
per seed. Pre-registered support band for linearity (eps0.2 / eps0.05) is
[3, 5], as in R8.

Also reports the aux/policy gradient-norm ratio, which is the quantity the
posted final reply cited qualitatively (AA ~0.145).
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
    ratios = [x["norm_aux"] * x["aux_lambda_eff"] / x["norm_policy"]
              for x in gd if x.get("norm_policy")]
    out["aux_policy_ratio"] = late(ratios)
    r = np.array(d["rewards"])
    out["f50"] = float(r[-int(0.05 * len(r)):].mean())
    return out


files = sorted(glob.glob("results/R17_mpe_jpi/R17_full_seed*.json"))
rows = [per_run(f) for f in files]
agg = {k: np.array([r[k] for r in rows]) for k in rows[0]}

print(f"n = {len(rows)} seeds\n")
print(f"{'seed':>4s} {'jpi@.05':>9s} {'jpi@.1':>9s} {'jpi@.2':>9s} "
      f"{'lin .2/.05':>11s} {'aux/pol':>9s} {'F50':>8s}")
for i, r in enumerate(rows):
    lin = r["0.2"] / r["0.05"] if r["0.05"] > 0 else float("nan")
    print(f"{i:4d} {r['0.05']:9.3f} {r['0.1']:9.3f} {r['0.2']:9.3f} "
          f"{lin:11.2f} {r['aux_policy_ratio']:9.4f} {r['f50']:8.2f}")


def ms(a):
    return f"{np.nanmean(a):.3f} +/- {np.nanstd(a, ddof=0):.3f}"


print()
for e in EPS:
    print(f"jpi_rel @ eps={e:<5s} : {ms(agg[e])}   "
          f"(range {np.nanmin(agg[e]):.3f} - {np.nanmax(agg[e]):.3f})")
lins = agg["0.2"] / agg["0.05"]
print(f"linearity .2/.05    : {ms(lins)}   "
      f"(pre-registered support band [3, 5]; per-seed "
      f"{' '.join(f'{x:.2f}' for x in lins)})")
print(f"aux/policy norm     : {np.nanmean(agg['aux_policy_ratio']):.4f} "
      f"+/- {np.nanstd(agg['aux_policy_ratio']):.4f}  (AA reference ~0.145)")
print(f"F50 reward          : {np.mean(agg['f50']):.2f} "
      f"+/- {np.std(agg['f50']):.2f}")

print()
print("=== Verdict ===")
lin_mean = float(np.nanmean(lins))
in_band = [3 <= x <= 5 for x in lins]
print(f"Linearity mean {lin_mean:.2f}; seeds in band [3,5]: "
      f"{sum(in_band)}/{len(in_band)}")
cv = float(np.nanstd(agg["0.1"]) / np.nanmean(agg["0.1"]))
print(f"Seed dispersion at eps=0.1: CV = {cv:.2f} "
      f"({'HIGH - report as open boundary case' if cv > 0.5 else 'acceptable'})")
