"""R9-R12 analysis, strictly against the pre-registrations in
VERIFIED_NUMBERS.md section 3e (written before results).

References (from the fact base):
  No-Aux n=20: 469.91 +/- 2.86      Full n=20: 467.99 +/- 2.96
  Drift gate tau=0.10 n=5: 471.90 +/- 1.93
"""
import glob
import json

import numpy as np

NOAUX_MEAN, NOAUX_SD = 469.91, 2.86
FULL_MEAN = 467.99
GATE_MEAN, GATE_SD = 471.90, 1.93


def f50s(pat):
    return np.array([float(np.array(json.load(open(f))["rewards"])[-3200:].mean())
                     for f in sorted(glob.glob(pat))])


def show(name, v):
    print(f"  {name:22s} n={len(v)}  {v.mean():7.2f} +/- {v.std(ddof=1):5.2f}   "
          f"per-seed: {' '.join(f'{x:.0f}' for x in sorted(v))}   sub-460: {int((v < 460).sum())}")


print("=== R9: aux-task variants (pre-reg: latent deficit, recon none) ===")
lat = f50s("results/R9_auxdef/R9_latent_seed*.json")
rec = f50s("results/R9_auxdef/R9_recon_seed*.json")
show("latent (drifting)", lat)
show("recon (stationary)", rec)
print(f"  deficits vs No-Aux {NOAUX_MEAN}: latent {NOAUX_MEAN - lat.mean():+.2f}, "
      f"recon {NOAUX_MEAN - rec.mean():+.2f}")
verdict = ("SUPPORTS (latent deficit > recon deficit, recon ~none)"
           if (NOAUX_MEAN - lat.mean()) > (NOAUX_MEAN - rec.mean()) + 0.5
           and (NOAUX_MEAN - rec.mean()) < 1.5
           else "check against pre-reg wording")
print(f"  verdict: {verdict}")

print()
print("=== R10: random gate p=0.70 (pre-reg: within gate band -> duty cycle) ===")
rg = f50s("results/R10_randgate/R10_rand070_seed*.json")
show("random gate 0.70", rg)
print(f"  drift gate tau=0.10 reference: {GATE_MEAN} +/- {GATE_SD}")
in_band = abs(rg.mean() - GATE_MEAN) <= (GATE_SD + rg.std(ddof=1))
print(f"  verdict: {'DUTY CYCLE EXPLAINS THE GATE (bands overlap): gate retired as mechanism evidence' if in_band else 'random gate clearly below: trigger timing matters'}")

print()
print("=== R11: schedules (pre-reg: all within/above No-Aux band, no sub-460) ===")
ok = True
for s in ["cosine", "exp", "kl_adaptive"]:
    v = f50s(f"results/R11_schedules/R11_{s}_seed*.json")
    show(s, v)
    if (v < 460).any() or v.mean() < FULL_MEAN - 1.0:
        ok = False
print(f"  verdict: {'SUPPORTS: fix family robust to schedule choice' if ok else 'WEAKENED: see arms above'}")

print()
print("=== R12: SMAX belief rank (pre-reg: Full << No-Aux -> collapse; comparable -> disfavored) ===")
for cond in ["full", "no_aux"]:
    ranks, f50 = [], []
    for f in sorted(glob.glob(f"results/R12_smax_rank/R12_{cond}_seed*.json")):
        d = json.load(open(f))
        rk = [x["belief_effective_rank"] for x in d["gradient_decomp"] if "belief_effective_rank" in x]
        n = len(rk)
        ranks.append(float(np.mean(rk[int(0.6 * n):])))
        f50.append(float(np.array(d["rewards"])[-3200:].mean()))
    print(f"  {cond:8s} late rank {np.mean(ranks):6.2f} +/- {np.std(ranks, ddof=1):5.2f}   "
          f"per-seed: {' '.join(f'{x:.1f}' for x in ranks)}   F50 {np.mean(f50):.2f}")
