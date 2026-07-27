"""Gap-consistency table + pooled collapse census across every run set.

The stable finding: the Full-vs-No-Aux mean deficit is consistent across four
code/stack configurations; the canonical n=5 magnitude is the outlier, driven
by two rare collapse events.
"""
import glob
import json

import numpy as np


def f50s(pat):
    return [float(np.array(json.load(open(f))["rewards"])[-3200:].mean())
            for f in sorted(glob.glob(pat))]


print("=== Gap consistency across every set ===")
for name, fp, npat in [
    ("fresh old-stack n=20 (current code, jax 0.6.2)",
     "results/R1_kl_instrumented/R1_full_seed*.json",
     "results/R1_kl_instrumented/R1_no_aux_seed*.json"),
    ("V2 stack n=20 (current code, jax 0.10.2)",
     "results/V2_verification/V2_full_seed*.json",
     "results/V2_verification/V2_no_aux_seed*.json"),
    ("pre-VAE replication n=5 (code 8e0c854, jax 0.6.2)",
     "results/prevae_replication/prevae_A_full_seed*.json",
     "results/prevae_replication/prevae_A_no_aux_seed*.json"),
]:
    a, b = np.array(f50s(fp)), np.array(f50s(npat))
    print(f"  {name:50s} Full {a.mean():6.2f}  NoAux {b.mean():6.2f}  gap {b.mean()-a.mean():5.2f}")
print("  canonical Phase2 n=5 (code 8e0c854, jax 0.6.2)      Full 463.21  NoAux 473.45  gap 10.24  (2 collapses drive it)")

print()
print("=== Pooled plain-Full collapse census (Final50<460, descriptive) ===")
pools = [
    ("canonical Phase2 A_full", [469.1, 456.5, 450.6, 465.6, 474.2]),
    ("canonical expB full", f50s("results/expB_gradient_decomp/expB_full_seed*.json")),
    ("fresh old-stack full n=20", f50s("results/R1_kl_instrumented/R1_full_seed*.json")),
    ("V2 full n=20", f50s("results/V2_verification/V2_full_seed*.json")),
    ("pre-VAE full n=5", f50s("results/prevae_replication/prevae_A_full_seed*.json")),
]
tc = tn = 0
for name, v in pools:
    v = np.array(v)
    c = int((v < 460).sum())
    tc += c
    tn += len(v)
    print(f"  {name:32s} {c}/{len(v)}")
print(f"  POOLED plain Full: {tc}/{tn} = {100 * tc / tn:.1f}%")

sh = 0
clean = True
for pat in [
    "results/R1_kl_instrumented/R1_no_aux_seed*.json",
    "results/R1_kl_instrumented/R1_stopgrad_seed*.json",
    "results/R1_kl_instrumented/R1_frozen_seed*.json",
    "results/V2_verification/V2_no_aux_seed*.json",
    "results/expB_gradient_decomp/expB_no_aux_seed*.json",
    "results/expB_gradient_decomp/expB_stopgrad_seed*.json",
    "results/expB_gradient_decomp/expB_anneal_seed*.json",
    "results/frozen_target_distinguishing/*.json",
    "results/prevae_replication/prevae_A_no_aux_seed*.json",
]:
    v = np.array(f50s(pat))
    sh += len(v)
    if len(v) and not (v >= 460).all():
        clean = False
        print(f"  NOTE: collapse found in shielded arm {pat}: min {v.min():.1f}")
print(f"  shielded/stationary arms: {'0' if clean else 'SOME'}/{sh} collapses")

print()
n = len(glob.glob("results/**/*seed*.json", recursive=True))
print(f"total per-seed result JSONs under results/: {n}")
