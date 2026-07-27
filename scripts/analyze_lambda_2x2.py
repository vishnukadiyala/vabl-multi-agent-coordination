"""Analyze the lambda x stationarity 2x2 (the mechanism-discrimination test).

Cells:
  lambda=0.05 x drifting   = canonical Full      (fresh n=20: 467.99 +/- 2.96)
  lambda=0.05 x frozen     = frozen-target arm    (fresh n=10: 468.8 +/- 2.5)
  lambda=0.95 x drifting   = hi_drift (new, n=5)
  lambda=0.95 x frozen     = hi_frozen (new, n=5)

Directional account: damage at high lambda requires drift -> hi_drift much worse
than hi_frozen. Magnitude account: both degrade alike. Also check LEARNING
CURVES for the high-lambda cells: "never learned" vs "learned then collapsed"
are different phenomena (late-training interference is the paper's claim).
"""
import glob
import json

import numpy as np

ENVS = 64


def load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        r = np.array(json.load(open(f))["rewards"])
        out.append(r)
    return out


def phase_means(r):
    """Mean reward in consecutive fifths of training."""
    n = len(r)
    return [float(r[i * n // 5:(i + 1) * n // 5].mean()) for i in range(5)]


print("=== 2x2 Final50 by cell ===")
for name, pat in [
    ("hi_drift  (l=0.95, drifting)", "results/lambda_stationarity_2x2/L2x2_hi_drift_seed*.json"),
    ("hi_frozen (l=0.95, frozen)  ", "results/lambda_stationarity_2x2/L2x2_hi_frozen_seed*.json"),
]:
    runs = load(pat)
    f50 = np.array([float(r[-3200:].mean()) for r in runs])
    best = np.array([float(np.convolve(r, np.ones(640) / 640, mode="valid").max()) for r in runs])
    print(f"  {name} n={len(runs)}  Final50 {f50.mean():7.1f} +/- {f50.std(ddof=1):5.1f}"
          f"   smoothed-peak {best.mean():7.1f}")
    print(f"      per-seed F50:  {' '.join(f'{x:.0f}' for x in sorted(f50))}")
    print(f"      per-seed peak: {' '.join(f'{x:.0f}' for x in sorted(best))}")

print()
print("reference cells (from VERIFIED_NUMBERS):")
print("  lo_drift  (l=0.05, drifting) n=20  Final50 467.99 +/- 2.96")
print("  lo_frozen (l=0.05, frozen)   n=10  Final50 468.8  +/- 2.5")

print()
print("=== Learning-curve shape, high-lambda cells (fifths of training) ===")
for name, pat in [
    ("hi_drift", "results/lambda_stationarity_2x2/L2x2_hi_drift_seed*.json"),
    ("hi_frozen", "results/lambda_stationarity_2x2/L2x2_hi_frozen_seed*.json"),
]:
    print(f"  {name}:")
    for i, r in enumerate(load(pat)):
        pm = phase_means(r)
        shape = "learned-then-declined" if max(pm[:3]) > pm[4] + 20 else (
            "never-learned" if max(pm) < 200 else "learned-and-held")
        print(f"    seed{i}: {' '.join(f'{x:6.0f}' for x in pm)}   [{shape}]")
