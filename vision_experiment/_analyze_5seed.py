"""Analyze CIFAR-100 4-config 5-seed results. Reports means, stds, and
Cohen's d (bootstrap CI) for the comparisons we cite in the paper."""
import json
import glob
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parent / "results_5seed"
CFG = ["A_full", "A_no_attn", "A_no_aux", "A_neither"]


def load(cfg):
    bests, f5s, last = [], [], []
    for f in sorted(glob.glob(str(R / f"vision_{cfg}_seed*.json"))):
        d = json.load(open(f))
        hist = d.get("test_acc_history")
        if hist is None:
            continue
        arr = list(hist)
        bests.append(float(max(arr)))
        f5s.append(float(np.mean(arr[-5:])))
        last.append(float(arr[-1]))
    return np.array(bests), np.array(f5s), np.array(last)


def cohen_d(a, b):
    pooled = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    )
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0


def bootstrap_d(a, b, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        sa = aa.std(ddof=1) if len(aa) > 1 else 0
        sb = bb.std(ddof=1) if len(bb) > 1 else 0
        p = np.sqrt(((len(aa) - 1) * sa**2 + (len(bb) - 1) * sb**2) / max(len(aa) + len(bb) - 2, 1))
        if p > 1e-9:
            ds.append((aa.mean() - bb.mean()) / p)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


stats = {c: load(c) for c in CFG}
print(f"{'config':<14}  n  {'Best (mean+/-std)':>22}  {'Final5 (mean+/-std)':>22}")
print("-" * 68)
for c in CFG:
    b, f, _ = stats[c]
    print(f"{c:<14}  {len(b)}  {b.mean():>10.2f} +/- {b.std(ddof=1):<6.2f}  {f.mean():>10.2f} +/- {f.std(ddof=1):<6.2f}")

print()
print("Primary comparison (No Aux vs Full, same attention on both):")
for metric, idx in [("Best", 0), ("Final5", 1)]:
    a = stats["A_no_aux"][idx]
    b = stats["A_full"][idx]
    d = cohen_d(a, b)
    lo, hi = bootstrap_d(a, b)
    cross = " (CI crosses 0)" if (lo < 0 < hi) else ""
    print(f"  {metric:<7s} No Aux - Full = {a.mean()-b.mean():+.2f}  d={d:+.2f}  CI=[{lo:+.2f},{hi:+.2f}]{cross}")

print()
print("2x2 table (Best):")
print(f"{'':>12}  {'attn':>10}  {'mean':>10}")
print(f"{'aux-ON':>12}  {stats['A_full'][0].mean():>10.2f}  {stats['A_no_attn'][0].mean():>10.2f}")
print(f"{'aux-OFF':>12}  {stats['A_no_aux'][0].mean():>10.2f}  {stats['A_neither'][0].mean():>10.2f}")
