"""Compute bootstrap CIs for ExpA target-source distinguishing test."""
import json
import glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Frozen-target data
frozen = []
for f in sorted(glob.glob(str(ROOT / "results" / "frozen_target_distinguishing" / "frozen_tgt_full_seed*.json"))):
    d = json.load(open(f))
    r = d.get("rewards") or d.get("episode_rewards") or d.get("sparse_rewards")
    if r is None:
        print(f"no rewards key in {f}: {list(d.keys())[:8]}")
        continue
    frozen.append(float(np.mean(r[-50:])))
frozen = np.array(frozen)
print(f"Frozen n={len(frozen)}: mean={frozen.mean():.2f}  std={frozen.std(ddof=1):.2f}")

# Reference from canonical Phase 2
ph = json.load(open(ROOT / "results" / "canonical_phase2.json"))
REF_FULL = np.array(ph["configs"]["A_full"]["final50_per_seed"])
REF_NOAUX = np.array(ph["configs"]["A_no_aux"]["final50_per_seed"])
print(f"Full (n={len(REF_FULL)}): mean={REF_FULL.mean():.2f}  std={REF_FULL.std(ddof=1):.2f}")
print(f"NoAux (n={len(REF_NOAUX)}): mean={REF_NOAUX.mean():.2f}  std={REF_NOAUX.std(ddof=1):.2f}")


def cohen_d(a, b):
    p = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / p if p > 0 else 0


def boot_ci(a, b, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        sa = aa.std(ddof=1) if len(aa) > 1 else 0
        sb = bb.std(ddof=1) if len(bb) > 1 else 0
        p = np.sqrt(((len(aa) - 1) * sa ** 2 + (len(bb) - 1) * sb ** 2) / max(len(aa) + len(bb) - 2, 1))
        if p > 1e-9:
            ds.append((aa.mean() - bb.mean()) / p)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


print()
print("Pairwise (a, b): d = Cohen's d, direction is a - b. Positive d => a > b.")
for lbl, a in [("Frozen", frozen)]:
    for lbl2, b in [("NoAux", REF_NOAUX), ("Full", REF_FULL)]:
        d = cohen_d(a, b)
        lo, hi = boot_ci(a, b)
        cross = " (CI crosses 0)" if lo < 0 < hi else ""
        print(f"  {lbl:8s} vs {lbl2:6s}  d={d:+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]{cross}")

# Full vs NoAux for reference
d = cohen_d(REF_NOAUX, REF_FULL)
lo, hi = boot_ci(REF_NOAUX, REF_FULL)
cross = " (CI crosses 0)" if lo < 0 < hi else ""
print(f"  {'NoAux':8s} vs {'Full':6s}  d={d:+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]{cross}")
