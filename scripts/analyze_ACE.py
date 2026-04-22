"""Analyze ExpA (frozen-target), ExpC (noise-target), ExpE (SMAX lam=0.025)
against canonical references for the NeurIPS paper.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"


def load_stats(glob):
    fs = sorted(ROOT.glob(glob))
    f50s, bests = [], []
    for f in fs:
        d = json.load(open(f))
        r = d["rewards"]
        f50s.append(float(np.mean(r[-50:])))
        bests.append(float(max(r)))
    return f50s, bests


def cohen_d(m1, s1, n1, m2, s2, n2):
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled


def bootstrap_ci_d(a, b, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    ds = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        sa = aa.std(ddof=1) if len(aa) > 1 else 0.0
        sb = bb.std(ddof=1) if len(bb) > 1 else 0.0
        pooled = np.sqrt(((len(aa) - 1) * sa ** 2 + (len(bb) - 1) * sb ** 2) /
                         max(len(aa) + len(bb) - 2, 1))
        if pooled > 1e-9:
            ds.append((aa.mean() - bb.mean()) / pooled)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def report(name, f50s, bests):
    a = np.array(f50s)
    print(f"{name:32s} n={len(a)}  Final50 = {a.mean():7.2f} +/- {a.std(ddof=1):5.2f}  "
          f"Best = {np.mean(bests):7.2f}")
    print(f"  per-seed F50: {[round(x,2) for x in f50s]}")


# ============================================================
# OVERCOOKED AA
# ============================================================
print("=" * 84)
print("OVERCOOKED ASYMMETRIC ADVANTAGES")
print("=" * 84)

ph = json.load(open(ROOT / "canonical_phase2.json"))["configs"]
REFS_AA = {
    "Full (attn+aux)": ph["A_full"]["final50_per_seed"],
    "No Aux (attn only)": ph["A_no_aux"]["final50_per_seed"],
    "Neither (mean only)": ph["A_neither"]["final50_per_seed"],
    "No Attn (mean+aux)": ph["A_no_attn"]["final50_per_seed"],
    "+ Stop-grad": ph["B_stopgrad"]["final50_per_seed"],
    "+ Anneal": ph["B_anneal"]["final50_per_seed"],
}
print("\n-- References --")
for name, arr in REFS_AA.items():
    a = np.array(arr)
    print(f"  {name:32s} n={len(a)}  {a.mean():7.2f} +/- {a.std(ddof=1):5.2f}")

# ExpA
print("\n-- ExpA: frozen-target policy (stationary aux targets) --")
A_f50, A_best = load_stats("frozen_target_distinguishing/*.json")
report("ExpA (frozen targets)", A_f50, A_best)
print()
print("  vs references (Cohen's d, bootstrap 95% CI; positive = ref better than ExpA)")
for name, arr in REFS_AA.items():
    a = np.array(arr); e = np.array(A_f50)
    d = cohen_d(a.mean(), a.std(ddof=1), len(a), e.mean(), e.std(ddof=1), len(e))
    lo, hi = bootstrap_ci_d(arr, A_f50)
    flag = "" if (lo > 0 or hi < 0) else "   (CI crosses 0)"
    print(f"    vs {name:22s}  d = {d:+.2f}  CI = [{lo:+.2f}, {hi:+.2f}]{flag}")

# ExpC
print("\n-- ExpC: uniform-random aux targets --")
C_f50, C_best = load_stats("expC_noise_targets/*.json")
report("ExpC (noise targets)", C_f50, C_best)
print()
print("  vs references (Cohen's d, bootstrap 95% CI; positive = ref better than ExpC)")
for name, arr in REFS_AA.items():
    a = np.array(arr); e = np.array(C_f50)
    d = cohen_d(a.mean(), a.std(ddof=1), len(a), e.mean(), e.std(ddof=1), len(e))
    lo, hi = bootstrap_ci_d(arr, C_f50)
    flag = "" if (lo > 0 or hi < 0) else "   (CI crosses 0)"
    print(f"    vs {name:22s}  d = {d:+.2f}  CI = [{lo:+.2f}, {hi:+.2f}]{flag}")

# ============================================================
# SMAX 3v3
# ============================================================
print()
print("=" * 84)
print("SMAX 3v3")
print("=" * 84)

smax = json.load(open(ROOT / "canonical_smax.json"))
try:
    sfx = json.load(open(ROOT / "canonical_smax_fixes.json"))["configs"]
except Exception:
    sfx = {}

def smax_get(d, key):
    node = d.get(key, {})
    return node.get("final_per_seed") or node.get("final50_per_seed") or []

REFS_SMAX = {
    "Full (lam=0.05)": smax_get(smax, "full"),
    "No Aux (lam=0)": smax_get(smax, "no_aux"),
    "+ Stop-grad": smax_get(sfx, "stopgrad"),
    "+ Anneal": smax_get(sfx, "anneal"),
}
print("\n-- References --")
for name, arr in REFS_SMAX.items():
    if arr:
        a = np.array(arr)
        print(f"  {name:32s} n={len(a)}  {a.mean():7.2f} +/- {a.std(ddof=1):5.2f}")

# ExpE
print("\n-- ExpE: intermediate lambda=0.025 --")
E_f50, E_best = load_stats("expE_smax_intermediate_lambda/*.json")
report("ExpE (lam=0.025)", E_f50, E_best)
print()
print("  vs references (Cohen's d, bootstrap 95% CI; positive = ref better than ExpE)")
for name, arr in REFS_SMAX.items():
    if arr:
        a = np.array(arr); e = np.array(E_f50)
        d = cohen_d(a.mean(), a.std(ddof=1), len(a), e.mean(), e.std(ddof=1), len(e))
        lo, hi = bootstrap_ci_d(arr, E_f50)
        flag = "" if (lo > 0 or hi < 0) else "   (CI crosses 0)"
        print(f"    vs {name:22s}  d = {d:+.2f}  CI = [{lo:+.2f}, {hi:+.2f}]{flag}")
