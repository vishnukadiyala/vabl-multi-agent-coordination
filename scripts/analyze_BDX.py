"""Analyze ExpB (gradient decomposition), ExpD (aux capacity scaling),
and ExpX-A (VAE belief 2x2) against canonical references.
"""
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
FN_RE = re.compile(r"(\w+?)_seed(\d+)\.json$")


def load_by_config(glob):
    """Returns dict[config_name] -> list[(seed, json_data)]."""
    out = {}
    for p in sorted(ROOT.glob(glob)):
        m = FN_RE.search(p.name)
        if not m:
            continue
        # Strip prefix like 'expB_' or 'expX_' from config name
        cfg = m.group(1)
        for pfx in ("expB_", "expC_", "expD_", "expX_", "expY_"):
            if cfg.startswith(pfx):
                cfg = cfg[len(pfx):]
                break
        out.setdefault(cfg, []).append((int(m.group(2)), json.load(open(p))))
    return out


def final50_stats(entries):
    f50s = [float(np.mean(d["rewards"][-50:])) for _, d in entries]
    bests = [float(max(d["rewards"])) for _, d in entries]
    return np.array(f50s), np.array(bests)


def cohen_d(a, b):
    ma, sa = a.mean(), a.std(ddof=1)
    mb, sb = b.mean(), b.std(ddof=1)
    pooled = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
    return (ma - mb) / pooled if pooled > 0 else 0.0


def bootstrap_ci_d(a, b, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        sa = aa.std(ddof=1) if len(aa) > 1 else 0.0
        sb = bb.std(ddof=1) if len(bb) > 1 else 0.0
        pooled = np.sqrt(((len(aa)-1)*sa**2 + (len(bb)-1)*sb**2) / max(len(aa)+len(bb)-2, 1))
        if pooled > 1e-9:
            ds.append((aa.mean() - bb.mean()) / pooled)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


# ============================================================
# Reference: canonical phase 2 on AA
# ============================================================
ph = json.load(open(ROOT / "canonical_phase2.json"))["configs"]
REF_FULL = np.array(ph["A_full"]["final50_per_seed"])
REF_NOAUX = np.array(ph["A_no_aux"]["final50_per_seed"])
REF_NOATTN = np.array(ph["A_no_attn"]["final50_per_seed"])
REF_NEITHER = np.array(ph["A_neither"]["final50_per_seed"])
REF_STOPGRAD = np.array(ph["B_stopgrad"]["final50_per_seed"])
REF_ANNEAL = np.array(ph["B_anneal"]["final50_per_seed"])


# ============================================================
# ExpB: gradient decomposition (Full VABL, 5 seeds)
# ============================================================
print("=" * 84)
print("ExpB: gradient decomposition under 4 conditions")
print("=" * 84)

B = load_by_config("expB_gradient_decomp/expB_*.json")
for cfg in ["full", "no_aux", "stopgrad", "anneal"]:
    entries = B.get(cfg, [])
    if not entries:
        print(f"  {cfg}: no runs")
        continue
    f50, best = final50_stats(entries)
    # cosine temporal std per seed (late 50%)
    cos_stds = []
    cos_late_mean = []
    for _, d in entries:
        gd = d.get("gradient_decomp", [])
        cosines = np.array([e["cosine"] for e in gd])
        k = len(cosines) // 2
        if len(cosines[k:]) > 1:
            cos_stds.append(float(np.std(cosines[k:], ddof=1)))
            cos_late_mean.append(float(np.mean(cosines[k:])))
    print(f"  {cfg:10s} n={len(entries)}  Final50={f50.mean():7.2f}+/-{f50.std(ddof=1):.2f}  "
          f"cosine late-50% std={np.mean(cos_stds) if cos_stds else 0:.3f} "
          f"mean={np.mean(cos_late_mean) if cos_late_mean else 0:+.3f}")


# ============================================================
# ExpD: aux capacity scaling
# ============================================================
print()
print("=" * 84)
print("ExpD: aux capacity scaling (Full, varying aux_hidden_dim)")
print("=" * 84)

D = load_by_config("expD_capacity_scaling/expD_*.json")
hdim_map = {}
for cfg_str, entries in D.items():
    m = re.match(r"auxhid(\d+)$", cfg_str)
    if m:
        hdim_map[int(m.group(1))] = entries

print()
print(f"  Reference: Full (hdim=64, canonical): Final50={REF_FULL.mean():.2f}+/-{REF_FULL.std(ddof=1):.2f}")
print()
for hdim in sorted(hdim_map.keys()):
    entries = hdim_map[hdim]
    f50, best = final50_stats(entries)
    d_vs_full = cohen_d(REF_FULL, f50)
    lo, hi = bootstrap_ci_d(REF_FULL, f50)
    cross = "" if (lo > 0 or hi < 0) else "  (CI crosses 0)"
    print(f"  hdim={hdim:3d}  n={len(entries)}  Final50={f50.mean():7.2f}+/-{f50.std(ddof=1):.2f}  "
          f"Best={best.mean():.2f}  d vs Full64={d_vs_full:+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]{cross}")


# ============================================================
# ExpX-A: VAE belief 2x2 on AA
# ============================================================
print()
print("=" * 84)
print("ExpX-A: VAE belief (Dynamic-Belief-style) 2x2 ablation")
print("=" * 84)

X = load_by_config("expX_vae_belief/expX_*.json")
print()
print(f"  Reference (GRU belief):")
print(f"    Full (attn+aux):      {REF_FULL.mean():7.2f}+/-{REF_FULL.std(ddof=1):.2f}  (pathological)")
print(f"    No Aux (attn only):   {REF_NOAUX.mean():7.2f}+/-{REF_NOAUX.std(ddof=1):.2f}")
print(f"    No Attn (mean+aux):   {REF_NOATTN.mean():7.2f}+/-{REF_NOATTN.std(ddof=1):.2f}")
print(f"    Neither (mean only):  {REF_NEITHER.mean():7.2f}+/-{REF_NEITHER.std(ddof=1):.2f}")
print()
print(f"  VAE belief:")
vae_f50s = {}
for cfg in ["full", "no_aux", "no_attn", "neither"]:
    entries = X.get(cfg, [])
    if not entries:
        print(f"    {cfg}: no runs")
        continue
    f50, best = final50_stats(entries)
    vae_f50s[cfg] = f50
    print(f"    {cfg:8s} (n={len(entries)}):  Final50={f50.mean():7.2f}+/-{f50.std(ddof=1):.2f}  Best={best.mean():.2f}")

# Key comparison: is VAE-Full pathological relative to VAE-NoAux?
if "full" in vae_f50s and "no_aux" in vae_f50s:
    a, b = vae_f50s["full"], vae_f50s["no_aux"]
    d = cohen_d(b, a)  # positive = NoAux better
    lo, hi = bootstrap_ci_d(b, a)
    cross = "" if (lo > 0 or hi < 0) else "  (CI crosses 0)"
    print()
    print(f"  VAE-Full vs VAE-NoAux: d={d:+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]{cross}")
    print(f"    (positive d means VAE-NoAux better than VAE-Full, i.e. pathology reproduces)")
