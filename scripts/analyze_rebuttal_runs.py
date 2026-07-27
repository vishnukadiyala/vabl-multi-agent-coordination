"""Full analysis of the rebuttal run chain (R1-R5).

Reports, per condition: Final50 mean +/- std, per-seed values, collapse
incidence (Final50 < 460), late-training KL drift, between-task cosine-std,
and the per-task self-cosine direction-stability diagnostics.

The mechanism's rescue prediction for the self-cosine diagnostic: aux_self_cos
is LOW/unstable under co-learning targets and HIGH/stable under frozen
targets, while policy_self_cos is similar across conditions.

Usage: python scripts/analyze_rebuttal_runs.py
"""
import glob
import json

import numpy as np

COLLAPSE = 460.0
FINAL_WINDOW = 3200


def load(path):
    d = json.load(open(path))
    r = np.array(d["rewards"])
    out = {"f50": float(r[-FINAL_WINDOW:].mean())}
    kl = [x["policy_kl"] for x in d.get("policy_kl", [])]
    n = len(kl)
    out["kl_late"] = float(np.mean(kl[int(0.6 * n):])) if n else np.nan
    gd = d.get("gradient_decomp", [])
    cos = [g["cosine"] for g in gd if np.isfinite(g["cosine"])]
    nc = len(cos)
    out["cos_std"] = float(np.std(cos[int(0.5 * nc):])) if nc > 4 else np.nan
    psc = [g["policy_self_cos"] for g in gd if "policy_self_cos" in g]
    asc = [g["aux_self_cos"] for g in gd
           if "aux_self_cos" in g and np.isfinite(g["aux_self_cos"])]
    np_l = len(psc)
    na_l = len(asc)
    out["policy_self_cos"] = float(np.mean(psc[int(0.5 * np_l):])) if np_l > 4 else np.nan
    out["aux_self_cos"] = float(np.mean(asc[int(0.5 * na_l):])) if na_l > 4 else np.nan
    gate = d.get("drift_gate", [])
    out["gate_frac"] = (float(np.mean([g["gate_active"] for g in gate]))
                        if gate else np.nan)
    return out


def group(pattern):
    rows = [load(f) for f in sorted(glob.glob(pattern))]
    if not rows:
        return None
    g = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    g["n"] = len(rows)
    return g


def line(name, g):
    if g is None:
        print(f"{name:22s} (no files)")
        return
    f50 = g["f50"]
    ncol = int((f50 < COLLAPSE).sum())
    extras = []
    if np.isfinite(g["kl_late"]).any():
        extras.append(f"KL {1e3*np.nanmean(g['kl_late']):.2f}e-3")
    if np.isfinite(g["cos_std"]).any():
        extras.append(f"cosStd {np.nanmean(g['cos_std']):.3f}")
    if np.isfinite(g["policy_self_cos"]).any():
        extras.append(f"selfcos P {np.nanmean(g['policy_self_cos']):.3f}"
                      f" A {np.nanmean(g['aux_self_cos']):.3f}")
    if np.isfinite(g["gate_frac"]).any():
        extras.append(f"gated {100*np.nanmean(g['gate_frac']):.0f}%")
    print(f"{name:22s} n={g['n']:2d} F50 {f50.mean():6.1f}+/-{f50.std():4.1f} "
          f"collapses {ncol}/{g['n']}  " + "  ".join(extras))
    print(f"{'':22s} per-seed: {' '.join(f'{x:.0f}' for x in sorted(f50))}")


print("=== R1+R5 pooled (fresh instrumented, seeds 0-9) ===")
for c in ["full", "no_aux", "stopgrad", "frozen"]:
    line(c, group(f"results/R1_kl_instrumented/R1_{c}_seed*.json"))

print("\n=== R3 snapshot-lag continuum (soft targets) ===")
for tag in ["lag1", "lag25", "lag100", "lagfrozen"]:
    line(tag, group(f"results/R3_snapshot_lag/R3_{tag}_seed*.json"))

print("\n=== R4 drift gate ===")
for tau in ["0.10", "0.15"]:
    line(f"gate tau={tau}", group(f"results/R4_drift_gate/R4_gate_tau{tau}_seed*.json"))

print("\n=== R2 gradient surgery (reference) ===")
for c in ["pcgrad", "gradnorm"]:
    line(c, group(f"results/R2_grad_surgery/R2_{c}_seed*.json"))

print("\n=== Self-cos mechanism test (seeds 5-9 only, have the diagnostic) ===")
for c in ["full", "frozen", "no_aux"]:
    rows = [load(f) for f in sorted(glob.glob(f"results/R1_kl_instrumented/R1_{c}_seed[5-9].json"))]
    a = [r["aux_self_cos"] for r in rows if np.isfinite(r["aux_self_cos"])]
    p = [r["policy_self_cos"] for r in rows if np.isfinite(r["policy_self_cos"])]
    if a or p:
        print(f"{c:10s} aux_self_cos {np.mean(a) if a else float('nan'):.3f}"
              f"  policy_self_cos {np.mean(p) if p else float('nan'):.3f}  (n={len(rows)})")
