"""Analysis for R15 (MPE Sigma_pi) and R16 (representation-drift probe).

R15: late-training consecutive-policy KL on MPE simple_spread vs the
Overcooked AA reference band (2.42-2.58e-3 nats, R1 runs).

R16: per-arm late-training feature self-cosine (representation drift,
common-mode claim) vs aux-gradient self-cosine (the arm-separating signal).

Usage:
    python scripts/analyze_R15_R16.py results/R15_mpe_kl
    python scripts/analyze_R15_R16.py results/R16_repdrift
Late window: last 20% of logged entries (matches the paper's late-training
convention; window sensitivity should be checked before quoting).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LATE_FRAC = 0.2
AA_KL_REF = (2.42e-3, 2.58e-3)


def late(vals):
    vals = list(vals)
    if not vals:
        return None
    k = max(1, int(len(vals) * LATE_FRAC))
    return np.array(vals[-k:], dtype=float)


def main(result_dir):
    result_dir = Path(result_dir)
    files = sorted(result_dir.glob("*.json"))
    if not files:
        sys.exit(f"no JSONs in {result_dir}")

    # group by arm name: R15_full_seed0.json -> full
    by_arm = defaultdict(list)
    for f in files:
        parts = f.stem.split("_seed")
        arm = parts[0].split("_", 1)[1]  # strip R15_/R16_ prefix
        by_arm[arm].append(f)

    for arm, arm_files in sorted(by_arm.items()):
        kl_late, feat_cos_late, feat_l2_late, aux_cos_late, pol_cos_late, finals = \
            [], [], [], [], [], []
        for f in arm_files:
            d = json.load(open(f))
            rw = d.get("rewards", [])
            if rw:
                finals.append(float(np.mean(rw[-3200:])))
            kl = late([e["policy_kl"] for e in d.get("policy_kl", [])])
            if kl is not None:
                kl_late.append(kl.mean())
            fd = d.get("feature_drift", [])
            fc = late([e["feat_self_cos_mean"] for e in fd])
            if fc is not None:
                feat_cos_late.append(fc.mean())
            fl = late([e["feat_rel_l2"] for e in fd])
            if fl is not None:
                feat_l2_late.append(fl.mean())
            gd = d.get("gradient_decomp", [])
            ac = late([e["aux_self_cos"] for e in gd if "aux_self_cos" in e])
            if ac is not None:
                aux_cos_late.append(ac.mean())
            pc = late([e["policy_self_cos"] for e in gd if "policy_self_cos" in e])
            if pc is not None:
                pol_cos_late.append(pc.mean())

        def fmt(xs, scale=1.0, prec=4):
            if not xs:
                return "n/a"
            xs = np.array(xs) * scale
            return f"{xs.mean():.{prec}f} +/- {xs.std():.{prec}f} (n={len(xs)})"

        print(f"\n== {arm} ({len(arm_files)} seeds) ==")
        print(f"  Final (last 3200 ep) : {fmt(finals, prec=2)}")
        print(f"  late policy KL (nats): {fmt(kl_late, prec=6)}")
        if kl_late:
            m = float(np.mean(kl_late))
            lo, hi = AA_KL_REF
            rel = m / (0.5 * (lo + hi))
            print(f"    vs AA reference {lo:.2e}-{hi:.2e}: ratio {rel:.2f}x")
        print(f"  late feat self-cos   : {fmt(feat_cos_late)}")
        print(f"  late feat rel-L2     : {fmt(feat_l2_late)}")
        print(f"  late aux self-cos    : {fmt(aux_cos_late)}")
        print(f"  late policy self-cos : {fmt(pol_cos_late)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/R15_mpe_kl")
