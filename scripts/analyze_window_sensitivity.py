"""Final50 window sensitivity + peak-to-final drops (answers fXvf Q6).

The reviewer asked us to justify the Final50 evaluation window and to report
peak-to-final drops. This recomputes the headline contrast under several
evaluation windows on both numerical stacks, so the conclusion can be shown
not to depend on the window choice.

Usage: python scripts/analyze_window_sensitivity.py
"""
import glob
import json

import numpy as np

ENVS = 64  # episodes per iteration


def series(path):
    return np.array(json.load(open(path))["rewards"])


def window_mean(r, iters):
    return float(r[-iters * ENVS:].mean())


def peak_to_final(r, iters=50):
    """Peak of the smoothed curve minus the final-window mean."""
    k = 10 * ENVS  # smooth over 10 iterations
    sm = np.convolve(r, np.ones(k) / k, mode="valid")
    return float(sm.max() - window_mean(r, iters))


def collect(pattern):
    return sorted(glob.glob(pattern))


def arm(files, iters):
    return np.array([window_mean(series(f), iters) for f in files])


SETS = {
    "old stack (jax 0.6.2)": (
        "results/R1_kl_instrumented/R1_full_seed*.json",
        "results/R1_kl_instrumented/R1_no_aux_seed*.json",
    ),
    "V2 stack (jax 0.10.2)": (
        "results/V2_verification/V2_full_seed*.json",
        "results/V2_verification/V2_no_aux_seed*.json",
    ),
}

print("=== Evaluation-window sensitivity of the Full vs No-Aux gap ===")
print(f"{'set':24s} {'window':>10s} {'Full':>9s} {'No-Aux':>9s} {'gap':>7s} {'n/arm':>6s}")
for name, (pf, pn) in SETS.items():
    ff, nf = collect(pf), collect(pn)
    if not ff or not nf:
        continue
    k = min(len(ff), len(nf))
    ff, nf = ff[:k], nf[:k]
    for iters, tag in [(25, "Final25"), (50, "Final50"), (100, "Final100"), (200, "Final200")]:
        a, b = arm(ff, iters), arm(nf, iters)
        print(f"{name:24s} {tag:>10s} {a.mean():9.2f} {b.mean():9.2f} {b.mean()-a.mean():7.2f} {k:6d}")
    print()

print("=== Peak-to-final drop (smoothed peak minus Final50) ===")
print(f"{'set / condition':34s} {'mean':>7s} {'sd':>6s} {'max':>7s} {'n':>4s}")
for name, (pf, pn) in SETS.items():
    for label, pat in [("Full", pf), ("No-Aux", pn)]:
        fs = collect(pat)
        if not fs:
            continue
        d = np.array([peak_to_final(series(f)) for f in fs])
        print(f"{name + ' / ' + label:34s} {d.mean():7.2f} {d.std(ddof=1):6.2f} {d.max():7.2f} {len(d):4d}")
