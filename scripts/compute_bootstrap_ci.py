"""Compute bootstrap CIs on Cohen's d for the headline Overcooked AA ablation.

Uses n=50,000 resamples (up from the 10,000 in the existing appendix table).
Outputs a LaTeX fragment for the appendix table and a plain-text summary.
"""
import json
import numpy as np

N_RESAMPLES = 50_000
RNG = np.random.default_rng(seed=42)


def cohens_d(a, b):
    # Pooled std, treating a as baseline to compare vs. b (we report d vs. Full
    # as (Full - other) / pooled, so positive means the other config is better).
    # We compute the "vs. Full" convention: d = (mean_other - mean_full) / pooled.
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2)
    if pooled < 1e-12:
        return 0.0
    return (mean_b - mean_a) / pooled


def bootstrap_ci(full_seeds, other_seeds, n_resamples=N_RESAMPLES, alpha=0.05):
    """Non-parametric bootstrap: resample each group independently."""
    full = np.asarray(full_seeds, dtype=float)
    other = np.asarray(other_seeds, dtype=float)
    n_full = len(full)
    n_other = len(other)

    ds = np.empty(n_resamples)
    for i in range(n_resamples):
        fs = full[RNG.integers(0, n_full, size=n_full)]
        os_ = other[RNG.integers(0, n_other, size=n_other)]
        ds[i] = cohens_d(fs, os_)

    lo = np.percentile(ds, 100 * alpha / 2)
    hi = np.percentile(ds, 100 * (1 - alpha / 2))
    point = cohens_d(full, other)
    crosses_zero = lo <= 0 <= hi
    return point, lo, hi, crosses_zero


def main():
    ph = json.load(open("results/canonical_phase2.json"))
    cr = json.load(open("results/canonical_phase2_cramped.json"))

    # Overcooked AA headline
    full_aa = ph["configs"]["A_full"]["final50_per_seed"]
    rows_aa = [
        ("Neither",         ph["configs"]["A_neither"]["final50_per_seed"]),
        ("No Attention",    ph["configs"]["A_no_attn"]["final50_per_seed"]),
        ("No Aux",          ph["configs"]["A_no_aux"]["final50_per_seed"]),
        ("+ Anneal",        ph["configs"]["B_anneal"]["final50_per_seed"]),
        ("+ Stop-gradient", ph["configs"]["B_stopgrad"]["final50_per_seed"]),
        ("+ Both",          ph["configs"]["B_anneal_stopgrad"]["final50_per_seed"]),
    ]

    # Cramped Room headline (if present)
    full_cr_key = "full" if "full" in cr["configs"] else None
    rows_cr = []
    if full_cr_key:
        full_cr = cr["configs"][full_cr_key]["final50_per_seed"]
        for name, key in [("Neither", "neither"), ("No Attention", "no_attn"), ("No Aux", "no_aux")]:
            if key in cr["configs"]:
                rows_cr.append((name, cr["configs"][key]["final50_per_seed"]))

    print(f"Bootstrap Cohen's d vs. Full with {N_RESAMPLES:,} resamples, 95% CI\n")
    print("=== Overcooked AA (headline ablation) ===")
    print(f"{'Config':<22} {'d':>8} {'CI lo':>8} {'CI hi':>8} {'Zero?':>8}")
    for name, vals in rows_aa:
        d, lo, hi, cz = bootstrap_ci(full_aa, vals)
        print(f"{name:<22} {d:>+8.3f} {lo:>+8.3f} {hi:>+8.3f} {'yes' if cz else 'no':>8}")

    if rows_cr:
        print("\n=== Overcooked Cramped Room ===")
        print(f"{'Config':<22} {'d':>8} {'CI lo':>8} {'CI hi':>8} {'Zero?':>8}")
        for name, vals in rows_cr:
            d, lo, hi, cz = bootstrap_ci(full_cr, vals)
            print(f"{name:<22} {d:>+8.3f} {lo:>+8.3f} {hi:>+8.3f} {'yes' if cz else 'no':>8}")


if __name__ == "__main__":
    main()
