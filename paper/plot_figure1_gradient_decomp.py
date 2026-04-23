"""Plot new Figure 1 from ExpB gradient-decomposition data.

Design B: focus on Full VABL's directional signature across 5 seeds.
Fix-path comparisons are omitted because under stop-gradient the
aux-to-encoder gradient is zero by construction (disjoint parameter
supports), so a cosine comparison does not yield a dynamical finding.
The paper claims "Full VABL exhibits directional aux-vs-policy
gradient conflict"; this figure supports that claim directly.

Inputs:  results/expB_gradient_decomp/expB_full_seed{0..4}.json

Panels:
    (a) Per-seed cosine trajectories for Full VABL (5 seeds, light
        lines; mean in bold). Shows the characteristic oscillation
        pattern is consistent across seeds.
    (b) Cosine distribution over the late 50% of training, pooled
        across seeds. Shows the distribution is approximately
        zero-mean with substantial spread, i.e. directional noise
        rather than a persistent conflict direction.
    (c) Magnitude ratio |g_aux|/|g_policy| over training across 5
        seeds. Supports the "not magnitude-driven" claim: the ratio
        stays bounded well below 0.25 throughout training.
"""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "expB_gradient_decomp"
OUT_PDF = ROOT / "paper" / "figures" / "neurips" / "f1_gradient_decomp_v2.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

FN_RE = re.compile(r"expB_(\w+?)_seed(\d+)\.json$")
# IBM colorblind-safe palette (matches the accessibility claim in Appendix K).
# Magenta reserved for Full VABL in other figures; seeds use the other five.
SEED_COLORS = ["#648FFF", "#785EF0", "#FE6100", "#FFB000", "#009E73"]
SEED_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
SEED_MARKERS = ["o", "s", "D", "^", "v"]
MEAN_COLOR = "#000000"
FULL_MAGENTA = "#DC267F"


def load_full_seeds():
    """Returns dict[seed] = list of grad-decomp entries."""
    out = {}
    if not DATA.exists():
        return out
    for p in sorted(DATA.glob("expB_full_seed*.json")):
        m = FN_RE.search(p.name)
        if not m or m.group(1) != "full":
            continue
        seed = int(m.group(2))
        d = json.load(open(p))
        gd = d.get("gradient_decomp", [])
        if gd:
            out[seed] = gd
    return out


def interp_series(iters_target, entries, key):
    its = np.array([e["iteration"] for e in entries])
    vals = np.array([e[key] for e in entries])
    return np.interp(iters_target, its, vals)


def plot(seeds_data, save_to):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Common grid: widest iteration range available
    iters = sorted(set().union(*[
        {e["iteration"] for e in gd} for gd in seeds_data.values()
    ]))
    ref_iters = np.array(iters)

    seeds = sorted(seeds_data.keys())

    # Place markers every ~7 points so lines stay readable.
    marker_every = max(1, len(ref_iters) // 7)

    # ---------- Panel (a): per-seed cosine trajectories ----------
    ax = axes[0]
    cos_matrix = []
    for i, s in enumerate(seeds):
        y = interp_series(ref_iters, seeds_data[s], "cosine")
        cos_matrix.append(y)
        ax.plot(
            ref_iters, y,
            color=SEED_COLORS[i % len(SEED_COLORS)],
            linestyle=SEED_LINESTYLES[i % len(SEED_LINESTYLES)],
            marker=SEED_MARKERS[i % len(SEED_MARKERS)],
            markersize=6, markevery=marker_every,
            linewidth=1.6, alpha=0.85, label=f"seed {s}",
        )
    cos_arr = np.array(cos_matrix)
    ax.plot(ref_iters, cos_arr.mean(axis=0), color=MEAN_COLOR,
            linewidth=2.6, linestyle="-", label="mean")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\cos(g_{\pi}, g_{\mathrm{aux}})$")
    ax.set_title("(a) Full VABL cosine, per-seed")
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---------- Panel (b): cosine distribution (late 50%) pooled ----------
    ax = axes[1]
    late_cos = []
    for s in seeds:
        cosines = [e["cosine"] for e in seeds_data[s]]
        k = len(cosines) // 2
        late_cos.extend(cosines[k:])
    late_cos = np.array(late_cos)
    ax.hist(
        late_cos, bins=18,
        color=FULL_MAGENTA, edgecolor="black",
        hatch="//", alpha=0.8, linewidth=0.9,
    )
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axvline(late_cos.mean(), color=MEAN_COLOR, linewidth=2.4, linestyle="-",
               label=fr"mean = {late_cos.mean():+.3f}")
    ax.set_xlabel(r"$\cos(g_{\pi}, g_{\mathrm{aux}})$")
    ax.set_ylabel("Count (iterations x seeds)")
    ax.set_title("(b) Cosine distribution, late 50%")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    std_txt = fr"std = {late_cos.std(ddof=1):.3f}"
    ax.text(0.02, 0.97, std_txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=12, bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    # ---------- Panel (c): magnitude ratio (all 5 seeds) ----------
    ax = axes[2]
    ratio_matrix = []
    for i, s in enumerate(seeds):
        itr = [e["iteration"] for e in seeds_data[s]]
        ratio = [e["norm_aux"] / (e["norm_policy"] + 1e-12) for e in seeds_data[s]]
        ratio_y = np.interp(ref_iters, itr, ratio)
        ratio_matrix.append(ratio_y)
        ax.plot(
            ref_iters, ratio_y,
            color=SEED_COLORS[i % len(SEED_COLORS)],
            linestyle=SEED_LINESTYLES[i % len(SEED_LINESTYLES)],
            marker=SEED_MARKERS[i % len(SEED_MARKERS)],
            markersize=6, markevery=marker_every,
            linewidth=1.6, alpha=0.85, label=f"seed {s}",
        )
    r_arr = np.array(ratio_matrix)
    ax.plot(ref_iters, r_arr.mean(axis=0), color=MEAN_COLOR,
            linewidth=2.6, linestyle="-", label="mean")
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="0.25")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$|g_{\mathrm{aux}}| / |g_{\pi}|$")
    ax.set_title("(c) Magnitude ratio")
    ax.legend(loc="best", frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.35, r_arr.max() * 1.1))

    plt.tight_layout()
    plt.savefig(save_to, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_to}")
    print(f"\n=== Numbers for caption ===")
    print(f"n_seeds = {len(seeds)}")
    print(f"Full cosine, late 50%: mean = {late_cos.mean():+.4f}, std = {late_cos.std(ddof=1):.4f}")
    print(f"Full cosine range (all pooled): [{late_cos.min():+.3f}, {late_cos.max():+.3f}]")
    final_ratios = r_arr[:, -1]
    peak_ratio = r_arr.max()
    print(f"Magnitude ratio: final mean = {final_ratios.mean():.3f}, peak = {peak_ratio:.3f}")


if __name__ == "__main__":
    seeds_data = load_full_seeds()
    print(f"Loaded Full VABL seeds: {sorted(seeds_data.keys())}")
    if not seeds_data:
        print("(no data yet)")
    else:
        plot(seeds_data, OUT_PDF)
