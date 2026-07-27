"""Regenerate Figure 7: sample-efficiency curves on Overcooked AA.

Reads `results/canonical_sample_efficiency.json` and produces a 2-panel
figure with Final50 vs training budget and Collapse-percent vs training
budget across the four ablation configurations. Compliant with the paper
appendix's accessibility commitments:
  - IBM colorblind-safe palette, Full VABL = magenta
  - Distinct line styles + marker shapes per series
  - Serif font, no matplotlib titles (panels labeled in LaTeX caption)
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "canonical_sample_efficiency.json"
OUT = ROOT / "paper" / "figures" / "neurips" / "f7_sample_efficiency.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("full",    "Full (attn+aux)", "#DC267F", "-",   "X"),
    ("no_attn", "No Attn",         "#648FFF", "--",  "s"),
    ("no_aux",  "No Aux",          "#785EF0", "-.",  "D"),
    ("neither", "Neither",         "#FE6100", ":",   "o"),
]
BUDGETS = [
    (1250,  "500K",  500_000),
    (6250,  "2.5M",  2_500_000),
    (12500, "5M",    5_000_000),
    (25000, "10M",   10_000_000),
]


def load_series(d, cfg):
    final_means, final_stds, collapse_pcts = [], [], []
    for ep, _, _ in BUDGETS:
        key = f"{cfg}_{ep}ep"
        rec = d[key]
        final_means.append(rec["final_mean"])
        final_stds.append(rec["final_std"])
        best = rec["best_mean"]
        collapse = 100.0 * (best - rec["final_mean"]) / best if best > 0 else 0.0
        collapse_pcts.append(collapse)
    return np.array(final_means), np.array(final_stds), np.array(collapse_pcts)


def plot():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    d = json.load(open(DATA))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    xs = np.arange(len(BUDGETS))
    labels = [b[1] for b in BUDGETS]

    for cfg, label, color, ls, marker in CONFIGS:
        mean, std, collapse = load_series(d, cfg)
        ax1.errorbar(
            xs, mean, yerr=std, label=label, color=color, linestyle=ls,
            marker=marker, markersize=8, linewidth=2.0, capsize=3,
        )
        ax2.plot(
            xs, collapse, label=label, color=color, linestyle=ls,
            marker=marker, markersize=8, linewidth=2.0,
        )

    for ax, ylabel in [(ax1, "Final50"), (ax2, "Collapse %")]:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Training budget (env steps)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    plot()
