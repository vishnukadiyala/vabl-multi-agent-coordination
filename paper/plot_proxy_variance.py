"""Scatter plot: cosine temporal std proxy vs Final50 cross-seed std,
across our four logged gradient-decomposition conditions on AA.

Supports the paper's claim that the cosine temporal std is a
finite-sample proxy for tr(J_pi Sigma_pi J_pi^T) normalized by the
gradient magnitudes, and that this proxy predicts the observed
Final50 cross-seed variance ordering.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "expB_gradient_decomp"
OUT = ROOT / "paper" / "figures" / "neurips" / "f_proxy_variance.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

CONFIGS = ["full", "no_aux", "stopgrad", "anneal"]
LABELS = {
    "full": "Full (attn + aux)",
    "no_aux": "No Aux",
    "stopgrad": "+ Stop-grad",
    "anneal": "+ Anneal",
}
COLORS = {
    "full": "#E41A1C",
    "no_aux": "#377EB8",
    "stopgrad": "#4DAF4A",
    "anneal": "#FF7F00",
}


def load():
    agg_x, agg_y, labels = [], [], []
    for cfg in CONFIGS:
        xs, f50s = [], []
        for seed in range(5):
            p = DATA / f"expB_{cfg}_seed{seed}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            r = d["rewards"]
            f50s.append(float(np.mean(r[-50:])))
            gd = d.get("gradient_decomp", [])
            cos = np.array([e["cosine"] for e in gd])
            k = len(cos) // 2
            cs = float(np.std(cos[k:], ddof=1)) if len(cos[k:]) > 1 else 0.0
            xs.append(cs)
        if xs:
            agg_x.append(np.mean(xs))
            agg_y.append(np.std(f50s, ddof=1))
            labels.append(cfg)
    return np.array(agg_x), np.array(agg_y), labels


def plot(x, y, labels, save_to):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6))

    for xi, yi, lab in zip(x, y, labels):
        ax.scatter(
            xi, yi, s=120, c=COLORS[lab], edgecolor="black", linewidth=0.8,
            label=LABELS[lab], zorder=3,
        )
        ax.annotate(
            LABELS[lab], xy=(xi, yi), xytext=(8, -2),
            textcoords="offset points", fontsize=9, color="black",
        )

    # Linear fit
    slope, intercept = np.polyfit(x, y, 1)
    xs_line = np.linspace(0, max(x) * 1.15, 100)
    ax.plot(
        xs_line, slope * xs_line + intercept, "k--", linewidth=1,
        alpha=0.5, zorder=1,
    )
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    ax.set_xlabel(
        r"Cosine temporal std, late 50\% (proxy for $\mathrm{tr}(J_\pi \Sigma_\pi J_\pi^\top)$)"
    )
    ax.set_ylabel(r"Final50 cross-seed std")
    # Title removed; figure labeled in LaTeX caption (R^2 reported there).
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, max(x) * 1.25)
    ax.set_ylim(0, max(y) * 1.15)

    # Note the construction zeros
    ax.text(
        0.005, -0.3, "0 by construction\n(disjoint param supports)",
        fontsize=7, color="gray", transform=ax.transData,
        ha="left", va="top",
    )

    plt.tight_layout()
    plt.savefig(save_to, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_to}")
    print(f"  Points: {list(zip([round(a,3) for a in x], [round(a,2) for a in y], labels))}")
    print(f"  R^2 = {r2:.3f}")


if __name__ == "__main__":
    x, y, labels = load()
    if len(x) == 0:
        print("No data")
    else:
        plot(x, y, labels, OUT)
