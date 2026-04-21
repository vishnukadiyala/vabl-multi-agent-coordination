"""
Generate a sample efficiency figure: episodes to reach X%% of peak performance.
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = pathlib.Path(__file__).resolve().parents[2] / "results"

FILES = {
    "Simple": {
        "VABL":  BASE / "vabl_simple_20260130_173050.json",
        "MAPPO": BASE / "mappo_simple_20260130_141911.json",
        "QMIX":  BASE / "qmix_simple_20260130_140926.json",
    },
    "Asymmetric\nAdvantages": {
        "VABL":  BASE / "vabl_overcooked_asymmetric_advantages_20260128_091429.json",
        "MAPPO": BASE / "mappo_overcooked_asymmetric_advantages_20260128_092319.json",
        "QMIX":  BASE / "qmix_overcooked_asymmetric_advantages_20260128_093017.json",
    },
    "Cramped\nRoom": {
        "VABL":  BASE / "vabl_overcooked_cramped_room_20260128_075958.json",
        "MAPPO": BASE / "mappo_overcooked_cramped_room_20260128_080824.json",
        "QMIX":  BASE / "qmix_overcooked_cramped_room_20260128_081504.json",
    },
}

ALGORITHMS = ["VABL", "MAPPO", "QMIX"]
COLORS = {"VABL": "#2ca02c", "MAPPO": "#1f77b4", "QMIX": "#ff7f0e"}
THRESHOLDS = [0.50, 0.75]
WINDOW = 20

def rolling_mean(x, w):
    kernel = np.ones(w) / w
    smoothed = np.convolve(x, kernel, mode="valid")
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, smoothed])

def first_episode_above(smoothed, threshold):
    indices = np.where(smoothed >= threshold)[0]
    if len(indices) == 0:
        return len(smoothed)
    return int(indices[0])

def load_rewards(path):
    with open(path) as f:
        data = json.load(f)
    return [seed["rewards"] for seed in data["results"]]

results = {}
details = {}

for env_name, algo_files in FILES.items():
    results[env_name] = {}
    details[env_name] = {}
    for algo, fpath in algo_files.items():
        seed_rewards = load_rewards(fpath)
        smoothed_seeds = [rolling_mean(np.array(r), WINDOW) for r in seed_rewards]
        global_peak = max(np.nanmax(s) for s in smoothed_seeds)
        per_seed_eps = {t: [] for t in THRESHOLDS}
        for s in smoothed_seeds:
            for t in THRESHOLDS:
                ep = first_episode_above(s, t * global_peak)
                per_seed_eps[t].append(ep)
        results[env_name][algo] = {}
        details[env_name][algo] = {"peak": global_peak, "n_seeds": len(seed_rewards)}
        for t in THRESHOLDS:
            mean_ep = float(np.mean(per_seed_eps[t]))
            std_ep = float(np.std(per_seed_eps[t]))
            results[env_name][algo][t] = mean_ep
            details[env_name][algo][t] = {"mean": mean_ep, "std": std_ep, "per_seed": per_seed_eps[t]}

print("=" * 80)
print("SAMPLE EFFICIENCY: Episodes to reach X%% of peak performance")
print("=" * 80)
for env_name in FILES:
    label = env_name.replace(chr(10), " ")
    print(f"\n  Environment: {label}")
    for algo in ALGORITHMS:
        d = details[env_name][algo]
        print(f"    {algo:6s}  peak={d[chr(112)+chr(101)+chr(97)+chr(107)]:.2f}  seeds={d[chr(110)+chr(95)+chr(115)+chr(101)+chr(101)+chr(100)+chr(115)]}")
        for t in THRESHOLDS:
            info = d[t]
            print(f"      {int(t*100):3d}%% threshold: mean={info[chr(109)+chr(101)+chr(97)+chr(110)]:.1f}  std={info[chr(115)+chr(116)+chr(100)]:.1f}  per_seed={info[chr(112)+chr(101)+chr(114)+chr(95)+chr(115)+chr(101)+chr(101)+chr(100)]}")
print()

env_names = list(FILES.keys())
n_envs = len(env_names)
n_algos = len(ALGORITHMS)
x = np.arange(n_envs)
bar_width = 0.22

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

for ax_idx, threshold in enumerate(THRESHOLDS):
    ax = axes[ax_idx]
    all_stds = []
    for algo in ALGORITHMS:
        for env in env_names:
            all_stds.append(details[env][algo][threshold]["std"])
    max_std = max(all_stds) if all_stds else 0
    for i, algo in enumerate(ALGORITHMS):
        vals = [results[env][algo][threshold] for env in env_names]
        stds = [details[env][algo][threshold]["std"] for env in env_names]
        offset = (i - (n_algos - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, yerr=stds, capsize=3,
                      label=algo, color=COLORS[algo], edgecolor="black",
                      linewidth=0.5, error_kw={"linewidth": 1.0})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_std * 0.15 + 2,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=9)
    ax.set_ylabel("Episodes to Threshold", fontsize=10)
    ax.set_title(f"{int(threshold*100)}%% of Peak Performance", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

fig.suptitle("Sample Efficiency Comparison", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()

out_path = pathlib.Path(__file__).resolve().parents[2] / "figures" / "sample_efficiency.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Figure saved to: {out_path}")
