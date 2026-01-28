"""Visualization utilities for MARL experiments."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def plot_learning_curves(
    results: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
    xlabel: str = "Timesteps",
    ylabel: str = "Return",
    smooth_window: int = 10,
) -> None:
    """Plot learning curves with confidence intervals."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return

    plt.figure(figsize=(10, 6))

    for name, values in results.items():
        values = np.array(values)

        if smooth_window > 1 and len(values) > smooth_window:
            smoothed = np.convolve(values, np.ones(smooth_window) / smooth_window, mode="valid")
            x = np.arange(len(smoothed))
            plt.plot(x, smoothed, label=name)
        else:
            plt.plot(values, label=name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_multi_seed_results(
    results: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Multi-Seed Results",
    xlabel: str = "Timesteps",
    ylabel: str = "Return",
) -> None:
    """Plot results across multiple seeds with mean and std."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for i, (name, data) in enumerate(results.items()):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))

        color = colors[i % len(colors)]
        plt.plot(x, mean, label=name, color=color)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Algorithm Comparison",
    ylabel: str = "Win Rate",
) -> None:
    """Create a bar plot comparing algorithms across environments."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return

    algorithms = list(results.keys())
    environments = list(results[algorithms[0]].keys())

    x = np.arange(len(environments))
    width = 0.8 / len(algorithms)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, algo in enumerate(algorithms):
        values = [results[algo][env] for env in environments]
        ax.bar(x + i * width, values, width, label=algo)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(environments, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_win_rate_matrix(
    win_rates: np.ndarray,
    agent_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cross-Play Win Rate Matrix",
) -> None:
    """Plot a heatmap of cross-play win rates."""
    if not HAS_PLOTTING:
        print("Matplotlib not available for plotting")
        return

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        win_rates,
        annot=True,
        fmt=".2f",
        xticklabels=agent_names,
        yticklabels=agent_names,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    plt.title(title)
    plt.xlabel("Partner Agent")
    plt.ylabel("Agent")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_results_table(
    results: Dict[str, Dict[str, Tuple[float, float]]],
    save_path: Optional[str] = None,
) -> str:
    """Create a LaTeX-formatted results table."""
    algorithms = list(results.keys())
    environments = list(results[algorithms[0]].keys())

    header = " & ".join(["Algorithm"] + environments) + " \\\\"
    lines = ["\\begin{tabular}{" + "l" + "c" * len(environments) + "}", "\\toprule", header, "\\midrule"]

    for algo in algorithms:
        row_values = []
        for env in environments:
            mean, std = results[algo][env]
            row_values.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        lines.append(f"{algo} & " + " & ".join(row_values) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    table = "\n".join(lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(table)

    return table
