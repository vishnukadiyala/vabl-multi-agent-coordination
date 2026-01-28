#!/usr/bin/env python
"""Publication-grade plotting script for VABL paper figures.

Generates publication-quality figures with:
- Smoothed learning curves with configurable window
- 95% confidence intervals using t-distribution
- Raw (faint) + smoothed (bold) curves
- Best-100 and Last-100 summary table (inset)
- Horizontal baseline reference lines
- Caption template with seeds/smoothing info
- 300 DPI output

Usage:
    python -m marl_research.scripts.plot_publication_figure \\
        --input results/vabl_*.json \\
        --output figures/learning_curve.png \\
        --smoothing-window 10 \\
        --baseline-value 0.0 \\
        --baseline-label "Random Policy" \\
        --show-table
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# Publication style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply exponential moving average smoothing.

    Args:
        data: 1D array of values
        window: Smoothing window size

    Returns:
        Smoothed array of same length
    """
    if window <= 1:
        return data

    # Use exponential moving average for smoother curves
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def compute_ci(
    data_matrix: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and confidence intervals using t-distribution.

    Args:
        data_matrix: [n_seeds, n_episodes] array
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        mean, ci_lower, ci_upper arrays
    """
    n_seeds = data_matrix.shape[0]
    mean = np.mean(data_matrix, axis=0)
    std = np.std(data_matrix, axis=0, ddof=1)  # Sample std
    se = std / np.sqrt(n_seeds)

    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n_seeds - 1)
    ci = t_crit * se

    return mean, mean - ci, mean + ci


def load_results(input_paths: List[str]) -> Dict[str, np.ndarray]:
    """Load results from JSON files.

    Supports multiple file patterns and aggregates by algorithm name.

    Args:
        input_paths: List of JSON file paths (can include glob patterns)

    Returns:
        Dict mapping algorithm names to [n_seeds, n_episodes] reward arrays
    """
    from glob import glob

    results = {}

    for pattern in input_paths:
        files = glob(pattern)
        # If no glob matches, try as literal path
        if not files:
            if Path(pattern).exists():
                files = [pattern]
        for fpath in files:
            with open(fpath, 'r') as f:
                data = json.load(f)

            # Handle different data structures
            # Check for ablation sweep format first
            if isinstance(data, dict) and 'ablation' in data and 'results' in data:
                # Ablation sweep format: {"ablation": "name", "param": "x", "results": [...]}
                param_name = data.get('param', 'param')
                for param_result in data['results']:
                    param_value = param_result.get('param_value', 'unknown')
                    key = f"{param_name}={param_value}"
                    if key not in results:
                        results[key] = []
                    for seed_result in param_result.get('results', []):
                        rewards = seed_result.get('rewards', [])
                        if rewards:
                            results[key].append(rewards)
                continue

            if isinstance(data, list):
                # List of experiments
                for exp in data:
                    algo_name = exp.get('algorithm', 'unknown')
                    env_name = exp.get('env', '')
                    layout = exp.get('layout', '')
                    key = f"{algo_name}"
                    if env_name:
                        key = f"{algo_name}_{env_name}"
                    if layout:
                        key = f"{key}_{layout}"

                    if 'results' in exp:
                        # Has seeds
                        for seed_result in exp['results']:
                            rewards = seed_result.get('rewards', [])
                            if rewards:
                                if key not in results:
                                    results[key] = []
                                results[key].append(rewards)
                    elif 'rewards' in exp:
                        # Single run
                        if key not in results:
                            results[key] = []
                        results[key].append(exp['rewards'])
            elif isinstance(data, dict):
                # Check if dict keys are algorithm names with reward arrays
                # Format: {"VABL": [[ep1, ep2, ...], [ep1, ep2, ...]], "QMIX": [...]}
                first_key = next(iter(data.keys()), None)
                if first_key and isinstance(data.get(first_key), list):
                    first_val = data[first_key]
                    if first_val and isinstance(first_val[0], list):
                        # This is algorithm -> list of seed reward arrays format
                        for algo_name, seed_arrays in data.items():
                            if algo_name not in results:
                                results[algo_name] = []
                            for seed_rewards in seed_arrays:
                                if seed_rewards:
                                    results[algo_name].append(seed_rewards)
                        continue

                # Single experiment dict
                algo_name = data.get('algorithm', Path(fpath).stem)
                if 'rewards' in data:
                    if algo_name not in results:
                        results[algo_name] = []
                    results[algo_name].append(data['rewards'])

    # Convert to numpy arrays (pad if needed)
    for key in results:
        seeds = results[key]
        if not seeds:
            continue
        max_len = max(len(s) for s in seeds)
        padded = []
        for s in seeds:
            if len(s) < max_len:
                # Pad with last value
                s = list(s) + [s[-1]] * (max_len - len(s))
            padded.append(s)
        results[key] = np.array(padded)

    return results


def compute_summary_stats(rewards: np.ndarray, n_episodes: int = 100) -> Dict[str, float]:
    """Compute Best-N and Last-N statistics.

    Args:
        rewards: [n_seeds, n_episodes] array
        n_episodes: Window size for statistics

    Returns:
        Dict with best_mean, best_std, last_mean, last_std
    """
    n_seeds, total_eps = rewards.shape

    # Last N episodes
    last_n = rewards[:, -n_episodes:]
    last_mean = np.mean(last_n)
    last_std = np.std(np.mean(last_n, axis=1))  # Std across seeds

    # Best N episodes (rolling window max across seeds)
    best_means = []
    for seed in range(n_seeds):
        rolling_means = []
        for i in range(total_eps - n_episodes + 1):
            rolling_means.append(np.mean(rewards[seed, i:i + n_episodes]))
        best_means.append(max(rolling_means) if rolling_means else np.mean(rewards[seed]))

    best_mean = np.mean(best_means)
    best_std = np.std(best_means)

    return {
        'best_mean': best_mean,
        'best_std': best_std,
        'last_mean': last_mean,
        'last_std': last_std,
    }


def plot_publication_figure(
    results: Dict[str, np.ndarray],
    output_path: str,
    smoothing_window: int = 10,
    baseline_value: Optional[float] = None,
    baseline_label: str = "Baseline",
    show_table: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Episode",
    ylabel: str = "Return",
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """Generate publication-quality learning curve figure.

    Args:
        results: Dict mapping algorithm names to [n_seeds, n_episodes] arrays
        output_path: Path to save figure
        smoothing_window: Window size for smoothing
        baseline_value: Optional horizontal baseline
        baseline_label: Label for baseline
        show_table: Whether to show Best-100/Last-100 table
        title: Optional figure title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Optional dict mapping algorithm names to colors
    """
    if colors is None:
        # Default color palette
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors = {name: default_colors[i % len(default_colors)]
                  for i, name in enumerate(results.keys())}

    fig, ax = plt.subplots(figsize=(10, 6))

    summary_data = {}
    n_seeds_total = 0

    for algo_name, rewards in results.items():
        if rewards.size == 0:
            continue

        n_seeds, n_episodes = rewards.shape
        n_seeds_total = max(n_seeds_total, n_seeds)
        episodes = np.arange(1, n_episodes + 1)
        color = colors.get(algo_name, '#1f77b4')

        # Compute smoothed curves for each seed
        smoothed_seeds = np.array([smooth(rewards[s], smoothing_window)
                                   for s in range(n_seeds)])

        # Compute statistics
        mean_raw, ci_low_raw, ci_high_raw = compute_ci(rewards)
        mean_smooth, ci_low_smooth, ci_high_smooth = compute_ci(smoothed_seeds)

        # Plot raw (faint)
        ax.fill_between(episodes, ci_low_raw, ci_high_raw,
                        alpha=0.1, color=color, linewidth=0)
        ax.plot(episodes, mean_raw, alpha=0.3, color=color, linewidth=1)

        # Plot smoothed (bold)
        ax.fill_between(episodes, ci_low_smooth, ci_high_smooth,
                        alpha=0.25, color=color, linewidth=0)
        ax.plot(episodes, mean_smooth, color=color, linewidth=2.5,
                label=f"{algo_name} (smoothed)")

        # Compute summary statistics
        summary_data[algo_name] = compute_summary_stats(rewards, n_episodes=100)

    # Add baseline
    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='gray', linestyle='--',
                   linewidth=1.5, label=baseline_label, alpha=0.7)

    # Add Best-100 / Last-100 table as inset
    if show_table and summary_data:
        table_text = "Algorithm       Best-100        Last-100\n"
        table_text += "-" * 45 + "\n"
        for algo_name, stats in summary_data.items():
            short_name = algo_name[:12]
            best_str = f"{stats['best_mean']:.1f} +/- {stats['best_std']:.1f}"
            last_str = f"{stats['last_mean']:.1f} +/- {stats['last_std']:.1f}"
            table_text += f"{short_name:<15} {best_str:<15} {last_str}\n"

        # Position table in upper left or lower right depending on trend
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, table_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace', bbox=props)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Generate caption
    caption = f"Figure: Learning curves with {smoothing_window}-episode smoothing. "
    caption += f"Shaded regions show 95% CI across {n_seeds_total} seeds. "
    caption += "Faint lines show raw data; bold lines show smoothed."
    print(f"\nSuggested caption:\n{caption}\n")

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality learning curve figures"
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input JSON file(s) or glob patterns"
    )
    parser.add_argument(
        "--output", "-o",
        default="figures/learning_curve.png",
        help="Output figure path"
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Smoothing window size (default: 10)"
    )
    parser.add_argument(
        "--baseline-value",
        type=float,
        default=None,
        help="Optional horizontal baseline value"
    )
    parser.add_argument(
        "--baseline-label",
        default="Random Policy",
        help="Label for baseline"
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Show Best-100/Last-100 summary table"
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title"
    )
    parser.add_argument(
        "--xlabel",
        default="Episode",
        help="X-axis label"
    )
    parser.add_argument(
        "--ylabel",
        default="Return",
        help="Y-axis label"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.input)

    if not results:
        print("No results found in input files!")
        return

    print(f"Loaded {len(results)} algorithm(s): {list(results.keys())}")
    for name, data in results.items():
        print(f"  {name}: {data.shape[0]} seeds, {data.shape[1]} episodes")

    # Generate figure
    plot_publication_figure(
        results=results,
        output_path=str(output_path),
        smoothing_window=args.smoothing_window,
        baseline_value=args.baseline_value,
        baseline_label=args.baseline_label,
        show_table=args.show_table,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()
