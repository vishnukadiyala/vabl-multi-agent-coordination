#!/usr/bin/env python
"""Generate comparison figures with improved legend placement."""

import json
import argparse
from pathlib import Path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def load_results(input_files):
    """Load results from JSON files."""
    results = {}

    for pattern in input_files:
        files = glob(pattern)
        if not files and Path(pattern).exists():
            files = [pattern]

        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)

            algo = data.get('algorithm', 'unknown').upper()

            if 'results' in data and data['results']:
                rewards = [np.array(r['rewards']) for r in data['results']]
                results[algo] = np.array(rewards)

    return results


def plot_comparison(results, output_path, title="Algorithm Comparison", smoothing=10):
    """Generate comparison figure with improved legend placement."""

    # Color scheme
    colors = {
        'VABL': '#2ca02c',   # Green
        'MAPPO': '#1f77b4',  # Blue
        'QMIX': '#ff7f0e',   # Orange
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    summary_stats = {}

    for algo, rewards in results.items():
        if rewards.size == 0:
            continue

        n_seeds, n_eps = rewards.shape
        episodes = np.arange(1, n_eps + 1)
        color = colors.get(algo, '#333333')

        # Smooth each seed
        smoothed = np.array([smooth(rewards[s], smoothing) for s in range(n_seeds)])

        # Compute mean and std
        mean_raw = np.mean(rewards, axis=0)
        std_raw = np.std(rewards, axis=0)
        mean_smooth = np.mean(smoothed, axis=0)
        std_smooth = np.std(smoothed, axis=0)

        # Plot raw (faint)
        ax.fill_between(episodes, mean_raw - std_raw, mean_raw + std_raw,
                        alpha=0.1, color=color)
        ax.plot(episodes, mean_raw, alpha=0.3, color=color, linewidth=1)

        # Plot smoothed (bold)
        ax.fill_between(episodes, mean_smooth - std_smooth, mean_smooth + std_smooth,
                        alpha=0.25, color=color)
        ax.plot(episodes, mean_smooth, color=color, linewidth=2.5, label=algo)

        # Compute summary stats
        final_reward = np.mean(rewards[:, -20:])
        best_reward = np.max(mean_smooth)
        summary_stats[algo] = {
            'final': final_reward,
            'best': best_reward,
            'std': np.std(np.mean(rewards[:, -20:], axis=1))
        }

    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend with statistics - placed outside the plot
    legend_labels = []
    legend_handles = []
    for algo in results.keys():
        if algo in summary_stats:
            s = summary_stats[algo]
            label = f"{algo}: Final={s['final']:.1f}, Best={s['best']:.1f}"
            line = plt.Line2D([0], [0], color=colors.get(algo, '#333333'), linewidth=2.5)
            legend_handles.append(line)
            legend_labels.append(label)

    # Place legend below the plot
    ax.legend(legend_handles, legend_labels,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.12),
              ncol=len(results),
              fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True)

    # Add summary table in upper right corner (outside main data area)
    table_text = "Summary Statistics\n"
    table_text += "-" * 30 + "\n"
    for algo in results.keys():
        if algo in summary_stats:
            s = summary_stats[algo]
            table_text += f"{algo}: {s['final']:.1f} ± {s['std']:.1f}\n"

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.98, table_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for legend below
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return summary_stats


def main():
    parser = argparse.ArgumentParser(description='Generate comparison figures')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Input JSON files')
    parser.add_argument('--output', '-o', required=True, help='Output figure path')
    parser.add_argument('--title', '-t', default='Algorithm Comparison', help='Figure title')
    parser.add_argument('--smoothing', type=int, default=10, help='Smoothing window')

    args = parser.parse_args()

    results = load_results(args.input)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} algorithms: {list(results.keys())}")

    stats = plot_comparison(results, args.output, args.title, args.smoothing)

    print("\nSummary:")
    for algo, s in stats.items():
        print(f"  {algo}: Final={s['final']:.1f} ± {s['std']:.1f}, Best={s['best']:.1f}")


if __name__ == '__main__':
    main()
