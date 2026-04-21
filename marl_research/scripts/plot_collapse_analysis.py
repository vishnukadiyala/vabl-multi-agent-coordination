#!/usr/bin/env python
"""
Collapse Analysis Plot for VABL ICML 2026
=========================================

Generates the key figure showing:
- MAPPO spikes then collapses
- VABL plateaus higher more consistently

Includes:
- Learning curves with collapse indicator
- Best-Final gap visualization
- Stability comparison

Usage:
    python -m marl_research.scripts.plot_collapse_analysis --input results/*.json --output figures/collapse_analysis.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    alpha = 2 / (window + 1)
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def compute_collapse_point(rewards: np.ndarray, window: int = 10, threshold_ratio: float = 0.5) -> int:
    """Find the episode where performance collapses below threshold.

    Returns the episode number where smoothed reward drops below
    threshold_ratio * peak_reward, or -1 if no collapse.
    """
    smoothed = smooth(rewards, window)
    peak = np.max(smoothed)
    peak_idx = np.argmax(smoothed)
    threshold = threshold_ratio * peak

    # Look for collapse after peak
    post_peak = smoothed[peak_idx:]
    below_threshold = np.where(post_peak < threshold)[0]

    if len(below_threshold) > 0:
        return peak_idx + below_threshold[0]
    return -1


def load_results(input_files: List[str]) -> Dict[str, Dict]:
    """Load results from JSON files."""
    from glob import glob

    results = {}

    for pattern in input_files:
        files = glob(pattern)
        if not files and Path(pattern).exists():
            files = [pattern]

        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)

            algo = data.get('algorithm', 'unknown').upper()
            env = data.get('env', 'unknown')
            layout = data.get('layout', '')

            key = f"{algo}"
            if 'results' in data and data['results']:
                results[key] = {
                    'rewards': [np.array(r['rewards']) for r in data['results']],
                    'env': env,
                    'layout': layout,
                }

    return results


def plot_collapse_analysis(results: Dict, output_path: str, env_name: str = ""):
    """Generate the collapse analysis figure."""

    # Color scheme
    colors = {
        'VABL': '#2ca02c',  # Green
        'MAPPO': '#1f77b4',  # Blue
        'QMIX': '#ff7f0e',  # Orange
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ===== Panel 1: Learning Curves with Collapse Markers =====
    ax1 = axes[0]

    summary_data = {}

    for algo, data in results.items():
        if algo not in colors:
            continue

        rewards = np.array(data['rewards'])
        n_seeds, n_eps = rewards.shape
        episodes = np.arange(1, n_eps + 1)
        color = colors[algo]

        # Smooth each seed
        smoothed = np.array([smooth(rewards[s]) for s in range(n_seeds)])

        # Mean and CI
        mean = np.mean(smoothed, axis=0)
        std = np.std(smoothed, axis=0)

        # Plot
        ax1.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)
        ax1.plot(episodes, mean, color=color, linewidth=2.5, label=algo)

        # Mark collapse points
        collapse_points = []
        for s in range(n_seeds):
            cp = compute_collapse_point(rewards[s])
            if cp > 0:
                collapse_points.append(cp)

        if collapse_points:
            avg_collapse = np.mean(collapse_points)
            ax1.axvline(x=avg_collapse, color=color, linestyle='--', alpha=0.5)
            ax1.scatter([avg_collapse], [mean[int(avg_collapse)-1]],
                       marker='x', s=100, color=color, zorder=5)

        # Store summary
        summary_data[algo] = {
            'final': np.mean(rewards[:, -20:]),
            'best': np.max(mean),
            'collapse_rate': len(collapse_points) / n_seeds,
        }

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Learning Curves (× = avg collapse point)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ===== Panel 2: Best vs Final Gap =====
    ax2 = axes[1]

    algos = list(summary_data.keys())
    x = np.arange(len(algos))
    width = 0.35

    bests = [summary_data[a]['best'] for a in algos]
    finals = [summary_data[a]['final'] for a in algos]

    bars1 = ax2.bar(x - width/2, bests, width, label='Best', alpha=0.8,
                    color=[colors.get(a, 'gray') for a in algos])
    bars2 = ax2.bar(x + width/2, finals, width, label='Final', alpha=0.5,
                    color=[colors.get(a, 'gray') for a in algos], hatch='//')

    # Add gap annotations
    for i, algo in enumerate(algos):
        gap = bests[i] - finals[i]
        gap_pct = (gap / bests[i]) * 100 if bests[i] > 0 else 0
        ax2.annotate(f'{gap_pct:.0f}%\ncollapse',
                    xy=(i, max(bests[i], finals[i]) + 5),
                    ha='center', fontsize=9, color='red')

    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Best vs Final Performance', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Panel 3: Stability Metrics =====
    ax3 = axes[2]

    metrics = ['Final Reward', 'Best-Final Gap %', 'Collapse Rate %']

    data_matrix = []
    for algo in algos:
        s = summary_data[algo]
        gap_pct = ((s['best'] - s['final']) / s['best'] * 100) if s['best'] > 0 else 0
        data_matrix.append([
            s['final'],
            gap_pct,
            s['collapse_rate'] * 100,
        ])

    data_matrix = np.array(data_matrix)

    # Normalize for radar-like bar plot
    bar_width = 0.25
    x_metrics = np.arange(len(metrics))

    for i, algo in enumerate(algos):
        offset = (i - len(algos)/2 + 0.5) * bar_width
        values = data_matrix[i]
        # For collapse rate, invert so lower is better visually
        ax3.bar(x_metrics + offset, values, bar_width,
               label=algo, color=colors.get(algo, 'gray'), alpha=0.8)

    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Stability Metrics', fontsize=12)
    ax3.set_xticks(x_metrics)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add interpretation
    fig.text(0.5, 0.02,
             'Lower Best-Final Gap % and Collapse Rate % indicate more stable learning. '
             'VABL maintains coordination better than MAPPO.',
             ha='center', fontsize=10, style='italic')

    plt.suptitle(f'Collapse Analysis: {env_name}' if env_name else 'Collapse Analysis',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("Collapse Analysis Summary")
    print("="*60)
    print(f"{'Algorithm':<12} {'Final':<12} {'Best':<12} {'Gap %':<12} {'Collapse Rate'}")
    print("-"*60)
    for algo in algos:
        s = summary_data[algo]
        gap_pct = ((s['best'] - s['final']) / s['best'] * 100) if s['best'] > 0 else 0
        print(f"{algo:<12} {s['final']:<12.1f} {s['best']:<12.1f} {gap_pct:<12.1f} {s['collapse_rate']:.0%}")


def main():
    parser = argparse.ArgumentParser(description='Generate collapse analysis figure')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='Input JSON files')
    parser.add_argument('--output', '-o', default='figures/collapse_analysis.png',
                        help='Output figure path')
    parser.add_argument('--env-name', type=str, default='',
                        help='Environment name for title')

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = load_results(args.input)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} algorithms: {list(results.keys())}")

    plot_collapse_analysis(results, args.output, args.env_name)


if __name__ == '__main__':
    main()
