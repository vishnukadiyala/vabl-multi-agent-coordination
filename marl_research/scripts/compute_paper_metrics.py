#!/usr/bin/env python
"""
Compute Paper Metrics for VABL ICML 2026
========================================

Computes the metrics that matter for reviewers:
1. AUC of learning curve (sample efficiency)
2. Success rate / completion rate
3. Stability index (fraction of seeds reaching threshold)
4. Time-to-threshold (median ± IQR)
5. Best-Final gap analysis

Usage:
    python -m marl_research.scripts.compute_paper_metrics --input results/*.json
    python -m marl_research.scripts.compute_paper_metrics --input results/*.json --threshold 50
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import integrate


def compute_auc(rewards: np.ndarray) -> float:
    """Compute Area Under the Curve (normalized by episode count).

    Args:
        rewards: Array of rewards [n_episodes]

    Returns:
        Normalized AUC (average reward across all episodes)
    """
    # Use trapezoidal integration, normalized by number of episodes
    episodes = np.arange(len(rewards))
    auc = integrate.trapezoid(rewards, episodes)
    return auc / len(rewards)


def compute_success_rate(rewards: np.ndarray, threshold: float) -> float:
    """Compute fraction of episodes that achieved >= threshold reward.

    Args:
        rewards: Array of rewards [n_episodes]
        threshold: Success threshold

    Returns:
        Success rate [0, 1]
    """
    return np.mean(rewards >= threshold)


def compute_stability_index(
    seed_rewards: List[np.ndarray],
    threshold: float,
    by_episode: Optional[int] = None
) -> Tuple[float, int, int]:
    """Compute stability index: fraction of seeds reaching threshold.

    Args:
        seed_rewards: List of reward arrays, one per seed
        threshold: Reward threshold to consider "successful"
        by_episode: If set, only consider first N episodes

    Returns:
        (stability_index, n_successful_seeds, n_total_seeds)
    """
    n_successful = 0
    for rewards in seed_rewards:
        if by_episode:
            rewards = rewards[:by_episode]
        if np.max(rewards) >= threshold:
            n_successful += 1

    stability = n_successful / len(seed_rewards)
    return stability, n_successful, len(seed_rewards)


def compute_time_to_threshold(
    seed_rewards: List[np.ndarray],
    threshold: float,
    window: int = 10
) -> Dict[str, float]:
    """Compute time (episode) to first reach threshold (using rolling average).

    Args:
        seed_rewards: List of reward arrays, one per seed
        threshold: Reward threshold
        window: Rolling window for smoothing

    Returns:
        Dict with median, q25, q75, mean, std of time-to-threshold
        Returns inf for seeds that never reach threshold
    """
    times = []
    for rewards in seed_rewards:
        # Compute rolling average
        if len(rewards) < window:
            rolling = rewards
        else:
            rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')

        # Find first episode where rolling average >= threshold
        reached = np.where(rolling >= threshold)[0]
        if len(reached) > 0:
            times.append(reached[0] + window // 2)  # Adjust for window offset
        else:
            times.append(float('inf'))

    times = np.array(times)
    finite_times = times[np.isfinite(times)]

    return {
        'median': float(np.median(finite_times)) if len(finite_times) > 0 else float('inf'),
        'q25': float(np.percentile(finite_times, 25)) if len(finite_times) > 0 else float('inf'),
        'q75': float(np.percentile(finite_times, 75)) if len(finite_times) > 0 else float('inf'),
        'mean': float(np.mean(finite_times)) if len(finite_times) > 0 else float('inf'),
        'std': float(np.std(finite_times)) if len(finite_times) > 0 else float('inf'),
        'n_reached': int(len(finite_times)),
        'n_total': int(len(times)),
        'reach_rate': float(len(finite_times) / len(times)),
    }


def compute_best_final_gap(rewards: np.ndarray, final_window: int = 20) -> Dict[str, float]:
    """Analyze the gap between best and final performance.

    Args:
        rewards: Array of rewards [n_episodes]
        final_window: Number of final episodes to average

    Returns:
        Dict with best, final, gap, gap_ratio
    """
    best = float(np.max(rewards))
    final = float(np.mean(rewards[-final_window:]))
    gap = best - final
    gap_ratio = gap / best if best > 0 else 0

    return {
        'best': best,
        'final': final,
        'gap': gap,
        'gap_ratio': gap_ratio,  # Higher = more collapse
    }


def compute_all_metrics(
    seed_rewards: List[np.ndarray],
    success_threshold: float = 50.0,
    final_window: int = 20
) -> Dict:
    """Compute all paper metrics for a set of seed results.

    Args:
        seed_rewards: List of reward arrays, one per seed
        success_threshold: Threshold for success/stability metrics
        final_window: Window for final reward calculation

    Returns:
        Dict with all metrics
    """
    all_rewards = np.array(seed_rewards)
    n_seeds, n_episodes = all_rewards.shape

    # Per-seed metrics
    aucs = [compute_auc(r) for r in seed_rewards]
    success_rates = [compute_success_rate(r, success_threshold) for r in seed_rewards]
    gaps = [compute_best_final_gap(r, final_window) for r in seed_rewards]

    # Aggregate metrics
    mean_rewards = np.mean(all_rewards, axis=0)

    stability = compute_stability_index(seed_rewards, success_threshold)
    time_to_thresh = compute_time_to_threshold(seed_rewards, success_threshold)

    return {
        # Sample efficiency
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),

        # Success rate
        'success_rate_mean': float(np.mean(success_rates)),
        'success_rate_std': float(np.std(success_rates)),

        # Stability
        'stability_index': stability[0],
        'n_successful_seeds': stability[1],
        'n_total_seeds': stability[2],

        # Time to threshold
        'time_to_threshold': time_to_thresh,

        # Best-Final gap (collapse indicator)
        'best_mean': float(np.mean([g['best'] for g in gaps])),
        'best_std': float(np.std([g['best'] for g in gaps])),
        'final_mean': float(np.mean([g['final'] for g in gaps])),
        'final_std': float(np.std([g['final'] for g in gaps])),
        'gap_mean': float(np.mean([g['gap'] for g in gaps])),
        'gap_std': float(np.std([g['gap'] for g in gaps])),
        'gap_ratio_mean': float(np.mean([g['gap_ratio'] for g in gaps])),

        # Raw data for plotting
        'n_seeds': n_seeds,
        'n_episodes': n_episodes,
        'threshold_used': success_threshold,
    }


def load_and_analyze(input_files: List[str], threshold: float = 50.0) -> Dict[str, Dict]:
    """Load result files and compute metrics for each algorithm.

    Args:
        input_files: List of JSON file paths
        threshold: Success threshold

    Returns:
        Dict mapping algorithm names to their metrics
    """
    from glob import glob

    results = {}

    for pattern in input_files:
        files = glob(pattern)
        if not files and Path(pattern).exists():
            files = [pattern]

        for fpath in files:
            with open(fpath) as f:
                data = json.load(f)

            # Extract algorithm name and rewards
            algo = data.get('algorithm', Path(fpath).stem.split('_')[0])
            env = data.get('env', 'unknown')
            layout = data.get('layout', '')

            key = algo.upper()
            if layout:
                key = f"{key}_{env}_{layout}"
            else:
                key = f"{key}_{env}"

            # Get rewards from results
            if 'results' in data:
                seed_rewards = [np.array(r['rewards']) for r in data['results']]
            elif 'rewards' in data:
                seed_rewards = [np.array(data['rewards'])]
            else:
                continue

            if seed_rewards:
                results[key] = compute_all_metrics(seed_rewards, threshold)
                results[key]['source_file'] = str(fpath)

    return results


def format_metrics_table(metrics: Dict[str, Dict], format: str = 'markdown') -> str:
    """Format metrics as a table.

    Args:
        metrics: Dict mapping algorithm names to metrics
        format: 'markdown' or 'latex'

    Returns:
        Formatted table string
    """
    lines = []

    if format == 'markdown':
        # Header
        lines.append("| Algorithm | AUC | Final | Best | Gap | Stability | Time-to-Thresh |")
        lines.append("|-----------|-----|-------|------|-----|-----------|----------------|")

        for algo, m in sorted(metrics.items()):
            ttt = m['time_to_threshold']
            ttt_str = f"{ttt['median']:.0f} ({ttt['reach_rate']:.0%})" if ttt['median'] != float('inf') else "N/A"

            lines.append(
                f"| {algo} | {m['auc_mean']:.1f}±{m['auc_std']:.1f} | "
                f"{m['final_mean']:.1f}±{m['final_std']:.1f} | "
                f"{m['best_mean']:.1f}±{m['best_std']:.1f} | "
                f"{m['gap_mean']:.1f} ({m['gap_ratio_mean']:.0%}) | "
                f"{m['stability_index']:.0%} | {ttt_str} |"
            )

    elif format == 'latex':
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(r"Algorithm & AUC & Final & Best & Gap & Stability & TTT \\")
        lines.append(r"\midrule")

        for algo, m in sorted(metrics.items()):
            ttt = m['time_to_threshold']
            ttt_str = f"{ttt['median']:.0f}" if ttt['median'] != float('inf') else "--"

            lines.append(
                f"{algo} & {m['auc_mean']:.1f}$\\pm${m['auc_std']:.1f} & "
                f"{m['final_mean']:.1f}$\\pm${m['final_std']:.1f} & "
                f"{m['best_mean']:.1f}$\\pm${m['best_std']:.1f} & "
                f"{m['gap_ratio_mean']:.0%} & "
                f"{m['stability_index']:.0%} & {ttt_str} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Compute paper metrics')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                        help='Input JSON files or glob patterns')
    parser.add_argument('--threshold', '-t', type=float, default=50.0,
                        help='Success threshold for stability metrics')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for metrics')
    parser.add_argument('--format', choices=['markdown', 'latex'], default='markdown',
                        help='Table output format')

    args = parser.parse_args()

    print("="*70)
    print("VABL Paper Metrics Calculator")
    print("="*70)

    metrics = load_and_analyze(args.input, args.threshold)

    if not metrics:
        print("No results found!")
        return

    print(f"\nAnalyzed {len(metrics)} algorithm/environment combinations\n")

    # Print table
    table = format_metrics_table(metrics, args.format)
    print(table)

    # Print detailed analysis
    print("\n" + "="*70)
    print("Detailed Analysis")
    print("="*70)

    for algo, m in sorted(metrics.items()):
        print(f"\n{algo}:")
        print(f"  Sample Efficiency (AUC): {m['auc_mean']:.2f} ± {m['auc_std']:.2f}")
        print(f"  Final Reward: {m['final_mean']:.2f} ± {m['final_std']:.2f}")
        print(f"  Best Reward: {m['best_mean']:.2f} ± {m['best_std']:.2f}")
        print(f"  Best-Final Gap: {m['gap_mean']:.2f} ({m['gap_ratio_mean']:.1%} collapse)")
        print(f"  Stability: {m['stability_index']:.0%} seeds reached threshold {m['threshold_used']}")
        ttt = m['time_to_threshold']
        if ttt['median'] != float('inf'):
            print(f"  Time-to-Threshold: {ttt['median']:.0f} episodes (median), {ttt['reach_rate']:.0%} reached")
        else:
            print(f"  Time-to-Threshold: N/A ({ttt['reach_rate']:.0%} reached threshold)")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {output_path}")


if __name__ == '__main__':
    main()
