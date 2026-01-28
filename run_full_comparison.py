#!/usr/bin/env python
"""
Full Comparison Script for ICML 2026 Paper
==========================================

Runs all algorithms (VABL, MAPPO, QMIX) across all environments and generates
comparison plots and summary tables.

Usage:
    python run_full_comparison.py --episodes 100 --seeds 3 --device auto
    python run_full_comparison.py --quick  # Quick test with 50 episodes, 2 seeds
    python run_full_comparison.py --full   # Full paper experiments (200 episodes, 5 seeds)
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
ALGORITHMS = ['vabl', 'mappo', 'qmix']

ENVIRONMENTS = [
    {'env': 'simple', 'layout': None},
    {'env': 'overcooked', 'layout': 'cramped_room'},
    {'env': 'overcooked', 'layout': 'asymmetric_advantages'},
]

def run_experiment(algo, env, layout, episodes, seeds, device):
    """Run a single experiment configuration."""
    cmd = [
        sys.executable, '-m', 'marl_research.scripts.run_vabl_experiments',
        '--algorithm', algo,
        '--env', env,
        '--episodes', str(episodes),
        '--seeds', str(seeds),
        '--device', device,
    ]
    if layout:
        cmd.extend(['--layout', layout])

    env_name = f"{env}_{layout}" if layout else env
    print(f"\n{'='*70}")
    print(f"Running: {algo.upper()} on {env_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def find_latest_result(algo, env, layout, results_dir):
    """Find the most recent result file for a given configuration."""
    pattern = f"{algo}_{env}"
    if layout:
        pattern += f"_{layout}"
    pattern += "_*.json"

    files = sorted(results_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_results(results_dir):
    """Load all results and organize by environment and algorithm."""
    results = {}

    for env_config in ENVIRONMENTS:
        env = env_config['env']
        layout = env_config['layout']
        env_name = f"{env}_{layout}" if layout else env
        results[env_name] = {}

        for algo in ALGORITHMS:
            result_file = find_latest_result(algo, env, layout, results_dir)
            if result_file:
                with open(result_file) as f:
                    data = json.load(f)
                results[env_name][algo] = data

    return results


def generate_summary_table(results):
    """Generate a markdown summary table."""
    lines = []
    lines.append("# Full Comparison Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for env_name, env_results in results.items():
        lines.append(f"\n## {env_name.replace('_', ' ').title()}\n")
        lines.append("| Algorithm | Final Reward | Best Reward | Notes |")
        lines.append("|-----------|--------------|-------------|-------|")

        best_final = -float('inf')
        best_algo = None

        for algo, data in env_results.items():
            if 'summary' in data:
                final_mean = data['summary']['final_reward_mean']
                final_std = data['summary']['final_reward_std']
                best_mean = data['summary']['best_reward_mean']
                best_std = data['summary']['best_reward_std']

                if final_mean > best_final:
                    best_final = final_mean
                    best_algo = algo

                lines.append(f"| {algo.upper()} | {final_mean:.2f} ± {final_std:.2f} | {best_mean:.2f} ± {best_std:.2f} | |")

        if best_algo:
            lines.append(f"\n**Winner: {best_algo.upper()}**\n")

    return '\n'.join(lines)


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots for each environment."""
    import matplotlib.pyplot as plt

    colors = {'vabl': '#2ca02c', 'mappo': '#1f77b4', 'qmix': '#ff7f0e'}

    for env_name, env_results in results.items():
        if not env_results:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        def smooth(data, window=10):
            alpha = 2 / (window + 1)
            smoothed = np.zeros_like(data, dtype=float)
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
            return smoothed

        for algo, data in env_results.items():
            if 'results' not in data:
                continue

            # Get rewards from all seeds
            all_rewards = [r['rewards'] for r in data['results']]
            rewards = np.array(all_rewards)

            n_seeds, n_eps = rewards.shape
            episodes = np.arange(1, n_eps + 1)
            color = colors.get(algo, '#333333')

            # Smooth each seed
            smoothed = np.array([smooth(rewards[s]) for s in range(n_seeds)])

            # Compute mean and std
            mean = np.mean(smoothed, axis=0)
            std = np.std(smoothed, axis=0)

            # Plot
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)
            ax.plot(episodes, mean, color=color, linewidth=2.5, label=algo.upper())

        ax.set_xlabel('Episode', fontsize=14)
        ax.set_ylabel('Reward', fontsize=14)
        ax.set_title(f'{env_name.replace("_", " ").title()}: Algorithm Comparison', fontsize=14)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add summary stats
        summary_text = "Final Reward:\n"
        for algo, data in env_results.items():
            if 'summary' in data:
                final = data['summary']['final_reward_mean']
                std = data['summary']['final_reward_std']
                summary_text += f"{algo.upper()}: {final:.1f}±{std:.1f}\n"

        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', bbox=props)

        plt.tight_layout()
        output_path = output_dir / f"comparison_{env_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run full VABL comparison experiments')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per seed')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--quick', action='store_true', help='Quick test (50 eps, 2 seeds)')
    parser.add_argument('--full', action='store_true', help='Full paper experiments (200 eps, 5 seeds)')
    parser.add_argument('--skip-experiments', action='store_true', help='Skip running, just generate plots')
    parser.add_argument('--algorithms', nargs='+', default=ALGORITHMS,
                        choices=ALGORITHMS, help='Algorithms to run')
    parser.add_argument('--envs', nargs='+', default=None,
                        help='Environments to run (e.g., simple overcooked_cramped_room)')

    args = parser.parse_args()

    if args.quick:
        args.episodes = 50
        args.seeds = 2
    elif args.full:
        args.episodes = 200
        args.seeds = 5

    project_root = Path(__file__).parent
    results_dir = project_root / 'results'
    figures_dir = project_root / 'figures'
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Filter environments if specified
    envs_to_run = ENVIRONMENTS
    if args.envs:
        envs_to_run = []
        for env_spec in args.envs:
            if '_' in env_spec and env_spec != 'simple':
                parts = env_spec.split('_', 1)
                envs_to_run.append({'env': parts[0], 'layout': parts[1]})
            else:
                envs_to_run.append({'env': env_spec, 'layout': None})

    print("="*70)
    print("VABL Full Comparison - ICML 2026")
    print("="*70)
    print(f"Algorithms: {args.algorithms}")
    print(f"Environments: {[e['env'] + ('_' + e['layout'] if e['layout'] else '') for e in envs_to_run]}")
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {args.device}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if not args.skip_experiments:
        # Run all experiments
        total = len(args.algorithms) * len(envs_to_run)
        completed = 0
        failed = []

        for env_config in envs_to_run:
            for algo in args.algorithms:
                success = run_experiment(
                    algo,
                    env_config['env'],
                    env_config['layout'],
                    args.episodes,
                    args.seeds,
                    args.device
                )
                completed += 1
                if not success:
                    env_name = f"{env_config['env']}_{env_config['layout']}" if env_config['layout'] else env_config['env']
                    failed.append(f"{algo} on {env_name}")
                print(f"\nProgress: {completed}/{total} experiments completed")

        if failed:
            print(f"\nWarning: {len(failed)} experiments failed:")
            for f in failed:
                print(f"  - {f}")

    # Load results and generate outputs
    print("\n" + "="*70)
    print("Generating Summary and Plots")
    print("="*70)

    results = load_results(results_dir)

    # Generate summary table
    summary = generate_summary_table(results)
    summary_file = results_dir / f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_file}")
    print("\n" + summary)

    # Generate plots
    try:
        generate_comparison_plots(results, figures_dir)
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")

    print("\n" + "="*70)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
