"""
VABL Experiment Runner for ICML 2026 Paper Reproduction
========================================================

Usage:
    conda activate icml2026
    python -m marl_research.scripts.run_vabl_experiments [--env ENV] [--layout LAYOUT] [--episodes N] [--seeds N]

Examples:
    python -m marl_research.scripts.run_vabl_experiments --env overcooked --layout cramped_room --episodes 100
    python -m marl_research.scripts.run_vabl_experiments --env simple --episodes 200 --seeds 3
    python -m marl_research.scripts.run_vabl_experiments --full  # Run full paper experiments
    python -m marl_research.scripts.run_vabl_experiments --device cuda  # Run on GPU
"""

import argparse
import warnings
import logging
import io
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime
import json

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import torch
import numpy as np
from omegaconf import OmegaConf

from marl_research.utils.misc import get_device, set_seed


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from marl_research/scripts to project root
    return Path(__file__).parent.parent.parent


def create_config(env_name, algo_name='vabl', layout_name=None, horizon=100, aux_lambda=0.5, 
                  use_attention=True, use_aux_loss=True):
    """Create experiment configuration."""
    config = {
        'training': {
            'lr': 0.001,
            'gamma': 0.99,
            'grad_clip': 10.0,
            'batch_size': 16,
            'buffer_size': 100,
            'epsilon_start': 1.0,
            'epsilon_finish': 0.05,
            'epsilon_anneal_time': 500,
            'target_update_interval': 200,
        },
        'experiment': {
            'seed': 42,
        }
    }

    if algo_name == 'vabl':
        config['algorithm'] = {
            'name': 'vabl',
            'embed_dim': 64,
            'hidden_dim': 128,
            'attention_dim': 64,
            'aux_hidden_dim': 64,
            'critic_hidden_dim': 128,
            'aux_lambda': aux_lambda,
            'clip_param': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.05,
            'gae_lambda': 0.95,
            # Ablation flags
            'use_attention': use_attention,
            'use_aux_loss': use_aux_loss,
        }
    elif algo_name == 'qmix':
        # ... (rest of qmix config)
        config['algorithm'] = {
            'name': 'qmix',
            'agent_network': {
                'hidden_dim': 64,
                'rnn_hidden_dim': 64,
            },
            'mixing_network': {
                'embed_dim': 32,
                'hypernet_hidden_dim': 64,
            },
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    if env_name == 'overcooked':
        # ... (rest of env config)
        config['environment'] = {
            'name': 'overcooked',
            'layout_name': layout_name or 'cramped_room',
            'horizon': horizon,
            'num_agents': 2,
        }
    elif env_name == 'simple':
       # ...
        config['environment'] = {
            'name': 'simple',
            'n_agents': 3,
            'obs_dim': 16,
            'n_actions': 5,
            'episode_limit': horizon,
            'visibility_prob': 0.7,
        }

    return OmegaConf.create(config)

# ... (rest of the file)

def main():
    parser = argparse.ArgumentParser(description='VABL Experiment Runner')
    # ... (existing args)
    parser.add_argument('--env', type=str, default='simple', choices=['overcooked', 'simple'],
                        help='Environment to use')
    parser.add_argument('--layout', type=str, default='cramped_room',
                        help='Layout for Overcooked (cramped_room, asymmetric_advantages, etc.)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per seed')
    parser.add_argument('--seeds', type=int, default=1, help='Number of random seeds')
    parser.add_argument('--horizon', type=int, default=100, help='Episode horizon')
    parser.add_argument('--algorithm', type=str, default='vabl', choices=['vabl', 'qmix'], help='Algorithm to run')
    parser.add_argument('--aux_lambda', type=float, default=0.5, help='Auxiliary loss coefficient (lambda)')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name suffix for saving results')
    parser.add_argument('--full', action='store_true', help='Run full paper experiments')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (auto will use GPU if available)')
    
    # Ablation arguments
    parser.add_argument('--no-attn', action='store_true', help='Disable attention mechanism (use mean pooling)')
    parser.add_argument('--no-aux', action='store_true', help='Disable auxiliary loss')

    args = parser.parse_args()

    # ...

    if args.full:
        # ... (full experiments logic, might need updates if we want ablations there too, but let's stick to single run for now)
        pass 
    else:
        # Single experiment
        # ...
        config = create_config(args.env, args.algorithm, args.layout, args.horizon, args.aux_lambda, 
                             use_attention=not args.no_attn, use_aux_loss=not args.no_aux)

        # ...

        all_results = []
        for seed in range(args.seeds):
            print(f'\n--- Seed {seed} ---')
            results = run_experiment(config, args.episodes, seed, device)
            all_results.append(results)

        # Summary
        print(f'\n{"="*70}')
        print('Final Results')
        print(f'{"="*70}')

        final_rewards = [r['final_reward'] for r in all_results]
        final_aux_loss = [r['final_aux_loss'] for r in all_results]
        final_aux_acc = [r['final_aux_acc'] for r in all_results]

        print(f'  Reward: {np.mean(final_rewards):.2f} +/- {np.std(final_rewards):.2f}')
        print(f'  Aux Loss: {np.mean(final_aux_loss):.4f} +/- {np.std(final_aux_loss):.4f}')
        print(f'  Aux Acc: {np.mean(final_aux_acc):.1%} +/- {np.std(final_aux_acc):.1%}')

    # Save results to project root
    results_dir = get_project_root() / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_suffix = f"_{args.exp_name}" if args.exp_name else ""
    
    if args.full:
        filename = results_dir / f"vabl_full_results{name_suffix}_{timestamp}.json"
    else:
        filename = results_dir / f"vabl_{args.env}{name_suffix}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(full_experiment_results if args.full else [{'env': args.env, 'layout': args.layout, 'results': all_results}], f, indent=2)
    
    print(f"\nResults saved to {filename}")

    print(f'\nEnd time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*70)


if __name__ == '__main__':
    main()
