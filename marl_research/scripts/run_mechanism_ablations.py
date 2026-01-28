#!/usr/bin/env python
"""
Mechanism Ablation Study for VABL ICML 2026
============================================

Runs the 4 key ablations that prove VABL's mechanism:
1. No attention (replace attention context with zeros)
2. No aux loss (λ=0 throughout)
3. Shuffle teammate actions (break causal signal)
4. Visibility stress-test (reduce visibility probability)

Usage:
    python -m marl_research.scripts.run_mechanism_ablations --env simple --episodes 100 --seeds 3
    python -m marl_research.scripts.run_mechanism_ablations --ablation no_attention --episodes 50
"""

import argparse
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import copy

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import torch
import numpy as np
from omegaconf import OmegaConf

from marl_research.utils.misc import get_device, set_seed


# Ablation configurations
ABLATIONS = {
    'full_vabl': {
        'name': 'VABL (Full)',
        'description': 'Full VABL with attention and auxiliary loss',
        'use_attention': True,
        'use_aux_loss': True,
        'aux_lambda': 0.05,
        'shuffle_actions': False,
        'visibility_prob': 1.0,
    },
    'no_attention': {
        'name': 'No Attention',
        'description': 'Replace attention context with zeros (mean pooling ablation)',
        'use_attention': False,
        'use_aux_loss': True,
        'aux_lambda': 0.05,
        'shuffle_actions': False,
        'visibility_prob': 1.0,
    },
    'no_aux_loss': {
        'name': 'No Aux Loss',
        'description': 'Disable auxiliary teammate prediction loss (λ=0)',
        'use_attention': True,
        'use_aux_loss': False,
        'aux_lambda': 0.0,
        'shuffle_actions': False,
        'visibility_prob': 1.0,
    },
    'shuffle_actions': {
        'name': 'Shuffle Actions',
        'description': 'Randomly shuffle teammate actions (breaks causal signal)',
        'use_attention': True,
        'use_aux_loss': True,
        'aux_lambda': 0.05,
        'shuffle_actions': True,
        'visibility_prob': 1.0,
    },
    'visibility_50': {
        'name': 'Visibility 50%',
        'description': 'Reduce visibility probability to 50%',
        'use_attention': True,
        'use_aux_loss': True,
        'aux_lambda': 0.05,
        'shuffle_actions': False,
        'visibility_prob': 0.5,
    },
    'visibility_25': {
        'name': 'Visibility 25%',
        'description': 'Reduce visibility probability to 25%',
        'use_attention': True,
        'use_aux_loss': True,
        'aux_lambda': 0.05,
        'shuffle_actions': False,
        'visibility_prob': 0.25,
    },
    'no_attention_no_aux': {
        'name': 'No Attn + No Aux',
        'description': 'Disable both attention and auxiliary loss',
        'use_attention': False,
        'use_aux_loss': False,
        'aux_lambda': 0.0,
        'shuffle_actions': False,
        'visibility_prob': 1.0,
    },
}


def create_ablation_config(env_name: str, ablation_config: Dict, layout: Optional[str] = None) -> Dict:
    """Create experiment configuration with ablation settings."""
    config = {
        'training': {
            'lr': 0.0005,
            'gamma': 0.99,
            'grad_clip': 10.0,
            'batch_size': 32,
            'buffer_size': 1000,
        },
        'experiment': {
            'seed': 42,
        },
        'algorithm': {
            'name': 'vabl',
            'embed_dim': 64,
            'hidden_dim': 128,
            'attention_dim': 64,
            'aux_hidden_dim': 64,
            'critic_hidden_dim': 128,
            'aux_lambda': ablation_config['aux_lambda'],
            'clip_param': 0.2,
            'ppo_epochs': 10,
            'value_clip': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'gae_lambda': 0.95,
            'warmup_steps': 50,
            'attention_heads': 4,
            'stop_gradient_belief': False,
            'aux_decay_rate': 1.0,
            'min_aux_lambda': 0.0,
            'use_attention': ablation_config['use_attention'],
            'use_aux_loss': ablation_config['use_aux_loss'],
            'use_value_norm': True,
            'use_orthogonal_init': True,
            'actor_lr': 0.0005,
            'critic_lr': 0.0005,
        }
    }

    if env_name == 'overcooked':
        config['environment'] = {
            'name': 'overcooked',
            'layout': layout or 'asymmetric_advantages',
            'horizon': 100,
        }
    else:
        config['environment'] = {
            'name': 'simple',
            'n_agents': 3,
            'horizon': 50,
        }

    return config


def run_ablation_experiment(
    config: Dict,
    ablation_config: Dict,
    n_episodes: int,
    seed: int,
    device: torch.device
) -> Dict:
    """Run a single ablation experiment."""
    from marl_research.algorithms import ALGORITHM_REGISTRY
    from marl_research.environments import make_env
    from marl_research.utils import ReplayBuffer
    import torch.nn.functional as F

    set_seed(seed)
    cfg = OmegaConf.create(config)
    cfg.experiment.seed = seed

    # Create environment
    env = make_env(cfg)
    env_info = env.get_env_info()

    # Create algorithm
    algorithm = ALGORITHM_REGISTRY[cfg.algorithm.name](
        config=cfg,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
        device=device,
    )

    # Set ablation flags on agent
    algorithm.agent.use_attention = ablation_config['use_attention']
    algorithm.use_aux_loss = ablation_config['use_aux_loss']

    buffer = ReplayBuffer(
        buffer_size=cfg.training.buffer_size,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
    )

    rewards = []
    aux_losses = []
    aux_accs = []

    visibility_prob = ablation_config['visibility_prob']
    shuffle_actions = ablation_config['shuffle_actions']

    for episode in range(n_episodes):
        obs, state = env.reset()
        algorithm.init_hidden(batch_size=1)
        done = False
        episode_reward = 0
        step = 0
        prev_actions = None

        episode_buffer = {
            'obs': [], 'state': [], 'actions': [], 'rewards': [],
            'next_obs': [], 'next_state': [], 'dones': [],
            'available_actions': [], 'visibility_masks': [],
        }

        while not done:
            available_actions = env.get_available_actions()

            # Apply visibility stress test
            if visibility_prob < 1.0:
                visibility_masks = (np.random.random((env_info.n_agents, env_info.n_agents - 1)) < visibility_prob).astype(np.float32)
            else:
                visibility_masks = np.ones((env_info.n_agents, env_info.n_agents - 1), dtype=np.float32)

            # Shuffle teammate actions if ablation is enabled
            input_prev_actions = prev_actions
            if shuffle_actions and prev_actions is not None:
                # Shuffle the teammate actions randomly
                input_prev_actions = prev_actions.copy()
                np.random.shuffle(input_prev_actions)

            obs_t = torch.FloatTensor(obs).to(device)
            avail_t = torch.FloatTensor(available_actions).to(device)
            vis_t = torch.FloatTensor(visibility_masks).to(device)

            with torch.no_grad():
                actions, _ = algorithm.select_actions(
                    obs_t, avail_t,
                    prev_actions=input_prev_actions,
                    visibility_masks=vis_t
                )

            actions_np = actions.cpu().numpy()
            next_obs, next_state, reward, done, info = env.step(actions_np)

            episode_buffer['obs'].append(obs)
            episode_buffer['state'].append(state)
            episode_buffer['actions'].append(actions_np)
            episode_buffer['rewards'].append(reward)
            episode_buffer['next_obs'].append(next_obs)
            episode_buffer['next_state'].append(next_state)
            episode_buffer['dones'].append(done)
            episode_buffer['available_actions'].append(available_actions)
            episode_buffer['visibility_masks'].append(visibility_masks)

            episode_reward += reward
            obs = next_obs
            state = next_state
            prev_actions = actions_np
            step += 1

        rewards.append(episode_reward)

        # Add episode to buffer and train
        if len(episode_buffer['obs']) > 0:
            buffer.add_episode(episode_buffer)

            if buffer.can_sample(cfg.training.batch_size):
                batch = buffer.sample(cfg.training.batch_size)
                metrics = algorithm.train_step(batch)
                aux_losses.append(metrics.get('aux_loss', 0))
                aux_accs.append(metrics.get('aux_accuracy', 0))

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"  Episode {episode+1:3d} | Reward: {avg_reward:7.1f}", end='')
            if aux_losses:
                print(f" | Aux Loss: {aux_losses[-1]:.4f}", end='')
            print()

    return {
        'rewards': rewards,
        'final_reward': float(np.mean(rewards[-20:])),
        'best_reward': float(np.max(rewards)),
        'aux_losses': aux_losses,
        'aux_accs': aux_accs,
    }


def main():
    parser = argparse.ArgumentParser(description='VABL Mechanism Ablation Study')
    parser.add_argument('--env', type=str, default='simple', choices=['simple', 'overcooked'])
    parser.add_argument('--layout', type=str, default='asymmetric_advantages')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--ablation', type=str, default=None,
                        choices=list(ABLATIONS.keys()),
                        help='Run single ablation (default: run all)')
    parser.add_argument('--list', action='store_true', help='List available ablations')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable ablations:")
        for key, cfg in ABLATIONS.items():
            print(f"  {key}: {cfg['description']}")
        return

    device = get_device(args.device)

    # Select ablations to run
    if args.ablation:
        ablations_to_run = {args.ablation: ABLATIONS[args.ablation]}
    else:
        ablations_to_run = ABLATIONS

    print("="*70)
    print("VABL Mechanism Ablation Study")
    print("="*70)
    print(f"Environment: {args.env}")
    if args.env == 'overcooked':
        print(f"Layout: {args.layout}")
    print(f"Episodes: {args.episodes}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    print(f"Ablations: {list(ablations_to_run.keys())}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    all_results = {}

    for ablation_name, ablation_cfg in ablations_to_run.items():
        print(f"\n{'='*70}")
        print(f"Running: {ablation_cfg['name']}")
        print(f"  {ablation_cfg['description']}")
        print("="*70)

        config = create_ablation_config(args.env, ablation_cfg, args.layout)
        ablation_results = []

        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            results = run_ablation_experiment(
                config, ablation_cfg, args.episodes, seed, device
            )
            ablation_results.append(results)

        # Aggregate results
        final_rewards = [r['final_reward'] for r in ablation_results]
        best_rewards = [r['best_reward'] for r in ablation_results]

        all_results[ablation_name] = {
            'config': ablation_cfg,
            'results': ablation_results,
            'summary': {
                'final_mean': float(np.mean(final_rewards)),
                'final_std': float(np.std(final_rewards)),
                'best_mean': float(np.mean(best_rewards)),
                'best_std': float(np.std(best_rewards)),
            }
        }

        print(f"\n{ablation_cfg['name']} Results:")
        print(f"  Final: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")
        print(f"  Best:  {np.mean(best_rewards):.2f} ± {np.std(best_rewards):.2f}")

    # Save results
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'ablations'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layout_suffix = f"_{args.layout}" if args.env == 'overcooked' else ""
    filename = results_dir / f"mechanism_ablation_{args.env}{layout_suffix}_{timestamp}.json"

    save_data = {
        'env': args.env,
        'layout': args.layout if args.env == 'overcooked' else None,
        'episodes': args.episodes,
        'seeds': args.seeds,
        'timestamp': timestamp,
        'results': all_results,
    }

    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n{'='*70}")
    print("Summary Table")
    print("="*70)
    print(f"{'Ablation':<25} {'Final Reward':<20} {'Best Reward':<20}")
    print("-"*70)
    for name, data in all_results.items():
        s = data['summary']
        print(f"{ABLATIONS[name]['name']:<25} {s['final_mean']:>6.1f} ± {s['final_std']:<6.1f}   {s['best_mean']:>6.1f} ± {s['best_std']:<6.1f}")

    print(f"\nResults saved to: {filename}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
