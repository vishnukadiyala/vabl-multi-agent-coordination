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
from pathlib import Path
from datetime import datetime
import json
import time

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf

from marl_research.utils.misc import get_device, set_seed
from marl_research.utils import ReplayBuffer


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def create_config(env_name, algo_name='vabl', layout_name=None, horizon=100, aux_lambda=0.1,
                  use_attention=True, use_aux_loss=True):
    """Create experiment configuration."""
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
            'ppo_epochs': 3,
            'value_clip': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'gae_lambda': 0.95,
            'warmup_steps': 50,
            'attention_heads': 4,
            'stop_gradient_belief': False,
            'aux_decay_rate': 1.0,  # No decay for cleaner comparison
            'min_aux_lambda': 0.0,
            'use_attention': use_attention,
            'use_aux_loss': use_aux_loss,
            'use_value_norm': True,  # Enable value normalization
        }
    elif algo_name == 'qmix':
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
        # QMIX needs epsilon parameters in training config
        config['training']['epsilon_start'] = 1.0
        config['training']['epsilon_finish'] = 0.05
        config['training']['epsilon_anneal_time'] = 50000
        config['training']['target_update_interval'] = 200
    elif algo_name == 'mappo':
        config['algorithm'] = {
            'name': 'mappo',
            'hidden_dim': 64,
            'use_rnn': True,
            'rnn_hidden_dim': 64,
            'clip_param': 0.2,
            'ppo_epochs': 10,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'gae_lambda': 0.95,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    if env_name == 'overcooked':
        config['environment'] = {
            'name': 'overcooked',
            'layout_name': layout_name or 'cramped_room',
            'horizon': horizon,
            'num_agents': 2,
            'use_shaped_rewards': True,
            'shaped_reward_scale': 1.0,
        }
    elif env_name == 'simple':
        config['environment'] = {
            'name': 'simple',
            'n_agents': 3,
            'obs_dim': 16,
            'n_actions': 5,
            'episode_limit': horizon,
            'visibility_prob': 0.7,
        }

    return OmegaConf.create(config)


def run_experiment(config, n_episodes, seed, device):
    """Run a single experiment with given config."""
    from marl_research.environments import make_env
    from marl_research.algorithms import ALGORITHM_REGISTRY

    set_seed(seed)

    # Create environment
    env = make_env(config)
    env_info = env.get_env_info()

    # Create algorithm
    algo_name = config.algorithm.name
    algorithm = ALGORITHM_REGISTRY[algo_name](
        config=config,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
        device=device,
    )

    # Create replay buffer
    buffer = ReplayBuffer(
        buffer_size=config.training.buffer_size,
        episode_limit=env_info.episode_limit,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
    )

    # Training metrics
    episode_rewards = []
    aux_losses = []
    aux_accuracies = []

    start_time = time.time()

    for episode in range(n_episodes):
        obs, state, _ = env.reset()
        algorithm.init_hidden(batch_size=1)

        episode_reward = 0
        prev_actions = None

        for t in range(env_info.episode_limit):
            avail = env.get_available_actions()

            # Get visibility masks if available
            vis_masks = None
            if hasattr(env, 'get_visibility_masks'):
                vis_masks = env.get_visibility_masks()

            # Convert to tensors
            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
            avail_t = torch.FloatTensor(np.array(avail)).unsqueeze(0).to(device)

            vis_t = None
            if vis_masks is not None:
                vis_t = torch.FloatTensor(vis_masks).unsqueeze(0).to(device)

            prev_actions_t = None
            if prev_actions is not None:
                prev_actions_t = torch.LongTensor(prev_actions).unsqueeze(0).to(device)

            # Select actions
            with torch.no_grad():
                if algo_name == 'vabl':
                    actions = algorithm.select_actions(
                        obs_t, avail_t, explore=True,
                        prev_actions=prev_actions_t,
                        visibility_masks=vis_t,
                    )
                else:
                    actions = algorithm.select_actions(obs_t, avail_t, explore=True)

            actions = actions.squeeze(0).cpu().numpy()

            # Step environment
            next_obs, next_state, reward, done, info = env.step(actions.tolist())
            next_avail = env.get_available_actions()

            # Store transition
            buffer.add_transition(
                obs=np.array(obs),
                state=np.array(state),
                actions=actions,
                reward=reward,
                done=done,
                next_obs=np.array(next_obs),
                next_state=np.array(next_state),
                available_actions=np.array(avail),
                next_available_actions=np.array(next_avail),
                visibility_masks=vis_masks,
            )

            episode_reward += reward
            prev_actions = actions
            obs, state = next_obs, next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        # Train if buffer has enough data
        if buffer.can_sample(batch_size=1):
            batch = buffer.sample(batch_size=1)
            metrics = algorithm.train_step(batch)

            if algo_name == 'vabl':
                aux_losses.append(metrics.get('aux_loss', 0))
                aux_accuracies.append(metrics.get('aux_accuracy', 0))
                algorithm.update_on_episode_end(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            elapsed = time.time() - start_time

            if algo_name == 'vabl' and aux_losses:
                avg_aux = np.mean(aux_losses[-10:]) if aux_losses else 0
                avg_acc = np.mean(aux_accuracies[-10:]) if aux_accuracies else 0
                print(f'  Episode {episode+1:3d} | Reward: {avg_reward:7.1f} | '
                      f'Aux Loss: {avg_aux:.4f} | Aux Acc: {avg_acc:.1%} | Time: {elapsed:.1f}s')
            else:
                print(f'  Episode {episode+1:3d} | Reward: {avg_reward:7.1f} | Time: {elapsed:.1f}s')

    # Compute final metrics
    final_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
    best_reward = np.max(episode_rewards)

    results = {
        'rewards': episode_rewards,
        'final_reward': final_reward,
        'best_reward': best_reward,
        'final_aux_loss': np.mean(aux_losses[-10:]) if aux_losses else 0,
        'final_aux_acc': np.mean(aux_accuracies[-10:]) if aux_accuracies else 0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='VABL Experiment Runner')
    parser.add_argument('--env', type=str, default='simple', choices=['overcooked', 'simple'],
                        help='Environment to use')
    parser.add_argument('--layout', type=str, default='cramped_room',
                        help='Layout for Overcooked (cramped_room, asymmetric_advantages, etc.)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per seed')
    parser.add_argument('--seeds', type=int, default=1, help='Number of random seeds')
    parser.add_argument('--horizon', type=int, default=100, help='Episode horizon')
    parser.add_argument('--algorithm', type=str, default='vabl', choices=['vabl', 'qmix', 'mappo'],
                        help='Algorithm to run')
    parser.add_argument('--aux_lambda', type=float, default=0.1, help='Auxiliary loss coefficient')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name suffix')
    parser.add_argument('--full', action='store_true', help='Run full paper experiments')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--no-attn', action='store_true', help='Disable attention mechanism')
    parser.add_argument('--no-aux', action='store_true', help='Disable auxiliary loss')

    args = parser.parse_args()

    device = get_device(args.device)

    print('=' * 70)
    print(f'VABL Experiment Runner')
    print('=' * 70)
    print(f'Environment: {args.env}')
    if args.env == 'overcooked':
        print(f'Layout: {args.layout}')
    print(f'Algorithm: {args.algorithm}')
    print(f'Episodes: {args.episodes}')
    print(f'Seeds: {args.seeds}')
    print(f'Device: {device}')
    print(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    config = create_config(
        args.env, args.algorithm, args.layout, args.horizon, args.aux_lambda,
        use_attention=not args.no_attn, use_aux_loss=not args.no_aux
    )

    all_results = []
    for seed in range(args.seeds):
        print(f'\n--- Seed {seed} ---')
        results = run_experiment(config, args.episodes, seed, device)
        all_results.append(results)

    # Summary
    print(f'\n{"="*70}')
    print('Final Results')
    print('=' * 70)

    final_rewards = [r['final_reward'] for r in all_results]
    best_rewards = [r['best_reward'] for r in all_results]

    print(f'  Final Reward: {np.mean(final_rewards):.2f} +/- {np.std(final_rewards):.2f}')
    print(f'  Best Reward:  {np.mean(best_rewards):.2f} +/- {np.std(best_rewards):.2f}')

    if args.algorithm == 'vabl':
        final_aux_loss = [r['final_aux_loss'] for r in all_results]
        final_aux_acc = [r['final_aux_acc'] for r in all_results]
        print(f'  Aux Loss: {np.mean(final_aux_loss):.4f} +/- {np.std(final_aux_loss):.4f}')
        print(f'  Aux Acc: {np.mean(final_aux_acc):.1%} +/- {np.std(final_aux_acc):.1%}')

    # Save results
    results_dir = get_project_root() / 'results'
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_suffix = f"_{args.exp_name}" if args.exp_name else ""
    layout_suffix = f"_{args.layout}" if args.env == 'overcooked' else ""

    filename = results_dir / f"{args.algorithm}_{args.env}{layout_suffix}{name_suffix}_{timestamp}.json"

    save_data = {
        'algorithm': args.algorithm,
        'env': args.env,
        'layout': args.layout if args.env == 'overcooked' else None,
        'episodes': args.episodes,
        'seeds': args.seeds,
        'results': all_results,
        'summary': {
            'final_reward_mean': float(np.mean(final_rewards)),
            'final_reward_std': float(np.std(final_rewards)),
            'best_reward_mean': float(np.mean(best_rewards)),
            'best_reward_std': float(np.std(best_rewards)),
        }
    }

    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f'\nResults saved to {filename}')
    print(f'\nEnd time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)


if __name__ == '__main__':
    main()
