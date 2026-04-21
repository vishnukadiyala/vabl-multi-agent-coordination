#!/usr/bin/env python
"""Ablation sweep script for VABL paper.

Runs systematic ablation studies across configurable hyperparameters.

Sweep configurations:
- lambda_sweep: aux_lambda values [0.0, 0.01, 0.1, 0.5, 1.0]
- belief_dim_sweep: hidden_dim values [16, 32, 64, 128]
- attention_heads_sweep: attention_heads values [1, 2, 4, 8]
- stop_gradient: stop_gradient_belief [False, True]
- no_attention: use_attention [True, False]
- no_aux_loss: use_aux_loss [True, False]

Usage:
    python -m marl_research.scripts.run_ablation_sweep \\
        --ablation lambda_sweep \\
        --seeds 3 \\
        --episodes 100 \\
        --device cuda
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from marl_research.utils.misc import set_seed, get_device


# Optional wandb support
_wandb = None
_wandb_run = None


def init_wandb(
    project: str = "vabl_icml2026",
    name: str = None,
    config: dict = None,
    group: str = None,
    entity: str = None,
) -> bool:
    """Initialize wandb logging.

    Args:
        project: Wandb project name
        name: Run name
        config: Config dict to log
        group: Run group for organizing related runs
        entity: Wandb entity (username or team). None uses personal account.
    """
    global _wandb, _wandb_run
    try:
        import wandb
        _wandb = wandb
        _wandb_run = wandb.init(
            project=project,
            entity=entity,  # None = personal account
            name=name,
            config=config,
            group=group,
            reinit=True,
        )
        return True
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return False


def log_wandb(metrics: dict, step: int = None):
    """Log metrics to wandb if available."""
    if _wandb is not None and _wandb_run is not None:
        _wandb.log(metrics, step=step)


def finish_wandb():
    """Finish wandb run."""
    global _wandb_run
    if _wandb is not None and _wandb_run is not None:
        _wandb.finish()
        _wandb_run = None


# Ablation configurations
ABLATIONS = {
    'lambda_sweep': {
        'param': 'aux_lambda',
        'values': [0.0, 0.01, 0.1, 0.5, 1.0],
        'description': 'Sweep auxiliary loss weight',
    },
    'belief_dim_sweep': {
        'param': 'hidden_dim',
        'values': [16, 32, 64, 128],
        'description': 'Sweep belief state dimension',
    },
    'attention_heads_sweep': {
        'param': 'attention_heads',
        'values': [1, 2, 4, 8],
        'description': 'Sweep number of attention heads',
    },
    'stop_gradient': {
        'param': 'stop_gradient_belief',
        'values': [False, True],
        'description': 'Ablation: detach beliefs in auxiliary loss',
    },
    'no_attention': {
        'param': 'use_attention',
        'values': [True, False],
        'description': 'Ablation: use mean pooling vs attention',
    },
    'no_aux_loss': {
        'param': 'use_aux_loss',
        'values': [True, False],
        'description': 'Ablation: enable/disable auxiliary loss',
    },
    'warmup_sweep': {
        'param': 'warmup_steps',
        'values': [0, 25, 50, 100, 200],
        'description': 'Sweep warmup steps before aux annealing',
    },
    'aux_decay_sweep': {
        'param': 'aux_decay_rate',
        'values': [0.99, 0.995, 0.999, 1.0],
        'description': 'Sweep auxiliary loss decay rate',
    },
}


def create_config(
    base_config: Dict[str, Any],
    param_name: str,
    param_value: Any,
) -> Dict[str, Any]:
    """Create a config dict with the specified parameter override.

    Args:
        base_config: Base configuration dict
        param_name: Parameter to override
        param_value: New value for parameter

    Returns:
        New config dict with override
    """
    config = base_config.copy()

    # Handle nested algorithm config
    if 'algorithm' not in config:
        config['algorithm'] = {}
    config['algorithm'] = config['algorithm'].copy()
    config['algorithm'][param_name] = param_value

    return config


def run_single_experiment(
    config: Dict[str, Any],
    seed: int,
    device: str,
    episodes: int,
) -> Dict[str, Any]:
    """Run a single experiment with the given config.

    Args:
        config: Configuration dict
        seed: Random seed
        device: Device to use
        episodes: Number of episodes

    Returns:
        Results dict with rewards, metrics
    """
    from omegaconf import OmegaConf

    from marl_research.algorithms.vabl import VABL
    from marl_research.environments import make_env
    from marl_research.environments.simple_env import SimpleCoordinationEnv

    set_seed(seed)
    device = get_device(device)

    # Create OmegaConf from dict
    cfg = OmegaConf.create({
        'environment': {
            'name': 'simple',
            'n_agents': 3,
            'obs_dim': 16,
            'n_actions': 5,
            'episode_limit': 50,
            'visibility_prob': 0.7,
        },
        'algorithm': {
            'name': 'vabl',
            'embed_dim': 64,
            'hidden_dim': config.get('algorithm', {}).get('hidden_dim', 128),
            'attention_dim': 64,
            'aux_lambda': config.get('algorithm', {}).get('aux_lambda', 0.1),
            'aux_hidden_dim': 64,
            'critic_hidden_dim': 128,
            'clip_param': 0.2,
            'ppo_epochs': 3,
            'value_clip': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'gae_lambda': 0.95,
            'warmup_steps': config.get('algorithm', {}).get('warmup_steps', 50),
            'attention_heads': config.get('algorithm', {}).get('attention_heads', 4),
            'stop_gradient_belief': config.get('algorithm', {}).get('stop_gradient_belief', False),
            'aux_decay_rate': config.get('algorithm', {}).get('aux_decay_rate', 0.995),
            'min_aux_lambda': config.get('algorithm', {}).get('min_aux_lambda', 0.05),
            'use_attention': config.get('algorithm', {}).get('use_attention', True),
            'use_aux_loss': config.get('algorithm', {}).get('use_aux_loss', True),
        },
        'training': {
            'lr': 0.0005,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 5000,
            'grad_clip': 10.0,
        },
        'experiment': {
            'seed': seed,
        },
        'hardware': {
            'device': str(device),
        },
    })

    # Create environment and algorithm
    env = SimpleCoordinationEnv(cfg)
    env_info = env.get_env_info()

    algorithm = VABL(
        config=cfg,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
        device=device,
    )

    # Simple replay buffer
    from marl_research.utils import ReplayBuffer
    buffer = ReplayBuffer(
        buffer_size=cfg.training.buffer_size,
        episode_limit=env_info.episode_limit,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
    )

    # Training loop
    rewards = []
    aux_losses = []
    aux_accuracies = []

    import torch.nn.functional as F

    for ep in range(episodes):
        obs, state, _ = env.reset()
        algorithm.init_hidden(batch_size=1)
        episode_reward = 0
        prev_actions = None

        for t in range(env_info.episode_limit):
            avail = env.get_available_actions()
            vis_masks = env.get_visibility_masks()

            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
            avail_t = torch.FloatTensor(np.array(avail)).unsqueeze(0).to(device)
            vis_t = torch.FloatTensor(vis_masks).unsqueeze(0).to(device)

            prev_actions_t = None
            if prev_actions is not None:
                prev_actions_t = torch.LongTensor(prev_actions).unsqueeze(0).to(device)

            with torch.no_grad():
                actions = algorithm.select_actions(
                    obs_t, avail_t, explore=True,
                    prev_actions=prev_actions_t,
                    visibility_masks=vis_t,
                )
            actions = actions.squeeze(0).cpu().numpy()

            next_obs, next_state, reward, done, info = env.step(actions.tolist())
            next_avail = env.get_available_actions()

            buffer.add_transition(
                obs=np.array(obs),
                state=np.array(state),
                actions=actions,
                reward=reward,
                next_obs=np.array(next_obs),
                next_state=np.array(next_state),
                done=done,
                available_actions=np.array(avail),
                next_available_actions=np.array(next_avail),
                visibility_masks=vis_masks,
            )

            episode_reward += reward
            prev_actions = actions.copy()
            obs = next_obs
            state = next_state

            if done:
                break

        rewards.append(episode_reward)

        # Training step
        if buffer.can_sample(cfg.training.batch_size):
            batch = buffer.sample(cfg.training.batch_size)
            metrics = algorithm.train_step(batch)
            aux_losses.append(metrics.get('aux_loss', 0.0))
            aux_accuracies.append(metrics.get('aux_accuracy', 0.0))

    env.close()

    return {
        'rewards': rewards,
        'aux_loss': aux_losses,
        'aux_accuracy': aux_accuracies,
        'final_reward_mean': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
        'best_reward_mean': np.max([np.mean(rewards[max(0, i-20):i+1]) for i in range(len(rewards))]),
    }


def run_ablation_sweep(
    ablation_name: str,
    seeds: int = 3,
    episodes: int = 100,
    device: str = 'auto',
    output_dir: str = 'results/ablations',
    use_wandb: bool = False,
    wandb_project: str = 'vabl_icml2026',
    wandb_entity: str = None,
) -> Dict[str, Any]:
    """Run a complete ablation sweep.

    Args:
        ablation_name: Name of ablation to run
        seeds: Number of seeds
        episodes: Episodes per run
        device: Device to use
        output_dir: Output directory

    Returns:
        Results dict
    """
    if ablation_name not in ABLATIONS:
        print(f"Unknown ablation: {ablation_name}")
        print(f"Available: {list(ABLATIONS.keys())}")
        return {}

    ablation = ABLATIONS[ablation_name]
    param_name = ablation['param']
    param_values = ablation['values']

    print(f"\n{'=' * 60}")
    print(f"Ablation Sweep: {ablation_name}")
    print(f"Description: {ablation['description']}")
    print(f"Parameter: {param_name}")
    print(f"Values: {param_values}")
    print(f"Seeds: {seeds}, Episodes: {episodes}")
    if use_wandb:
        print(f"Wandb: enabled (project={wandb_project})")
    print(f"{'=' * 60}\n")

    # Initialize wandb for the sweep
    if use_wandb:
        init_wandb(
            project=wandb_project,
            entity=wandb_entity,  # None = personal account
            name=f"ablation_{ablation_name}",
            config={
                'ablation': ablation_name,
                'param': param_name,
                'values': param_values,
                'seeds': seeds,
                'episodes': episodes,
            },
            group=f"ablation_{ablation_name}",
        )

    all_results = []
    base_config = {}

    for value in param_values:
        print(f"\n--- {param_name} = {value} ---")
        config = create_config(base_config, param_name, value)

        value_results = []
        for seed in range(seeds):
            print(f"  Seed {seed + 1}/{seeds}...", end=" ", flush=True)
            start_time = time.time()

            result = run_single_experiment(
                config=config,
                seed=seed,
                device=device,
                episodes=episodes,
            )

            elapsed = time.time() - start_time
            print(f"Final: {result['final_reward_mean']:.2f}, "
                  f"Best: {result['best_reward_mean']:.2f} ({elapsed:.1f}s)")

            value_results.append(result)

        # Aggregate across seeds
        final_means = [r['final_reward_mean'] for r in value_results]
        best_means = [r['best_reward_mean'] for r in value_results]

        all_results.append({
            'param_value': value,
            'results': value_results,
            'final_mean': np.mean(final_means),
            'final_std': np.std(final_means),
            'best_mean': np.mean(best_means),
            'best_std': np.std(best_means),
        })

        print(f"  Aggregated: Final={np.mean(final_means):.2f}+/-{np.std(final_means):.2f}, "
              f"Best={np.mean(best_means):.2f}+/-{np.std(best_means):.2f}")

        # Log aggregated results to wandb
        if use_wandb:
            log_wandb({
                f'{param_name}': value,
                f'final_mean': np.mean(final_means),
                f'final_std': np.std(final_means),
                f'best_mean': np.mean(best_means),
                f'best_std': np.std(best_means),
            })

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"Summary: {ablation_name}")
    print(f"{'=' * 60}")
    print(f"{'Value':<15} {'Final (mean+/-std)':<25} {'Best (mean+/-std)':<25}")
    print("-" * 65)
    for r in all_results:
        val_str = str(r['param_value'])
        final_str = f"{r['final_mean']:.2f} +/- {r['final_std']:.2f}"
        best_str = f"{r['best_mean']:.2f} +/- {r['best_std']:.2f}"
        print(f"{val_str:<15} {final_str:<25} {best_str:<25}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"ablation_{ablation_name}_{timestamp}.json"

    output_data = {
        'ablation': ablation_name,
        'description': ablation['description'],
        'param': param_name,
        'values': param_values,
        'seeds': seeds,
        'episodes': episodes,
        'timestamp': timestamp,
        'results': all_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\nResults saved to: {output_file}")

    # Finish wandb
    if use_wandb:
        finish_wandb()

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run VABL ablation sweeps"
    )
    parser.add_argument(
        "--ablation", "-a",
        choices=list(ABLATIONS.keys()),
        help="Ablation to run"
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=3,
        help="Number of seeds (default: 3)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=100,
        help="Episodes per run (default: 100)"
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/ablations",
        help="Output directory"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available ablations and exit"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        default="vabl_icml2026",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Wandb entity (username/team); defaults to your wandb default"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable ablations:")
        print("-" * 60)
        for name, ablation in ABLATIONS.items():
            print(f"  {name:<25} - {ablation['description']}")
            print(f"    Parameter: {ablation['param']}")
            print(f"    Values: {ablation['values']}")
        return

    if args.ablation is None:
        parser.error("--ablation is required when not using --list")

    run_ablation_sweep(
        ablation_name=args.ablation,
        seeds=args.seeds,
        episodes=args.episodes,
        device=args.device,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
