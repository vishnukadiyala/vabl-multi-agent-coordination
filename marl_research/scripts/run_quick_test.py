"""Quick training test for VABL on Overcooked.

Usage:
    python -m marl_research.scripts.run_quick_test
    python -m marl_research.scripts.run_quick_test --device cuda
    python -m marl_research.scripts.run_quick_test --wandb  # Enable wandb logging
"""
import argparse
import warnings
import logging
import io
from contextlib import redirect_stdout
from datetime import datetime

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import torch
import numpy as np
from omegaconf import OmegaConf

from marl_research.utils.misc import get_device, set_seed


# Optional wandb support
_wandb = None
_wandb_run = None


def init_wandb(project: str = "vabl_icml2026", name: str = None, config: dict = None, entity: str = None):
    """Initialize wandb logging.

    Args:
        project: Wandb project name
        name: Run name
        config: Config dict to log
        entity: Wandb entity (username or team). None uses personal account.
    """
    global _wandb, _wandb_run
    try:
        import wandb
        _wandb = wandb
        _wandb_run = wandb.init(
            project=project,
            entity=entity,  # None = personal account
            name=name or f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
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
    if _wandb is not None and _wandb_run is not None:
        _wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='VABL Quick Test')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (auto will use GPU if available)')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='vabl_icml2026', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Wandb entity (username/team); defaults to your wandb default')
    args = parser.parse_args()

    device = get_device(args.device)

    # Initialize wandb if requested
    if args.wandb:
        init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,  # None = personal account
            name=f"quick_test_{device}",
            config={'device': str(device), 'test_type': 'quick_test'}
        )

    print('='*65)
    print('VABL Quick Training Test - Overcooked (cramped_room)')
    print('='*65)
    print(f'Device: {device}')

    # Config
    config = OmegaConf.create({
        'algorithm': {
            'name': 'vabl',
            'embed_dim': 64,
            'hidden_dim': 128,
            'attention_dim': 64,
            'aux_hidden_dim': 64,
            'critic_hidden_dim': 128,
            'aux_lambda': 0.5,
            'clip_param': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.05,
            'gae_lambda': 0.95,
        },
        'environment': {
            'name': 'overcooked',
            'layout_name': 'cramped_room',
            'horizon': 100,
            'num_agents': 2,
        },
        'training': {
            'lr': 0.001,
            'gamma': 0.99,
            'grad_clip': 10.0,
            'batch_size': 8,
            'buffer_size': 50,
        },
        'experiment': {
            'seed': 42,
        }
    })

    set_seed(42)

    from marl_research.environments.overcooked_env import OvercookedEnv
    from marl_research.algorithms.vabl import VABL
    from marl_research.utils.replay_buffer import ReplayBuffer

    print('\nInitializing...')
    with redirect_stdout(io.StringIO()):
        env = OvercookedEnv(config)
    env_info = env.get_env_info()

    algo = VABL(config, env_info.n_agents, env_info.obs_shape, env_info.state_shape, env_info.n_actions, device)

    buffer = ReplayBuffer(
        buffer_size=config.training.buffer_size,
        episode_limit=env_info.episode_limit,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
    )

    print(f'  Environment: Overcooked (cramped_room)')
    print(f'  VABL params: {sum(p.numel() for p in algo.agent.parameters()):,}')

    def get_visibility_masks():
        return np.ones((env_info.n_agents, env_info.n_agents - 1), dtype=np.float32)

    # Training
    n_episodes = 20
    rewards_history = []
    aux_loss_history = []

    print(f'\nTraining for {n_episodes} episodes...')
    print('-'*65)

    for ep in range(n_episodes):
        with redirect_stdout(io.StringIO()):
            obs, state, info = env.reset()
        algo.init_hidden(batch_size=1)

        episode_reward = 0
        prev_actions = None

        for t in range(env_info.episode_limit):
            available_actions = env.get_available_actions()
            visibility_masks = get_visibility_masks()

            obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
            avail_tensor = torch.FloatTensor(np.array(available_actions)).unsqueeze(0).to(device)
            vis_tensor = torch.FloatTensor(visibility_masks).unsqueeze(0).to(device)

            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = torch.LongTensor(prev_actions).unsqueeze(0).to(device)

            with torch.no_grad():
                actions = algo.select_actions(
                    obs_tensor, avail_tensor, explore=True,
                    prev_actions=prev_actions_tensor,
                    visibility_masks=vis_tensor,
                )

            actions_np = actions.squeeze(0).cpu().numpy()
            next_obs, next_state, reward, done, info = env.step(actions_np.tolist())
            next_available_actions = env.get_available_actions()

            buffer.add_transition(
                obs=np.array(obs),
                state=np.array(state),
                actions=actions_np,
                reward=reward,
                next_obs=np.array(next_obs),
                next_state=np.array(next_state),
                done=done,
                available_actions=np.array(available_actions),
                next_available_actions=np.array(next_available_actions),
                visibility_masks=visibility_masks,
            )

            episode_reward += reward
            prev_actions = actions_np.copy()
            obs = next_obs
            state = next_state

            if done:
                break

        rewards_history.append(episode_reward)

        if buffer.can_sample(config.training.batch_size):
            batch = buffer.sample(config.training.batch_size)
            metrics = algo.train_step(batch)
            aux_loss_history.append(metrics['aux_loss'])

            # Log to wandb
            log_wandb({
                'episode': ep + 1,
                'reward': episode_reward,
                'aux_loss': metrics['aux_loss'],
                'aux_accuracy': metrics['aux_accuracy'],
            }, step=ep + 1)

            if (ep + 1) % 10 == 0:
                print(f'Episode {ep+1:3d} | Aux Loss: {np.mean(aux_loss_history[-10:]):.4f} | Aux Acc: {metrics["aux_accuracy"]:.1%}')

    print('-'*65)
    print(f'\nAux loss (first 10): {np.mean(aux_loss_history[:10]):.4f}')
    print(f'Aux loss (last 10):  {np.mean(aux_loss_history[-10:]):.4f}')
    print(f'Improvement: {np.mean(aux_loss_history[:10]) - np.mean(aux_loss_history[-10:]):.4f}')

    # Log final summary to wandb
    log_wandb({
        'final/aux_loss_first_10': np.mean(aux_loss_history[:10]),
        'final/aux_loss_last_10': np.mean(aux_loss_history[-10:]),
        'final/improvement': np.mean(aux_loss_history[:10]) - np.mean(aux_loss_history[-10:]),
    })

    finish_wandb()

    print('\n' + '='*65)
    print('Quick test: SUCCESS')
    print('='*65)


if __name__ == '__main__':
    main()
