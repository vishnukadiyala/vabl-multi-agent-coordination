"""Main training script for MARL experiments."""

import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from marl_research.algorithms import ALGORITHM_REGISTRY
from marl_research.algorithms.qmix import QMIX  # Register QMIX
from marl_research.algorithms.vabl import VABL  # Register VABL
from marl_research.environments import make_env
from marl_research.environments.smac_env import SMACEnv, SMACV2Env  # Register environments
from marl_research.environments.overcooked_env import OvercookedEnv
from marl_research.utils import set_seed, get_device, ReplayBuffer
from marl_research.utils.logger import setup_logger, get_logger
from marl_research.utils.misc import AverageMeter


class Trainer:
    """Main trainer class for MARL experiments."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = None

        set_seed(config.experiment.seed)
        self.device = get_device(config.hardware.device)

        self.env = make_env(config)
        self.env_info = self.env.get_env_info()

        self._build_algorithm()
        self._build_buffer()

        self.timestep = 0
        self.episode = 0
        self.best_eval_reward = float('-inf')
        self.best_checkpoint_path = None

    def _build_algorithm(self) -> None:
        """Initialize the algorithm."""
        algo_name = self.config.algorithm.name
        algo_cls = ALGORITHM_REGISTRY[algo_name]

        self.algorithm = algo_cls(
            config=self.config,
            n_agents=self.env_info.n_agents,
            obs_shape=self.env_info.obs_shape,
            state_shape=self.env_info.state_shape,
            n_actions=self.env_info.n_actions,
            device=self.device,
        )

    def _build_buffer(self) -> None:
        """Initialize the replay buffer."""
        self.buffer = ReplayBuffer(
            buffer_size=self.config.training.buffer_size,
            episode_limit=self.env_info.episode_limit,
            n_agents=self.env_info.n_agents,
            obs_shape=self.env_info.obs_shape,
            state_shape=self.env_info.state_shape,
            n_actions=self.env_info.n_actions,
        )

    def _compute_visibility_masks(self, obs: np.ndarray) -> np.ndarray:
        """Compute visibility masks from observations.

        For SMAC: Agents outside sight range have zero features in observations.
        For Overcooked: Full visibility (all ones).

        Args:
            obs: Agent observations [n_agents, obs_dim]

        Returns:
            visibility_masks: [n_agents, n_agents-1] binary masks
        """
        n_agents = self.env_info.n_agents
        obs_array = np.array(obs)

        # Default: full visibility
        visibility_masks = np.ones((n_agents, n_agents - 1), dtype=np.float32)

        # For SMAC environments, we can infer visibility from observations
        # Each agent's observation includes features for other agents
        # If all features for an agent are zero, that agent is not visible
        env_name = self.config.environment.name if hasattr(self.config.environment, 'name') else ''

        if 'smac' in env_name.lower():
            # SMAC observation structure includes ally features
            # Typically structured as: [move_feats, enemy_feats, ally_feats, own_feats]
            # Ally features section contains info about other agents
            # If ally features are all zeros, that ally is not visible

            # This is a simplified heuristic - actual visibility depends on sight range
            # For now, we use full visibility and let the environment provide masks if available
            pass

        return visibility_masks

    def collect_episode(self) -> Dict[str, float]:
        """Collect one episode of experience."""
        obs, state, info = self.env.reset()
        self.algorithm.init_hidden(batch_size=1)

        episode_reward = 0
        episode_sparse_reward = 0
        episode_length = 0

        # Track previous actions for VABL attention mechanism
        prev_actions = None

        for t in range(self.env_info.episode_limit):
            available_actions = self.env.get_available_actions()

            obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
            avail_tensor = torch.FloatTensor(np.array(available_actions)).unsqueeze(0).to(self.device)

            # Get visibility masks from environment or compute from observations
            if hasattr(self.env, 'get_visibility_masks'):
                visibility_masks = self.env.get_visibility_masks()
            else:
                visibility_masks = self._compute_visibility_masks(obs)

            # Prepare tensors for VABL
            prev_actions_tensor = None
            if prev_actions is not None:
                prev_actions_tensor = torch.LongTensor(prev_actions).unsqueeze(0).to(self.device)

            vis_tensor = torch.FloatTensor(visibility_masks).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Check if algorithm supports extended select_actions signature (VABL)
                if hasattr(self.algorithm, 'select_actions'):
                    import inspect
                    sig = inspect.signature(self.algorithm.select_actions)
                    if 'prev_actions' in sig.parameters:
                        actions = self.algorithm.select_actions(
                            obs_tensor, avail_tensor, explore=True,
                            prev_actions=prev_actions_tensor,
                            visibility_masks=vis_tensor,
                        )
                    else:
                        actions = self.algorithm.select_actions(
                            obs_tensor, avail_tensor, explore=True
                        )
                else:
                    actions = self.algorithm.select_actions(
                        obs_tensor, avail_tensor, explore=True
                    )

            actions = actions.squeeze(0).cpu().numpy()

            next_obs, next_state, reward, done, info = self.env.step(actions.tolist())
            next_available_actions = self.env.get_available_actions()

            self.buffer.add_transition(
                obs=np.array(obs),
                state=np.array(state),
                actions=actions,
                reward=reward,
                next_obs=np.array(next_obs),
                next_state=np.array(next_state),
                done=done,
                available_actions=np.array(available_actions),
                next_available_actions=np.array(next_available_actions),
                visibility_masks=visibility_masks,
            )

            # Track shaped reward (total)
            episode_reward += reward
            # Track sparse reward separately (for dual metrics)
            sparse_reward = info.get('sparse_reward', reward)
            episode_sparse_reward += sparse_reward
            episode_length += 1
            self.timestep += 1

            # Update previous actions for next iteration
            prev_actions = actions.copy()
            obs = next_obs
            state = next_state

            if done:
                break

        self.episode += 1

        return {
            "episode_reward": episode_reward,
            "episode_sparse_reward": episode_sparse_reward,
            "episode_length": episode_length,
            "battle_won": info.get("battle_won", False),
            "coordination_rate": info.get("coordination_rate", 0.0),
        }

    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if not self.buffer.can_sample(self.config.training.batch_size):
            return {}

        batch = self.buffer.sample(self.config.training.batch_size)
        metrics = self.algorithm.train_step(batch)

        return metrics

    def evaluate(self, num_episodes: int = 32) -> Dict[str, float]:
        """Evaluate the current policy."""
        rewards = []
        wins = []

        for _ in range(num_episodes):
            obs, state, info = self.env.reset()
            self.algorithm.init_hidden(batch_size=1)

            episode_reward = 0
            prev_actions = None

            for t in range(self.env_info.episode_limit):
                available_actions = self.env.get_available_actions()

                obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
                avail_tensor = torch.FloatTensor(np.array(available_actions)).unsqueeze(0).to(self.device)

                # Get visibility masks
                if hasattr(self.env, 'get_visibility_masks'):
                    visibility_masks = self.env.get_visibility_masks()
                else:
                    visibility_masks = self._compute_visibility_masks(obs)

                prev_actions_tensor = None
                if prev_actions is not None:
                    prev_actions_tensor = torch.LongTensor(prev_actions).unsqueeze(0).to(self.device)

                vis_tensor = torch.FloatTensor(visibility_masks).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Check if algorithm supports extended select_actions signature (VABL)
                    sig = inspect.signature(self.algorithm.select_actions)
                    if 'prev_actions' in sig.parameters:
                        actions = self.algorithm.select_actions(
                            obs_tensor, avail_tensor, explore=False,
                            prev_actions=prev_actions_tensor,
                            visibility_masks=vis_tensor,
                        )
                    else:
                        actions = self.algorithm.select_actions(
                            obs_tensor, avail_tensor, explore=False
                        )
                actions = actions.squeeze(0).cpu().numpy()

                next_obs, next_state, reward, done, info = self.env.step(actions.tolist())

                episode_reward += reward
                prev_actions = actions.copy()
                obs = next_obs

                if done:
                    break

            rewards.append(episode_reward)
            wins.append(float(info.get("battle_won", False)))

        return {
            "eval_reward_mean": np.mean(rewards),
            "eval_reward_std": np.std(rewards),
            "eval_win_rate": np.mean(wins),
        }

    def train(self) -> str:
        """Main training loop."""
        self.logger = setup_logger(self.config)

        reward_meter = AverageMeter(window_size=100)
        sparse_reward_meter = AverageMeter(window_size=100)
        win_meter = AverageMeter(window_size=100)
        coordination_meter = AverageMeter(window_size=100)

        pbar = tqdm(total=self.config.experiment.total_timesteps, desc="Training")

        while self.timestep < self.config.experiment.total_timesteps:
            episode_metrics = self.collect_episode()
            reward_meter.update(episode_metrics["episode_reward"])
            sparse_reward_meter.update(episode_metrics["episode_sparse_reward"])
            win_meter.update(float(episode_metrics["battle_won"]))
            coordination_meter.update(episode_metrics.get("coordination_rate", 0.0))

            train_metrics = self.train_step()

            if self.timestep % self.config.experiment.log_interval == 0:
                metrics = {
                    "train/shaped_reward": reward_meter.avg,
                    "train/sparse_reward": sparse_reward_meter.avg,
                    "train/win_rate": win_meter.avg,
                    "train/coordination_rate": coordination_meter.avg,
                    "train/episode": self.episode,
                    "train/timestep": self.timestep,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                }
                self.logger.log_metrics(metrics, step=self.timestep)

            if self.timestep % self.config.experiment.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.logger.log_metrics(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step=self.timestep,
                )
                pbar.set_postfix(
                    reward=f"{reward_meter.avg:.2f}",
                    win_rate=f"{win_meter.avg:.2%}",
                    eval_win=f"{eval_metrics['eval_win_rate']:.2%}",
                )

                # Best checkpoint selection (early-stopping style)
                if eval_metrics['eval_reward_mean'] > self.best_eval_reward:
                    self.best_eval_reward = eval_metrics['eval_reward_mean']
                    self.best_checkpoint_path = Path(self.logger.run_dir) / "best_checkpoint.pt"
                    self.algorithm.save(str(self.best_checkpoint_path))
                    get_logger().info(
                        f"New best checkpoint saved: reward={self.best_eval_reward:.2f}"
                    )

            if self.timestep % self.config.experiment.save_interval == 0:
                checkpoint_path = Path(self.logger.run_dir) / f"checkpoint_{self.timestep}.pt"
                self.algorithm.save(str(checkpoint_path))

            pbar.update(episode_metrics["episode_length"])

        pbar.close()

        final_path = Path(self.logger.run_dir) / "final_checkpoint.pt"
        self.algorithm.save(str(final_path))

        # Log summary comparing best vs final performance
        get_logger().info("=" * 50)
        get_logger().info("Training Summary")
        get_logger().info("=" * 50)
        get_logger().info(f"Final shaped reward (100-ep avg): {reward_meter.avg:.2f}")
        get_logger().info(f"Final sparse reward (100-ep avg): {sparse_reward_meter.avg:.2f}")
        get_logger().info(f"Best eval reward: {self.best_eval_reward:.2f}")
        if self.best_checkpoint_path:
            get_logger().info(f"Best checkpoint: {self.best_checkpoint_path}")
        get_logger().info(f"Final checkpoint: {final_path}")
        get_logger().info("=" * 50)

        self.logger.close()
        self.env.close()

        return str(final_path)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    """Main entry point."""
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)
    checkpoint_path = trainer.train()

    print(f"\nTraining completed! Checkpoint saved at: {checkpoint_path}")


if __name__ == "__main__":
    main()
