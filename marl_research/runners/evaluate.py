"""Evaluation script for MARL experiments."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from marl_research.algorithms import ALGORITHM_REGISTRY
from marl_research.algorithms.qmix import QMIX
from marl_research.environments import make_env
from marl_research.environments.smac_env import SMACEnv, SMACV2Env
from marl_research.environments.overcooked_env import OvercookedEnv
from marl_research.utils import set_seed, get_device


class Evaluator:
    """Evaluator for trained MARL policies."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = get_device(config.hardware.device)

        self.env = make_env(config)
        self.env_info = self.env.get_env_info()

        self.algorithm = None

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a trained model checkpoint."""
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
        self.algorithm.load(checkpoint_path)

    def evaluate(
        self,
        num_episodes: int = 32,
        render: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the loaded policy."""
        if self.algorithm is None:
            raise RuntimeError("No checkpoint loaded. Call load_checkpoint first.")

        rewards = []
        wins = []
        episode_lengths = []

        iterator = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)

        for ep in iterator:
            obs, state, info = self.env.reset()
            self.algorithm.init_hidden(batch_size=1)

            episode_reward = 0
            episode_length = 0

            for t in range(self.env_info.episode_limit):
                available_actions = self.env.get_available_actions()

                obs_tensor = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)
                avail_tensor = torch.FloatTensor(np.array(available_actions)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    actions = self.algorithm.select_actions(
                        obs_tensor, avail_tensor, explore=False
                    )
                actions = actions.squeeze(0).cpu().numpy()

                if render:
                    self.env.render()

                next_obs, next_state, reward, done, info = self.env.step(actions.tolist())

                episode_reward += reward
                episode_length += 1
                obs = next_obs

                if done:
                    break

            rewards.append(episode_reward)
            wins.append(float(info.get("battle_won", False)))
            episode_lengths.append(episode_length)

        results = {
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "reward_min": np.min(rewards),
            "reward_max": np.max(rewards),
            "win_rate": np.mean(wins),
            "episode_length_mean": np.mean(episode_lengths),
        }

        if verbose:
            print("\n" + "=" * 50)
            print("Evaluation Results")
            print("=" * 50)
            print(f"Episodes: {num_episodes}")
            print(f"Reward: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
            print(f"Win Rate: {results['win_rate']:.2%}")
            print(f"Episode Length: {results['episode_length_mean']:.1f}")
            print("=" * 50)

        return results

    def cross_play_evaluate(
        self,
        checkpoint_path_1: str,
        checkpoint_path_2: str,
        num_episodes: int = 100,
    ) -> Dict[str, float]:
        """Evaluate two agents playing together (for Overcooked)."""
        algo_name = self.config.algorithm.name
        algo_cls = ALGORITHM_REGISTRY[algo_name]

        agent1 = algo_cls(
            config=self.config,
            n_agents=1,
            obs_shape=self.env_info.obs_shape,
            state_shape=self.env_info.state_shape,
            n_actions=self.env_info.n_actions,
            device=self.device,
        )
        agent1.load(checkpoint_path_1)

        agent2 = algo_cls(
            config=self.config,
            n_agents=1,
            obs_shape=self.env_info.obs_shape,
            state_shape=self.env_info.state_shape,
            n_actions=self.env_info.n_actions,
            device=self.device,
        )
        agent2.load(checkpoint_path_2)

        rewards = []

        for ep in tqdm(range(num_episodes), desc="Cross-play evaluation"):
            obs, state, info = self.env.reset()
            agent1.init_hidden(batch_size=1)
            agent2.init_hidden(batch_size=1)

            episode_reward = 0

            for t in range(self.env_info.episode_limit):
                available_actions = self.env.get_available_actions()

                obs1 = torch.FloatTensor(obs[0]).unsqueeze(0).unsqueeze(0).to(self.device)
                obs2 = torch.FloatTensor(obs[1]).unsqueeze(0).unsqueeze(0).to(self.device)

                avail1 = torch.FloatTensor(available_actions[0]).unsqueeze(0).unsqueeze(0).to(self.device)
                avail2 = torch.FloatTensor(available_actions[1]).unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action1 = agent1.select_actions(obs1, avail1, explore=False)
                    action2 = agent2.select_actions(obs2, avail2, explore=False)

                actions = [action1.item(), action2.item()]
                next_obs, next_state, reward, done, info = self.env.step(actions)

                episode_reward += reward
                obs = next_obs

                if done:
                    break

            rewards.append(episode_reward)

        return {
            "cross_play_reward_mean": np.mean(rewards),
            "cross_play_reward_std": np.std(rewards),
        }

    def close(self) -> None:
        """Clean up resources."""
        self.env.close()


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_episodes", type=int, default=32, help="Number of eval episodes")
    parser.add_argument("--render", action="store_true", help="Render episodes")

    args, _ = parser.parse_known_args()

    evaluator = Evaluator(config)
    evaluator.load_checkpoint(args.checkpoint)
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=args.render,
    )

    evaluator.close()


if __name__ == "__main__":
    main()
