"""Simple test environment for verifying MARL training pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


@register_env("simple")
class SimpleCoordinationEnv(BaseMAEnv):
    """Simple coordination environment for testing VABL.

    Agents must coordinate to take the same action to receive positive reward.
    Partial observability is simulated via visibility masks.
    """

    def __init__(self, config: DictConfig):
        env_config = config.environment

        self.n_agents = getattr(env_config, 'n_agents', 3)
        self.obs_dim = getattr(env_config, 'obs_dim', 16)
        self.n_actions = getattr(env_config, 'n_actions', 5)
        self.episode_limit = getattr(env_config, 'episode_limit', 50)
        self.visibility_prob = getattr(env_config, 'visibility_prob', 0.7)

        self._step_count = 0
        self._target_action = 0
        self._visibility_masks = None
        self._coordination_count = 0
        self._total_steps = 0

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        self._step_count = 0
        self._target_action = np.random.randint(0, self.n_actions)
        self._coordination_count = 0
        self._total_steps = 0
        self._update_visibility()

        obs = self._get_obs()
        state = self._get_state()
        return obs, state, {}

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self._step_count += 1
        self._total_steps += 1

        # Reward based on coordination
        actions = list(actions)

        # Track coordination (all agents take same action)
        joint_action_agreement = int(len(set(actions)) == 1)
        self._coordination_count += joint_action_agreement

        # Bonus for all agents taking the same action
        if len(set(actions)) == 1:
            reward = 1.0
            # Extra bonus if they match the target
            if actions[0] == self._target_action:
                reward = 2.0
        else:
            reward = -0.1

        # Change target occasionally
        if np.random.random() < 0.1:
            self._target_action = np.random.randint(0, self.n_actions)

        self._update_visibility()

        done = self._step_count >= self.episode_limit

        obs = self._get_obs()
        state = self._get_state()

        info = {
            "battle_won": reward > 0,
            "sparse_reward": reward,  # In simple env, shaped = sparse
            "coordination_rate": self._coordination_count / max(1, self._total_steps),
            "joint_action_agreement": joint_action_agreement,
        }

        return obs, state, reward, done, info

    def _update_visibility(self):
        """Update visibility masks (partial observability)."""
        # Each agent can see each other agent with visibility_prob
        self._visibility_masks = np.zeros((self.n_agents, self.n_agents - 1), dtype=np.float32)
        for i in range(self.n_agents):
            for j_idx, j in enumerate([k for k in range(self.n_agents) if k != i]):
                if np.random.random() < self.visibility_prob:
                    self._visibility_masks[i, j_idx] = 1.0

    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for each agent."""
        obs_list = []
        for i in range(self.n_agents):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            # Encode target action as one-hot in first few dims
            obs[self._target_action] = 1.0
            # Add agent ID
            obs[self.n_actions + i] = 1.0
            # Add step count (normalized)
            obs[-1] = self._step_count / self.episode_limit
            # Add some noise
            obs += np.random.randn(self.obs_dim) * 0.01
            obs_list.append(obs)
        return obs_list

    def _get_state(self) -> np.ndarray:
        """Get global state."""
        state = np.zeros(self.obs_dim * self.n_agents, dtype=np.float32)
        for i, obs in enumerate(self._get_obs()):
            state[i * self.obs_dim:(i + 1) * self.obs_dim] = obs
        return state

    def get_env_info(self) -> EnvInfo:
        """Get environment information."""
        return EnvInfo(
            n_agents=self.n_agents,
            obs_shape=(self.obs_dim,),
            state_shape=(self.obs_dim * self.n_agents,),
            n_actions=self.n_actions,
            episode_limit=self.episode_limit,
        )

    def get_available_actions(self) -> List[np.ndarray]:
        """All actions are available."""
        return [np.ones(self.n_actions, dtype=np.float32) for _ in range(self.n_agents)]

    def get_visibility_masks(self) -> np.ndarray:
        """Get current visibility masks."""
        return self._visibility_masks

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        print(f"Step: {self._step_count}, Target: {self._target_action}")
        return None

    def close(self) -> None:
        pass
