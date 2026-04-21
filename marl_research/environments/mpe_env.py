"""MPE (Multi-Agent Particle Environment) Wrapper using PettingZoo."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


@register_env("mpe")
class MPEEnv(BaseMAEnv):
    """Wrapper for PettingZoo MPE environments."""

    SCENARIOS = {
        'simple_spread': 'simple_spread_v3',
        'simple_tag': 'simple_tag_v3',
        'simple_adversary': 'simple_adversary_v3',
        'simple_crypto': 'simple_crypto_v3',
        'simple_push': 'simple_push_v3',
        'simple_reference': 'simple_reference_v3',
        'simple_speaker_listener': 'simple_speaker_listener_v4',
        'simple_world_comm': 'simple_world_comm_v3',
    }

    def __init__(self, config: DictConfig):
        self.config = config.environment

        scenario = self.config.get('scenario', 'simple_spread')
        self.max_cycles = self.config.get('max_cycles', 25)
        self.continuous_actions = self.config.get('continuous_actions', False)

        try:
            from pettingzoo.mpe import (
                simple_spread_v3, simple_tag_v3, simple_adversary_v3,
                simple_crypto_v3, simple_push_v3, simple_reference_v3,
                simple_speaker_listener_v4, simple_world_comm_v3
            )
        except ImportError:
            raise ImportError(
                "PettingZoo MPE not installed. Install with: pip install pettingzoo[mpe]"
            )

        # Map scenario names to environment constructors
        env_constructors = {
            'simple_spread': simple_spread_v3,
            'simple_tag': simple_tag_v3,
            'simple_adversary': simple_adversary_v3,
            'simple_crypto': simple_crypto_v3,
            'simple_push': simple_push_v3,
            'simple_reference': simple_reference_v3,
            'simple_speaker_listener': simple_speaker_listener_v4,
            'simple_world_comm': simple_world_comm_v3,
        }

        if scenario not in env_constructors:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(env_constructors.keys())}")

        self.env = env_constructors[scenario].parallel_env(
            max_cycles=self.max_cycles,
            continuous_actions=self.continuous_actions,
        )

        self.scenario = scenario
        self._env_info = None
        self._step_count = 0
        self._agents = None

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        obs_dict, infos = self.env.reset()
        self._step_count = 0
        self._agents = list(obs_dict.keys())

        obs = [obs_dict[agent].astype(np.float32) for agent in self._agents]
        state = self._get_state(obs)
        return obs, state, {}

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        # Convert action list to dict
        action_dict = {agent: actions[i] for i, agent in enumerate(self._agents)}

        obs_dict, rewards_dict, terminations, truncations, infos = self.env.step(action_dict)
        self._step_count += 1

        # Handle agents that might have been removed
        obs = []
        for agent in self._agents:
            if agent in obs_dict:
                obs.append(obs_dict[agent].astype(np.float32))
            else:
                # Agent removed, use zeros
                obs.append(np.zeros(self._env_info.obs_shape, dtype=np.float32))

        state = self._get_state(obs)

        # Sum rewards across all agents (cooperative)
        total_reward = sum(rewards_dict.values())

        # Check if done
        done = all(terminations.values()) or all(truncations.values()) or self._step_count >= self.max_cycles

        return obs, state, total_reward, done, infos

    def _get_state(self, obs: List[np.ndarray]) -> np.ndarray:
        """Global state is concatenation of all observations."""
        return np.concatenate(obs).astype(np.float32)

    def get_env_info(self) -> EnvInfo:
        if self._env_info is None:
            # Reset to get observation shapes
            obs_dict, _ = self.env.reset()
            self._agents = list(obs_dict.keys())

            # Get observation shape from first agent
            sample_obs = obs_dict[self._agents[0]]
            obs_shape = sample_obs.shape

            # Get action space
            sample_action_space = self.env.action_space(self._agents[0])
            if hasattr(sample_action_space, 'n'):
                n_actions = sample_action_space.n
            else:
                n_actions = sample_action_space.shape[0]  # Continuous

            # State shape is concat of all obs
            state_shape = (obs_shape[0] * len(self._agents),)

            self._env_info = EnvInfo(
                n_agents=len(self._agents),
                obs_shape=obs_shape,
                state_shape=state_shape,
                n_actions=n_actions,
                episode_limit=self.max_cycles,
            )
        return self._env_info

    def get_available_actions(self) -> List[np.ndarray]:
        """All actions are available in MPE."""
        n_actions = self._env_info.n_actions if self._env_info else 5
        return [np.ones(n_actions, dtype=np.float32) for _ in range(len(self._agents) if self._agents else 3)]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
