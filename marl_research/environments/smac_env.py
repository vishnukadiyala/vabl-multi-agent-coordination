"""SMAC Environment Wrapper."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


@register_env("smac")
class SMACEnv(BaseMAEnv):
    """Wrapper for StarCraft Multi-Agent Challenge environment."""

    def __init__(self, config: DictConfig):
        self.config = config.environment

        try:
            from smac.env import StarCraft2Env
        except ImportError:
            raise ImportError(
                "SMAC not installed. Install with: pip install pysc2 smac"
            )

        self.env = StarCraft2Env(
            map_name=self.config.map_name,
            difficulty=self.config.difficulty,
            obs_all_health=self.config.obs_all_health,
            obs_own_health=self.config.obs_own_health,
            obs_last_action=self.config.obs_last_action,
            state_last_action=self.config.state_last_action,
            reward_sparse=self.config.reward_sparse,
            reward_only_positive=self.config.reward_only_positive,
            reward_scale=self.config.reward_scale,
            reward_scale_rate=self.config.reward_scale_rate,
            debug=self.config.debug,
        )

        self._env_info = None

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        self.env.reset()
        obs = self.env.get_obs()
        state = self.env.get_state()
        return obs, state, {}

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        reward, done, info = self.env.step(actions)
        obs = self.env.get_obs()
        state = self.env.get_state()

        info["battle_won"] = info.get("battle_won", False)

        return obs, state, reward, done, info

    def get_env_info(self) -> EnvInfo:
        if self._env_info is None:
            env_info = self.env.get_env_info()
            self._env_info = EnvInfo(
                n_agents=env_info["n_agents"],
                obs_shape=(env_info["obs_shape"],),
                state_shape=(env_info["state_shape"],),
                n_actions=env_info["n_actions"],
                episode_limit=env_info["episode_limit"],
            )
        return self._env_info

    def get_available_actions(self) -> List[np.ndarray]:
        return self.env.get_avail_actions()

    def close(self) -> None:
        self.env.close()

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        pass


@register_env("smac_v2")
class SMACV2Env(BaseMAEnv):
    """Wrapper for SMAC V2 environment with distribution shift."""

    def __init__(self, config: DictConfig):
        self.config = config.environment

        try:
            from smacv2.env import StarCraft2Env
        except ImportError:
            raise ImportError(
                "SMACv2 not installed. Install with: pip install smacv2"
            )

        self.env = StarCraft2Env(
            map_name=self.config.map_name,
            difficulty=self.config.difficulty,
            capability_config=dict(self.config.capability_config),
            obs_all_health=self.config.obs_all_health,
            obs_own_health=self.config.obs_own_health,
            state_last_action=self.config.state_last_action,
            reward_sparse=self.config.reward_sparse,
            reward_only_positive=self.config.reward_only_positive,
            reward_scale=self.config.reward_scale,
            debug=self.config.debug,
        )

        self._env_info = None

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        self.env.reset()
        obs = self.env.get_obs()
        state = self.env.get_state()
        return obs, state, {}

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        reward, done, info = self.env.step(actions)
        obs = self.env.get_obs()
        state = self.env.get_state()
        return obs, state, reward, done, info

    def get_env_info(self) -> EnvInfo:
        if self._env_info is None:
            env_info = self.env.get_env_info()
            self._env_info = EnvInfo(
                n_agents=env_info["n_agents"],
                obs_shape=(env_info["obs_shape"],),
                state_shape=(env_info["state_shape"],),
                n_actions=env_info["n_actions"],
                episode_limit=env_info["episode_limit"],
            )
        return self._env_info

    def get_available_actions(self) -> List[np.ndarray]:
        return self.env.get_avail_actions()

    def close(self) -> None:
        self.env.close()
