"""JaxMARL Overcooked Environment Wrapper.

Wraps JaxMARL's vectorized Overcooked into our BaseMAEnv interface.
~1000x faster than overcooked_ai_py — no MotionPlanner, pure JAX grid sim.

Observations are spatial (H x W x 26 channels) flattened to 1D vectors
for compatibility with MLP-based agents.

Usage:
    # Drop-in replacement for overcooked_env.py
    cfg.environment.name = "overcooked_jax"
    cfg.environment.layout_name = "cramped_room"
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


# Layout name mapping: our names → JaxMARL names
LAYOUT_MAP = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymm_advantages",
    "coordination_ring": "coord_ring",
    "forced_coordination": "forced_coord",
    "counter_circuit": "counter_circuit",
}


def _import_jaxmarl():
    """Import JaxMARL Overcooked, bypassing unneeded physics deps."""
    import sys
    import types

    # Block mabrax (needs MuJoCo) — we only need Overcooked
    if "jaxmarl.environments.mabrax" not in sys.modules:
        fake = types.ModuleType("jaxmarl.environments.mabrax")
        fake.Ant = fake.Humanoid = fake.Hopper = fake.Walker2d = fake.HalfCheetah = None
        sys.modules["jaxmarl.environments.mabrax"] = fake

    import jax
    from jaxmarl.environments.overcooked.overcooked import Overcooked, layouts

    return jax, Overcooked, layouts


@register_env("overcooked_jax")
class OvercookedJaxEnv(BaseMAEnv):
    """JaxMARL Overcooked wrapper for MARL research.

    Drop-in replacement for OvercookedEnv with ~1000x speedup.
    Observations are flattened spatial grids (H * W * 26 features).
    """

    def __init__(self, config: DictConfig):
        self.config = config.environment

        jax, Overcooked, available_layouts = _import_jaxmarl()
        self._jax = jax
        self._jnp = jax.numpy

        layout_name = getattr(self.config, "layout_name", "cramped_room")
        jax_layout_name = LAYOUT_MAP.get(layout_name, layout_name)

        if jax_layout_name not in available_layouts:
            raise ValueError(
                f"Layout '{layout_name}' not found. Available: {list(LAYOUT_MAP.keys())}"
            )

        self.env = Overcooked(layout=available_layouts[jax_layout_name])
        self.n_agents = len(self.env.agents)
        self._agent_names = sorted(self.env.agents)
        self.n_actions = self.env.action_space(self._agent_names[0]).n
        self.horizon = getattr(self.config, "horizon", 400)

        # JIT-compile reset and step for speed
        self._jit_reset = jax.jit(self.env.reset)
        self._jit_step = jax.jit(self.env.step)

        # Get observation shape from a test reset
        key = jax.random.PRNGKey(0)
        test_obs, _ = self._jit_reset(key)
        sample_obs = np.array(test_obs[self._agent_names[0]])
        self._spatial_shape = sample_obs.shape  # (H, W, 26)
        self.obs_dim = int(np.prod(self._spatial_shape))

        # State = concatenated observations of all agents
        self.state_dim = self.obs_dim * self.n_agents

        # JAX random key management
        self._key = jax.random.PRNGKey(42)
        self._state = None
        self._step_count = 0
        self._done = False
        self._env_info = None

        # Shaped reward tracking
        self.use_shaped_rewards = getattr(self.config, "use_shaped_rewards", True)
        self.shaped_reward_scale = getattr(self.config, "shaped_reward_scale", 1.0)

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        self._key, reset_key = self._jax.random.split(self._key)
        obs_dict, self._state = self._jit_reset(reset_key)
        self._step_count = 0
        self._done = False

        obs = self._extract_obs(obs_dict)
        state = self._get_state(obs)
        return obs, state, {}

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        self._step_count += 1

        if self._done:
            obs = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)]
            state = np.zeros(self.state_dim, dtype=np.float32)
            return obs, state, 0.0, True, {"sparse_reward": 0.0}

        # Convert actions to JaxMARL format
        actions_dict = {
            self._agent_names[i]: self._jnp.int32(actions[i])
            for i in range(self.n_agents)
        }

        self._key, step_key = self._jax.random.split(self._key)
        obs_dict, self._state, reward_dict, done_dict, info = self._jit_step(
            step_key, self._state, actions_dict
        )

        # Extract reward — add shaped rewards from JaxMARL
        sparse_reward = float(reward_dict[self._agent_names[0]])
        shaped_reward = 0.0
        if self.use_shaped_rewards and "shaped_reward" in info:
            # Sum shaped rewards across agents
            for agent_name in self._agent_names:
                if agent_name in info["shaped_reward"]:
                    shaped_reward += float(info["shaped_reward"][agent_name])
            shaped_reward *= self.shaped_reward_scale
        reward = sparse_reward + shaped_reward

        # Check done
        self._done = bool(done_dict["__all__"]) or self._step_count >= self.horizon

        obs = self._extract_obs(obs_dict)
        state = self._get_state(obs)

        info_out = {
            "sparse_reward": sparse_reward,
            "shaped_reward": shaped_reward,
        }

        return obs, state, reward, self._done, info_out

    def _extract_obs(self, obs_dict) -> List[np.ndarray]:
        """Convert JaxMARL obs dict to list of flattened numpy arrays."""
        obs_list = []
        for agent_name in self._agent_names:
            obs_jax = obs_dict[agent_name]
            obs_np = np.array(obs_jax, dtype=np.float32).flatten()
            obs_list.append(obs_np)
        return obs_list

    def _get_state(self, obs_list: List[np.ndarray]) -> np.ndarray:
        """Global state = concatenated observations."""
        return np.concatenate(obs_list, axis=0)

    def get_env_info(self) -> EnvInfo:
        if self._env_info is None:
            self._env_info = EnvInfo(
                n_agents=self.n_agents,
                obs_shape=(self.obs_dim,),
                state_shape=(self.state_dim,),
                n_actions=self.n_actions,
                episode_limit=self.horizon,
            )
        return self._env_info

    def get_available_actions(self) -> List[np.ndarray]:
        """All actions always available in JaxMARL Overcooked."""
        return [
            np.ones(self.n_actions, dtype=np.float32)
            for _ in range(self.n_agents)
        ]

    def get_visibility_masks(self) -> np.ndarray:
        """Full visibility between agents."""
        return np.ones((self.n_agents, self.n_agents - 1), dtype=np.float32)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        pass

    def seed(self, seed: int) -> None:
        self._key = self._jax.random.PRNGKey(seed)
