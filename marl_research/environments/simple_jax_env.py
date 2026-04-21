"""Simple Coordination Environment in JAX.

N-agent coordination task. Each step, agents must take the same action,
with bonus reward when that action matches a target. Stochastic visibility:
each agent sees each teammate with probability `visibility_prob`.

Vectorizable via jax.vmap, no Python state.

Reward structure:
  +2.0 if all agents match target action
  +1.0 if all agents take same action (but not target)
  -0.1 otherwise

Target action changes randomly with 10% probability each step.
"""

from typing import NamedTuple, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import chex


@chex.dataclass
class SimpleEnvState:
    target_action: jnp.ndarray  # scalar int
    step_count: jnp.ndarray     # scalar int
    visibility_masks: jnp.ndarray  # [n_agents, n_teammates]


class SimpleJaxEnv:
    """N-agent simple coordination environment in JAX.

    Compatible with the train_unified loop's expected env interface
    (mimics JaxMARL Overcooked's reset/step signature with dicts).
    """

    def __init__(
        self,
        n_agents: int = 4,
        n_actions: int = 5,
        obs_dim: int = 16,
        episode_limit: int = 50,
        visibility_prob: float = 0.7,
        target_change_prob: float = 0.1,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.episode_limit = episode_limit
        self.visibility_prob = visibility_prob
        self.target_change_prob = target_change_prob
        self.agents = [f"agent_{i}" for i in range(n_agents)]

    def reset(self, key: chex.PRNGKey) -> Tuple[dict, SimpleEnvState]:
        """Reset env. Returns (obs_dict, state)."""
        key_target, key_vis = jax.random.split(key)
        target = jax.random.randint(key_target, (), 0, self.n_actions)

        # Sample visibility masks: [n_agents, n_teammates]
        n_teammates = self.n_agents - 1
        vis_masks = (jax.random.uniform(key_vis, (self.n_agents, n_teammates))
                     < self.visibility_prob).astype(jnp.float32)

        state = SimpleEnvState(
            target_action=target,
            step_count=jnp.int32(0),
            visibility_masks=vis_masks,
        )
        obs = self._get_obs(state, key)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: SimpleEnvState,
        actions: dict,
    ) -> Tuple[dict, SimpleEnvState, dict, dict, dict]:
        """Step env. actions is a dict {agent_name: int_array}."""
        # Stack actions in agent order
        action_array = jnp.stack([actions[a] for a in self.agents])  # [n_agents]

        # Reward computation
        all_same = jnp.all(action_array == action_array[0])
        match_target = all_same & (action_array[0] == state.target_action)
        reward = jnp.where(
            match_target, 2.0,
            jnp.where(all_same, 1.0, -0.1)
        )

        # Maybe change target
        key_target, key_vis = jax.random.split(key)
        change = jax.random.uniform(key_target) < self.target_change_prob
        new_target = jnp.where(
            change,
            jax.random.randint(key_target, (), 0, self.n_actions),
            state.target_action,
        )

        # Resample visibility
        n_teammates = self.n_agents - 1
        new_vis = (jax.random.uniform(key_vis, (self.n_agents, n_teammates))
                   < self.visibility_prob).astype(jnp.float32)

        new_step = state.step_count + 1
        done = new_step >= self.episode_limit

        new_state = SimpleEnvState(
            target_action=new_target,
            step_count=new_step,
            visibility_masks=new_vis,
        )
        obs = self._get_obs(new_state, key)

        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        info = {"shaped_reward": {a: jnp.float32(0.0) for a in self.agents}}

        return obs, new_state, rewards, dones, info

    def _get_obs(self, state: SimpleEnvState, key: chex.PRNGKey) -> dict:
        """Build observations for each agent.

        Each agent gets:
        - One-hot of target action (first n_actions dims)
        - One-hot of own agent ID (next n_agents dims)
        - Normalized step count (1 dim)
        - Padding zeros to obs_dim
        + small noise
        """
        obs_dict = {}
        for i in range(self.n_agents):
            target_oh = jax.nn.one_hot(state.target_action, self.n_actions)
            agent_oh = jax.nn.one_hot(i, self.n_agents)
            step_norm = state.step_count.astype(jnp.float32) / self.episode_limit

            features = jnp.concatenate([
                target_oh,
                agent_oh,
                jnp.array([step_norm]),
            ])

            # Pad to obs_dim
            padding_len = self.obs_dim - features.shape[0]
            if padding_len > 0:
                features = jnp.concatenate([features, jnp.zeros(padding_len)])
            elif padding_len < 0:
                features = features[:self.obs_dim]

            # Small noise
            noise_key = jax.random.fold_in(key, i)
            noise = jax.random.normal(noise_key, features.shape) * 0.01
            features = features + noise

            obs_dict[self.agents[i]] = features.reshape(1, 1, self.obs_dim)  # [H, W, C] like Overcooked

        return obs_dict

    def action_space(self, agent: str):
        """Return action space (compatible with JaxMARL's interface)."""
        class _Discrete:
            def __init__(self, n):
                self.n = n
        return _Discrete(self.n_actions)
