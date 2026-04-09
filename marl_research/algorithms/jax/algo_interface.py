"""Common interface for vectorized JAX MARL algorithms.

Each algorithm implements this interface so the training loop is algorithm-agnostic.
The training loop handles rollout collection, GAE, PPO epochs.
The algorithm handles forward passes, loss functions, and recurrent state.
"""

from typing import Protocol, Tuple, Any, NamedTuple
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


class RolloutBatch(NamedTuple):
    """Batch of rollout data for PPO update.
    All shapes: leading dim = N*H (batch_size * horizon) flat batch.
    """
    obs: jnp.ndarray            # [N*H, n_agents, obs_dim]
    actions: jnp.ndarray        # [N*H, n_agents]
    next_actions: jnp.ndarray   # [N*H, n_agents] — for VABL aux loss (predict next teammate actions)
    log_probs: jnp.ndarray      # [N*H, n_agents]
    carry: jnp.ndarray          # [N*H, n_agents, carry_dim] or [N*H, carry_dim]
    states: jnp.ndarray         # [N*H, state_dim]
    advantages: jnp.ndarray     # [N*H]
    returns: jnp.ndarray        # [N*H]


class JaxMARLAlgo(Protocol):
    """Interface for vectorized JAX MARL algorithms."""

    n_agents: int
    n_actions: int
    obs_dim: int
    state_dim: int

    def init(self, rng: jax.Array) -> Tuple[TrainState, TrainState]:
        """Initialize parameters. Returns (agent_state, critic_state)."""
        ...

    def init_carry(self, n_envs: int) -> jnp.ndarray:
        """Initial recurrent state for N envs. Shape algorithm-specific."""
        ...

    def step(
        self,
        agent_params: Any,
        obs: jnp.ndarray,            # [N, n_agents, obs_dim]
        carry: jnp.ndarray,          # [N, ...]
        prev_actions: jnp.ndarray,   # [N, n_agents]
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Vectorized forward pass for N envs.

        Returns:
            actions: [N, n_agents] sampled actions
            new_carry: [N, ...] updated recurrent state
            log_probs: [N, n_agents] log probability of sampled actions
        """
        ...

    def get_value(self, critic_params: Any, state: jnp.ndarray) -> jnp.ndarray:
        """Vectorized critic. state: [N, state_dim] -> [N]"""
        ...

    def actor_loss(
        self,
        params: Any,
        batch: RolloutBatch,
        clip_param: float,
        entropy_coef: float,
    ) -> jnp.ndarray:
        """PPO actor loss on a flat rollout batch."""
        ...

    def critic_loss(self, params: Any, batch: RolloutBatch) -> jnp.ndarray:
        """Value loss on a flat rollout batch."""
        ...
