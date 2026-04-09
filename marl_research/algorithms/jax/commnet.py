"""CommNet (Communication Neural Network) in JAX/Flax.

End-to-end JAX implementation for vectorized training.
Based on "Learning Multiagent Communication with Backpropagation" (Sukhbaatar et al., 2016).

Architecture:
  - Observation encoder: Dense -> ReLU
  - Communication rounds (default 2): for each agent, average all others' hidden
    states, concatenate [own || avg_msg], Dense -> ReLU
  - Policy head: 2-layer MLP -> action logits
  - Centralized critic: 3-layer MLP on global state -> value
  - PPO training with GAE
"""

from typing import NamedTuple, Tuple, Dict, Any
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import chex


class CommNetConfig(NamedTuple):
    """CommNet hyperparameters."""
    # Architecture
    hidden_dim: int = 64
    comm_rounds: int = 2
    critic_hidden_dim: int = 64

    # PPO Training
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_clip: float = 0.2
    ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 10.0

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 30
    state_dim: int = 60


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class CommNetModule(nn.Module):
    """CommNet communication module.

    Forward pass for all agents simultaneously:
      1. h = encoder(obs)                         [n_agents, hidden_dim]
      2. For each communication round k:
           msg_i = mean(h_j for j != i)           [n_agents, hidden_dim]
           h_i   = relu(W_k [h_i || msg_i])       [n_agents, hidden_dim]
      3. logits = policy_head(h)                   [n_agents, n_actions]
    """
    n_agents: int
    n_actions: int
    hidden_dim: int
    comm_rounds: int

    @nn.compact
    def __call__(self, obs, available_actions=None):
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions] binary mask or None
        Returns:
            logits: [batch, n_agents, n_actions]
        """
        batch_size = obs.shape[0]

        # 1. Encode observations
        h = nn.Dense(self.hidden_dim)(obs)
        h = nn.relu(h)  # [batch, n_agents, hidden_dim]

        # 2. Communication rounds
        for k in range(self.comm_rounds):
            # Compute mean message from all other agents for each agent
            # Sum over all agents, subtract self, divide by (n_agents - 1)
            h_sum = jnp.sum(h, axis=1, keepdims=True)  # [batch, 1, hidden_dim]
            # For agent i: msg_i = (h_sum - h_i) / (n_agents - 1)
            # Broadcast: h_sum is [batch, 1, hidden], h is [batch, n_agents, hidden]
            n_others = jnp.maximum(self.n_agents - 1, 1)  # avoid div by zero for single agent
            messages = (h_sum - h) / n_others  # [batch, n_agents, hidden_dim]

            # Concatenate own hidden state with received message
            combined = jnp.concatenate([h, messages], axis=-1)  # [batch, n_agents, 2*hidden_dim]

            # Communication layer (unique weights per round)
            h = nn.Dense(self.hidden_dim, name=f"comm_{k}")(combined)
            h = nn.relu(h)  # [batch, n_agents, hidden_dim]

        # 3. Policy head: 2-layer MLP -> logits
        x = nn.Dense(self.hidden_dim, name="policy_fc1")(h)
        x = nn.relu(x)
        logits = nn.Dense(self.n_actions, name="policy_fc2")(x)  # [batch, n_agents, n_actions]

        # Mask unavailable actions
        if available_actions is not None:
            logits = jnp.where(available_actions > 0, logits, -1e10)

        return logits


class CommNetAgentWrapper(nn.Module):
    """Wrapper around CommNet to match the standardized per-agent interface.

    Signature: (obs_i, belief_i, teammate_actions_oh, vis_mask_i) -> (logits, new_belief, aux_logits)

    In per-agent mode, encodes the single agent's observation through an MLP
    and policy head (no inter-agent communication, since that requires all
    agents' obs simultaneously). belief_i is passed through unchanged.
    Returns zeros for aux_logits.
    """
    config: CommNetConfig

    @nn.compact
    def __call__(self, obs_i, belief_i, teammate_actions_oh, vis_mask_i):
        cfg = self.config

        # Encode observation (same architecture as CommNet encoder)
        x = nn.Dense(cfg.hidden_dim, name="obs_enc")(obs_i)
        x = nn.relu(x)

        # Policy head (2-layer MLP)
        x = nn.Dense(cfg.hidden_dim, name="policy_fc1")(x)
        x = nn.relu(x)
        logits = nn.Dense(cfg.n_actions, name="policy_fc2")(x)

        # No recurrent state; pass belief through unchanged
        new_belief = belief_i
        n_teammates = cfg.n_agents - 1
        aux_logits = jnp.zeros((n_teammates, cfg.n_actions))
        return logits, new_belief, aux_logits


class Critic(nn.Module):
    """Centralized critic: global state -> scalar value.

    3-layer MLP: Dense(hidden) -> ReLU -> Dense(hidden) -> ReLU -> Dense(1).
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        """
        Args:
            state: [..., state_dim]
        Returns:
            value: [...]
        """
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class CommNet:
    """CommNet algorithm with PPO training in JAX.

    Provides:
        - init(): create actor (CommNet) and critic params/states
        - get_action(): sample actions from policy (jit-compiled)
        - train_step(): PPO update on a batch (jit-compiled)
    """

    def __init__(self, config: CommNetConfig):
        self.config = config
        self.actor = CommNetModule(
            n_agents=config.n_agents,
            n_actions=config.n_actions,
            hidden_dim=config.hidden_dim,
            comm_rounds=config.comm_rounds,
        )
        self.agent = CommNetAgentWrapper(config)
        self.critic = Critic(hidden_dim=config.critic_hidden_dim)

    def init(self, rng):
        """Initialize parameters and optimizer states.

        Returns:
            agent_state: TrainState for the CommNet agent wrapper
            critic_state: TrainState for the centralized critic
        """
        cfg = self.config
        rng_agent, rng_critic = jax.random.split(rng)

        # Dummy inputs for wrapper signature
        n_teammates = cfg.n_agents - 1
        dummy_obs = jnp.zeros(cfg.obs_dim)
        dummy_belief = jnp.zeros(cfg.hidden_dim)
        dummy_teammate_actions_oh = jnp.zeros((n_teammates, cfg.n_actions))
        dummy_vis_mask = jnp.zeros(n_teammates)
        dummy_state = jnp.zeros(cfg.state_dim)

        agent_params = self.agent.init(
            rng_agent, dummy_obs, dummy_belief, dummy_teammate_actions_oh, dummy_vis_mask
        )
        critic_params = self.critic.init(rng_critic, dummy_state)

        agent_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.actor_lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.critic_lr, eps=1e-5),
        )

        agent_state = TrainState.create(
            apply_fn=self.agent.apply,
            params=agent_params,
            tx=agent_tx,
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=critic_tx,
        )

        return agent_state, critic_state

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        actor_state: TrainState,
        obs: jnp.ndarray,
        rng: chex.PRNGKey,
        available_actions: jnp.ndarray = None,
    ):
        """Sample actions for all agents with communication.

        Args:
            actor_state: TrainState with actor params
            obs: [n_agents, obs_dim]
            rng: PRNGKey
            available_actions: [n_agents, n_actions] binary mask or None

        Returns:
            actions: [n_agents] int32
            log_probs: [n_agents] log probability of chosen actions
        """
        cfg = self.config

        # Add batch dimension
        obs_batch = obs[None, ...]  # [1, n_agents, obs_dim]
        avail_batch = None
        if available_actions is not None:
            avail_batch = available_actions[None, ...]

        logits = actor_state.apply_fn(actor_state.params, obs_batch, avail_batch)
        logits = logits[0]  # remove batch dim -> [n_agents, n_actions]

        # Sample actions
        log_probs_all = jax.nn.log_softmax(logits, axis=-1)  # [n_agents, n_actions]

        # Split rng per agent
        rngs = jax.random.split(rng, cfg.n_agents)
        actions = jax.vmap(
            lambda key, log_p: jax.random.categorical(key, log_p)
        )(rngs, logits)  # [n_agents]

        # Gather log probs of chosen actions
        log_probs = jax.vmap(lambda lp, a: lp[a])(log_probs_all, actions)  # [n_agents]

        return actions, log_probs

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, critic_state: TrainState, state: jnp.ndarray):
        """Compute value of a global state.

        Args:
            critic_state: TrainState with critic params
            state: [state_dim] or [batch, state_dim]

        Returns:
            value: scalar or [batch]
        """
        return critic_state.apply_fn(critic_state.params, state)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        batch: Dict[str, jnp.ndarray],
    ):
        """Perform PPO training step with multiple epochs.

        Args:
            actor_state: TrainState for CommNet actor
            critic_state: TrainState for centralized critic
            batch: dict with keys:
                obs: [batch, seq_len, n_agents, obs_dim]
                state: [batch, seq_len, state_dim]
                actions: [batch, seq_len, n_agents] int32
                rewards: [batch, seq_len]
                dones: [batch, seq_len]
                mask: [batch, seq_len] (1 = valid timestep)
                available_actions: [batch, seq_len, n_agents, n_actions] (optional)

        Returns:
            actor_state: updated actor TrainState
            critic_state: updated critic TrainState
            metrics: dict of scalar metrics
        """
        cfg = self.config

        obs = batch["obs"]
        state = batch["state"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        mask = batch.get("mask", jnp.ones_like(rewards))
        available_actions = batch.get("available_actions", None)

        batch_size, seq_len = obs.shape[0], obs.shape[1]

        # --- Compute old log probs and values (no grad) ---
        # Flatten batch and seq_len for forward pass
        obs_flat = obs.reshape(batch_size * seq_len, cfg.n_agents, -1)
        state_flat = state.reshape(batch_size * seq_len, -1)

        avail_flat = None
        if available_actions is not None:
            avail_flat = available_actions.reshape(batch_size * seq_len, cfg.n_agents, -1)

        old_logits = actor_state.apply_fn(actor_state.params, obs_flat, avail_flat)
        # [batch*seq, n_agents, n_actions]
        old_log_probs_all = jax.nn.log_softmax(old_logits, axis=-1)

        actions_flat = actions.reshape(batch_size * seq_len, cfg.n_agents)
        # Gather log probs for chosen actions, then sum across agents
        old_log_probs = jnp.sum(
            jnp.take_along_axis(
                old_log_probs_all,
                actions_flat[..., None],
                axis=-1,
            ).squeeze(-1),
            axis=-1,
        ).reshape(batch_size, seq_len)

        old_values = critic_state.apply_fn(
            critic_state.params, state_flat
        ).reshape(batch_size, seq_len)

        # --- GAE computation ---
        returns, advantages = self._compute_gae(rewards, old_values, dones, mask, cfg)
        # Normalize advantages
        adv_mean = (advantages * mask).sum() / (mask.sum() + 1e-8)
        adv_std = jnp.sqrt(
            ((advantages - adv_mean) ** 2 * mask).sum() / (mask.sum() + 1e-8) + 1e-8
        )
        advantages = jnp.clip((advantages - adv_mean) / adv_std, -10.0, 10.0)

        # --- PPO epochs ---
        def _ppo_epoch(carry, _):
            actor_st, critic_st = carry

            # Actor loss
            def _actor_loss_fn(actor_params):
                logits = self.actor.apply(actor_params, obs_flat, avail_flat)
                log_probs_all = jax.nn.log_softmax(logits, axis=-1)
                probs_all = jax.nn.softmax(logits, axis=-1)

                # Log prob of chosen actions summed over agents
                log_probs = jnp.sum(
                    jnp.take_along_axis(
                        log_probs_all,
                        actions_flat[..., None],
                        axis=-1,
                    ).squeeze(-1),
                    axis=-1,
                ).reshape(batch_size, seq_len)

                # Entropy: -sum(p * log_p) averaged over agents
                entropy = -jnp.sum(probs_all * log_probs_all, axis=-1)  # [batch*seq, n_agents]
                entropy = entropy.mean(axis=-1).reshape(batch_size, seq_len)  # [batch, seq]

                # PPO clipped objective
                ratio = jnp.exp(log_probs - jax.lax.stop_gradient(old_log_probs))
                ratio = jnp.clip(ratio, 0.0, 10.0)
                surr1 = ratio * advantages
                surr2 = jnp.clip(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages

                policy_loss = -jnp.minimum(surr1, surr2) * mask
                policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

                entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

                total_loss = policy_loss + cfg.entropy_coef * entropy_loss

                return total_loss, {
                    "policy_loss": policy_loss,
                    "entropy": (entropy * mask).sum() / (mask.sum() + 1e-8),
                }

            (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
                _actor_loss_fn, has_aux=True
            )(actor_st.params)
            actor_st = actor_st.apply_gradients(grads=actor_grads)

            # Critic loss
            def _critic_loss_fn(critic_params):
                values = self.critic.apply(critic_params, state_flat).reshape(batch_size, seq_len)
                value_loss = ((values - returns) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                return cfg.value_loss_coef * value_loss, {"value_loss": value_loss}

            (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
                _critic_loss_fn, has_aux=True
            )(critic_st.params)
            critic_st = critic_st.apply_gradients(grads=critic_grads)

            metrics = {**actor_metrics, **critic_metrics}
            return (actor_st, critic_st), metrics

        (actor_state, critic_state), all_metrics = jax.lax.scan(
            _ppo_epoch,
            (actor_state, critic_state),
            None,
            length=cfg.ppo_epochs,
        )

        # Average metrics across PPO epochs
        avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)

        return actor_state, critic_state, avg_metrics

    @staticmethod
    def _compute_gae(
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        mask: jnp.ndarray,
        cfg: CommNetConfig,
    ):
        """Compute GAE returns and advantages.

        Args:
            rewards: [batch, seq_len]
            values: [batch, seq_len]
            dones: [batch, seq_len]
            mask: [batch, seq_len]
            cfg: CommNetConfig

        Returns:
            returns: [batch, seq_len]
            advantages: [batch, seq_len]
        """

        def _gae_step(carry, t_data):
            """Reverse-scan step for GAE computation."""
            last_gae, next_value = carry
            reward_t, value_t, done_t, mask_t = t_data

            delta = reward_t + cfg.gamma * next_value * (1.0 - done_t) - value_t
            last_gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done_t) * last_gae
            last_gae = last_gae * mask_t  # zero out padded steps
            return_t = last_gae + value_t

            return (last_gae, value_t), (return_t, last_gae)

        batch_size, seq_len = rewards.shape

        # Reverse the sequences for backward scan
        rewards_rev = jnp.flip(rewards, axis=1)
        values_rev = jnp.flip(values, axis=1)
        dones_rev = jnp.flip(dones, axis=1)
        mask_rev = jnp.flip(mask, axis=1)

        # Transpose for scan: [seq_len, batch]
        scan_inputs = (
            rewards_rev.T,  # [seq_len, batch]
            values_rev.T,
            dones_rev.T,
            mask_rev.T,
        )

        init_carry = (
            jnp.zeros(batch_size),  # last_gae
            jnp.zeros(batch_size),  # next_value (bootstrap = 0)
        )

        _, (returns_rev, advantages_rev) = jax.lax.scan(
            _gae_step, init_carry, scan_inputs
        )
        # returns_rev, advantages_rev: [seq_len, batch]

        # Reverse back and transpose: [batch, seq_len]
        returns = jnp.flip(returns_rev.T, axis=1)
        advantages = jnp.flip(advantages_rev.T, axis=1)

        return returns * mask, advantages * mask
