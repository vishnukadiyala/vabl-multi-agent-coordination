"""MAPPO (Multi-Agent PPO) in JAX/Flax.

End-to-end JAX implementation for vectorized training with JaxMARL.
Architecture:
  - Actor: 2-layer MLP (obs -> hidden -> hidden) + GRU + policy head
    Parameter-shared across agents.
  - Critic: 3-layer MLP on global state -> value (centralized, CTDE)
  - PPO training with GAE, clipped surrogate objective, value clipping
  - Separate actor/critic optimizers
"""

from typing import NamedTuple, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import chex


class MAPPOConfig(NamedTuple):
    """MAPPO hyperparameters."""
    # Architecture
    hidden_dim: int = 128
    critic_hidden_dim: int = 128

    # Training
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_clip: float = 0.2
    ppo_epochs: int = 10
    num_minibatches: int = 1
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 10.0
    max_grad_norm: float = 10.0

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520  # flattened overcooked obs


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class GRUCell(nn.Module):
    """GRU cell for recurrent hidden state update."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h):
        gru = nn.GRUCell(features=self.hidden_dim)
        carry, _ = gru(h, x)  # Flax GRU returns (carry, output); carry == output for GRU
        return carry


class Actor(nn.Module):
    """MAPPO actor: 2-layer MLP + GRU + policy head.

    Parameter-shared across agents (same network applied to each agent's obs).
    """
    hidden_dim: int
    n_actions: int

    @nn.compact
    def __call__(self, obs, hidden):
        """
        Args:
            obs: [obs_dim] single agent observation
            hidden: [hidden_dim] GRU hidden state

        Returns:
            logits: [n_actions] action logits
            new_hidden: [hidden_dim] updated GRU hidden state
        """
        # 2-layer MLP feature extractor
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # GRU recurrence
        x = nn.Dense(self.hidden_dim)(x)  # project to GRU input dim
        new_hidden = GRUCell(self.hidden_dim)(x, hidden)

        # Policy head
        logits = nn.Dense(self.n_actions)(new_hidden)

        return logits, new_hidden


class MAPPOAgentWrapper(nn.Module):
    """Wrapper around Actor to match the standardized agent interface.

    Signature: (obs_i, belief_i, teammate_actions_oh, vis_mask_i) -> (logits, new_belief, aux_logits)

    Maps belief_i to the GRU hidden state. Ignores teammate_actions_oh and vis_mask_i
    (MAPPO doesn't use them). Returns zeros for aux_logits.
    """
    hidden_dim: int
    n_actions: int
    n_agents: int

    @nn.compact
    def __call__(self, obs_i, belief_i, teammate_actions_oh, vis_mask_i):
        actor = Actor(self.hidden_dim, self.n_actions)
        logits, new_belief = actor(obs_i, belief_i)
        n_teammates = self.n_agents - 1
        aux_logits = jnp.zeros((n_teammates, self.n_actions))
        return logits, new_belief, aux_logits


class Critic(nn.Module):
    """Centralized critic: global state -> value.

    3-layer MLP matching the VABL critic architecture.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer (NamedTuple for jax-friendly storage)
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Single timestep transition for all agents."""
    obs: chex.Array          # [n_agents, obs_dim]
    state: chex.Array        # [state_dim]
    actions: chex.Array      # [n_agents]
    log_probs: chex.Array    # [n_agents]
    rewards: chex.Array      # [n_agents]
    dones: chex.Array        # [n_agents]
    hiddens: chex.Array      # [n_agents, hidden_dim]
    values: chex.Array       # [n_agents]


# ---------------------------------------------------------------------------
# MAPPO algorithm
# ---------------------------------------------------------------------------

class MAPPO:
    """MAPPO algorithm with JAX training."""

    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.actor = Actor(config.hidden_dim, config.n_actions)
        self.agent = MAPPOAgentWrapper(config.hidden_dim, config.n_actions, config.n_agents)
        self.critic = Critic(config.critic_hidden_dim)

    def init(self, rng):
        """Initialize agent (wrapper) and critic parameters + optimizer states."""
        cfg = self.config
        rng_agent, rng_critic = jax.random.split(rng)

        # Dummy inputs for parameter initialization (matches wrapper signature)
        n_teammates = cfg.n_agents - 1
        dummy_obs = jnp.zeros(cfg.obs_dim)
        dummy_belief = jnp.zeros(cfg.hidden_dim)
        dummy_teammate_actions_oh = jnp.zeros((n_teammates, cfg.n_actions))
        dummy_vis_mask = jnp.zeros(n_teammates)
        dummy_state = jnp.zeros(cfg.obs_dim * cfg.n_agents)

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
            apply_fn=self.agent.apply, params=agent_params, tx=agent_tx
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=critic_tx
        )

        return agent_state, critic_state

    def init_hidden(self):
        """Initial GRU hidden state (zeros) for all agents."""
        return jnp.zeros((self.config.n_agents, self.config.hidden_dim))

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, actor_state, critic_state, obs, hiddens, state, rng):
        """Select actions for all agents.

        Args:
            actor_state: TrainState with actor params
            critic_state: TrainState with critic params
            obs: [n_agents, obs_dim]
            hiddens: [n_agents, hidden_dim] GRU hidden states
            state: [state_dim] global state for critic
            rng: PRNGKey

        Returns:
            actions: [n_agents] sampled actions
            log_probs: [n_agents] log probabilities of sampled actions
            new_hiddens: [n_agents, hidden_dim]
            values: [n_agents] centralized value estimates
        """
        cfg = self.config

        # Actor: vmap over agents (parameter-shared)
        def actor_forward(obs_i, hidden_i):
            logits, new_hidden = actor_state.apply_fn(
                actor_state.params, obs_i, hidden_i
            )
            return logits, new_hidden

        all_logits, new_hiddens = jax.vmap(actor_forward)(obs, hiddens)
        # all_logits: [n_agents, n_actions], new_hiddens: [n_agents, hidden_dim]

        # Sample actions
        rngs = jax.random.split(rng, cfg.n_agents)
        actions = jax.vmap(jax.random.categorical)(rngs, all_logits)  # [n_agents]

        # Log probs
        all_log_probs = jax.nn.log_softmax(all_logits)  # [n_agents, n_actions]
        log_probs = all_log_probs[jnp.arange(cfg.n_agents), actions]  # [n_agents]

        # Critic: centralized value (same value for all agents, but we broadcast)
        value = critic_state.apply_fn(critic_state.params, state)  # scalar
        values = jnp.broadcast_to(value, (cfg.n_agents,))

        return actions, log_probs, new_hiddens, values

    @partial(jax.jit, static_argnums=(0,))
    def get_value(self, critic_state, state):
        """Get value estimate for a global state.

        Args:
            critic_state: TrainState with critic params
            state: [state_dim] global state

        Returns:
            value: scalar value estimate
        """
        return critic_state.apply_fn(critic_state.params, state)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_actions(self, actor_state, obs, hiddens, actions):
        """Evaluate log_probs and entropy for given obs/actions.

        Used during PPO update on stored rollout data.

        Args:
            actor_state: TrainState with actor params
            obs: [n_agents, obs_dim]
            hiddens: [n_agents, hidden_dim]
            actions: [n_agents] integer actions

        Returns:
            log_probs: [n_agents]
            entropy: [n_agents]
        """
        cfg = self.config

        def actor_forward(obs_i, hidden_i):
            logits, _ = actor_state.apply_fn(actor_state.params, obs_i, hidden_i)
            return logits

        all_logits = jax.vmap(actor_forward)(obs, hiddens)  # [n_agents, n_actions]

        # Log probs of taken actions
        all_log_probs = jax.nn.log_softmax(all_logits)
        log_probs = all_log_probs[jnp.arange(cfg.n_agents), actions]

        # Entropy: -sum(p * log p)
        probs = jax.nn.softmax(all_logits)
        entropy = -jnp.sum(probs * all_log_probs, axis=-1)  # [n_agents]

        return log_probs, entropy

    @staticmethod
    def compute_gae(
        rewards: chex.Array,
        values: chex.Array,
        dones: chex.Array,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[chex.Array, chex.Array]:
        """Compute GAE advantages and returns.

        Args:
            rewards: [T] rewards for one agent
            values: [T] value estimates
            dones: [T] episode done flags
            last_value: value estimate at T+1
            gamma: discount factor
            gae_lambda: GAE lambda

        Returns:
            advantages: [T]
            returns: [T]
        """
        T = rewards.shape[0]

        def scan_fn(gae, t_rev):
            t = T - 1 - t_rev
            r_t = rewards[t]
            v_t = values[t]
            d_t = dones[t]

            # Next value: either last_value (if t is last step) or values[t+1]
            next_val = jnp.where(t == T - 1, last_value, values[t + 1])
            next_done = jnp.where(t == T - 1, 0.0, dones[t + 1])

            delta = r_t + gamma * next_val * (1.0 - d_t) - v_t
            gae = delta + gamma * gae_lambda * (1.0 - d_t) * gae
            return gae

        # Reverse scan over timesteps
        advantages = jax.lax.scan(
            lambda carry, t_rev: (scan_fn(carry, t_rev), scan_fn(carry, t_rev)),
            0.0,
            jnp.arange(T),
        )[1]
        # scan returns (final_carry, stacked_outputs) — we want stacked

        returns = advantages + values
        return advantages, returns

    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        batch_obs: chex.Array,
        batch_hiddens: chex.Array,
        batch_actions: chex.Array,
        batch_old_log_probs: chex.Array,
        batch_advantages: chex.Array,
        batch_returns: chex.Array,
        batch_old_values: chex.Array,
        batch_states: chex.Array,
        rng: jax.random.PRNGKey,
    ) -> Tuple[TrainState, TrainState, dict]:
        """Run PPO update for ppo_epochs.

        All batch arrays have shape [T, n_agents, ...] except batch_states [T, state_dim].

        Args:
            actor_state: current actor TrainState
            critic_state: current critic TrainState
            batch_obs: [T, n_agents, obs_dim]
            batch_hiddens: [T, n_agents, hidden_dim]
            batch_actions: [T, n_agents]
            batch_old_log_probs: [T, n_agents]
            batch_advantages: [T, n_agents]
            batch_returns: [T, n_agents]
            batch_old_values: [T, n_agents]
            batch_states: [T, state_dim]
            rng: PRNGKey

        Returns:
            new_actor_state, new_critic_state, metrics dict
        """
        cfg = self.config
        T = batch_obs.shape[0]

        # Normalize advantages
        adv_mean = jnp.mean(batch_advantages)
        adv_std = jnp.std(batch_advantages) + 1e-8
        batch_advantages_normed = (batch_advantages - adv_mean) / adv_std

        def ppo_epoch_step(carry, epoch_rng):
            actor_st, critic_st = carry

            # --- Actor loss ---
            def actor_loss_fn(actor_params):
                # Evaluate all timesteps and agents
                def eval_step(obs_t, hidden_t, action_t):
                    # obs_t: [n_agents, obs_dim], hidden_t: [n_agents, hidden_dim]
                    def fwd(obs_i, hidden_i):
                        logits, _ = self.actor.apply(actor_params, obs_i, hidden_i)
                        return logits
                    all_logits = jax.vmap(fwd)(obs_t, hidden_t)
                    all_log_probs = jax.nn.log_softmax(all_logits)
                    log_probs = all_log_probs[jnp.arange(cfg.n_agents), action_t]
                    probs = jax.nn.softmax(all_logits)
                    entropy = -jnp.sum(probs * all_log_probs, axis=-1)
                    return log_probs, entropy

                new_log_probs, entropies = jax.vmap(eval_step)(
                    batch_obs, batch_hiddens, batch_actions
                )
                # new_log_probs: [T, n_agents], entropies: [T, n_agents]

                # PPO clipped objective
                ratio = jnp.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages_normed
                surr2 = jnp.clip(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * batch_advantages_normed
                policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                entropy_loss = -jnp.mean(entropies)

                total_loss = policy_loss + cfg.entropy_coef * entropy_loss

                return total_loss, {
                    'policy_loss': policy_loss,
                    'entropy': jnp.mean(entropies),
                    'approx_kl': jnp.mean(0.5 * (new_log_probs - batch_old_log_probs) ** 2),
                    'clip_frac': jnp.mean(
                        jnp.abs(ratio - 1.0) > cfg.clip_param
                    ),
                }

            actor_grads, actor_metrics = jax.grad(actor_loss_fn, has_aux=True)(
                actor_st.params
            )
            actor_st = actor_st.apply_gradients(grads=actor_grads)

            # --- Critic loss ---
            def critic_loss_fn(critic_params):
                # Evaluate value for each timestep
                def eval_value(state_t):
                    return self.critic.apply(critic_params, state_t)

                new_values_scalar = jax.vmap(eval_value)(batch_states)  # [T]
                # Broadcast to [T, n_agents] to match returns shape
                new_values = jnp.broadcast_to(
                    new_values_scalar[:, None], batch_returns.shape
                )

                # Value clipping
                value_pred_clipped = batch_old_values + jnp.clip(
                    new_values - batch_old_values,
                    -cfg.value_clip,
                    cfg.value_clip,
                )
                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * jnp.mean(
                    jnp.maximum(value_loss_unclipped, value_loss_clipped)
                )

                return value_loss, {'value_loss': value_loss}

            critic_grads, critic_metrics = jax.grad(critic_loss_fn, has_aux=True)(
                critic_st.params
            )
            critic_st = critic_st.apply_gradients(grads=critic_grads)

            metrics = {**actor_metrics, **critic_metrics}
            return (actor_st, critic_st), metrics

        # Run ppo_epochs
        epoch_rngs = jax.random.split(rng, cfg.ppo_epochs)
        (actor_state, critic_state), all_metrics = jax.lax.scan(
            ppo_epoch_step,
            (actor_state, critic_state),
            epoch_rngs,
        )

        # Average metrics across epochs
        metrics = jax.tree.map(lambda x: jnp.mean(x), all_metrics)

        return actor_state, critic_state, metrics
