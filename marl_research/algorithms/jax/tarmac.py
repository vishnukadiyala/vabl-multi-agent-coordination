"""TarMAC (Targeted Multi-Agent Communication) in JAX/Flax.

End-to-end JAX implementation for vectorized training with JaxMARL.
Matches the PyTorch TarMAC architecture:
  - Observation encoder (obs -> hidden state)
  - Per communication round: message_head, key_head, query_head (all Linear)
  - Scaled dot-product attention for targeted communication (self-masking)
  - GRU state update after each comm round
  - Policy head (hidden -> action logits)
  - Centralized critic (global state -> value)

Reference: "TarMAC: Targeted Multi-Agent Communication" (Das et al., 2019, ICML).
"""

from typing import NamedTuple, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import chex


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TarMACConfig(NamedTuple):
    """TarMAC hyperparameters."""
    # Architecture
    hidden_dim: int = 128
    message_dim: int = 64
    key_dim: int = 64
    comm_rounds: int = 1
    critic_hidden_dim: int = 128

    # Training
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_clip: float = 0.2
    ppo_epochs: int = 10
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 10.0

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520  # flattened overcooked obs


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    """Observation encoder: obs -> hidden state (2-layer MLP)."""
    hidden_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x


class CommRound(nn.Module):
    """One round of targeted communication.

    For each agent i:
      1. m_i = message_head(h_i)           -- message vector
      2. k_i = key_head(h_i)               -- signature key
      3. q_i = query_head(h_i)             -- attention query
      4. w_ij = softmax(q_i . k_j / sqrt(d_key)) for j != i
      5. c_i = sum_j w_ij * m_j            -- attended message
      6. h_i = GRU(c_i, h_i)              -- update hidden state
    """
    hidden_dim: int
    message_dim: int
    key_dim: int
    n_agents: int

    @nn.compact
    def __call__(self, h):
        """
        Args:
            h: Hidden states for all agents [n_agents, hidden_dim]

        Returns:
            h_new: Updated hidden states [n_agents, hidden_dim]
            attn_weights: Attention weight matrix [n_agents, n_agents]
        """
        n_agents = self.n_agents

        # Produce messages, keys, queries
        messages = nn.Dense(self.message_dim, name="message_head")(h)  # [n_agents, message_dim]
        keys = nn.Dense(self.key_dim, name="key_head")(h)             # [n_agents, key_dim]
        queries = nn.Dense(self.key_dim, name="query_head")(h)        # [n_agents, key_dim]

        # Scaled dot-product attention: scores[i, j] = q_i . k_j / sqrt(d_key)
        scale = jnp.sqrt(jnp.float32(self.key_dim))
        attn_scores = jnp.matmul(queries, keys.T) / scale  # [n_agents, n_agents]

        # Clamp for numerical stability
        attn_scores = jnp.clip(attn_scores, -20.0, 20.0)

        # Self-masking: agent should not attend to its own message
        self_mask = jnp.eye(n_agents)
        attn_scores = jnp.where(self_mask > 0, -1e8, attn_scores)

        # Softmax over sender dimension
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # [n_agents, n_agents]

        # Attended messages: c_i = sum_j w_ij * m_j
        attended = jnp.matmul(attn_weights, messages)  # [n_agents, message_dim]

        # GRU update: integrate attended message into hidden state
        # Process each agent through the same GRU cell via vmap
        # Flax GRUCell returns (carry, output); we need carry as the new hidden state
        gru = nn.GRUCell(features=self.hidden_dim, name="gru_cell")
        h_new = jax.vmap(
            lambda h_i, c_i: gru(h_i[None, :], c_i[None, :])[0].squeeze(0)
        )(h, attended)

        return h_new, attn_weights


class TarMACAgent(nn.Module):
    """Single-step TarMAC forward pass for all agents.

    Architecture:
      1. Encode observations -> initial hidden states
      2. For each comm round: targeted attention + GRU update
      3. Policy head: final hidden -> action logits
    """
    config: TarMACConfig

    @nn.compact
    def __call__(self, obs):
        """
        Args:
            obs: Observations for all agents [n_agents, obs_dim]

        Returns:
            logits: Action logits [n_agents, n_actions]
            attn_weights: Attention weights from last round [n_agents, n_agents]
        """
        cfg = self.config

        # 1. Encode observations -> initial hidden states
        h = ObsEncoder(cfg.hidden_dim)(obs)  # [n_agents, hidden_dim]

        # 2. Communication rounds
        last_attn_weights = jnp.zeros((cfg.n_agents, cfg.n_agents))
        for r in range(cfg.comm_rounds):
            comm = CommRound(
                hidden_dim=cfg.hidden_dim,
                message_dim=cfg.message_dim,
                key_dim=cfg.key_dim,
                n_agents=cfg.n_agents,
                name=f"comm_round_{r}",
            )
            h, attn_w = comm(h)
            last_attn_weights = attn_w

        # 3. Policy head: 2-layer MLP -> action logits
        x = nn.Dense(cfg.hidden_dim, name="policy_fc1")(h)
        x = nn.relu(x)
        logits = nn.Dense(cfg.n_actions, name="policy_fc2")(x)

        return logits, last_attn_weights


class TarMACAgentWrapper(nn.Module):
    """Wrapper around TarMAC to match the standardized per-agent interface.

    Signature: (obs_i, belief_i, teammate_actions_oh, vis_mask_i) -> (logits, new_belief, aux_logits)

    In per-agent mode, encodes the observation through an MLP and policy head
    (no cross-agent communication, since that requires all agents' obs at once).
    belief_i is unused (TarMAC is not recurrent across steps in the original design).
    Returns zeros for aux_logits.
    """
    config: TarMACConfig

    @nn.compact
    def __call__(self, obs_i, belief_i, teammate_actions_oh, vis_mask_i):
        cfg = self.config
        # Encode observation
        x = nn.Dense(cfg.hidden_dim, name="obs_enc1")(obs_i)
        x = nn.relu(x)
        x = nn.Dense(cfg.hidden_dim, name="obs_enc2")(x)
        x = nn.relu(x)

        # Policy head
        x = nn.Dense(cfg.hidden_dim, name="policy_fc1")(x)
        x = nn.relu(x)
        logits = nn.Dense(cfg.n_actions, name="policy_fc2")(x)

        # No recurrent state update; pass belief through unchanged
        new_belief = belief_i
        n_teammates = cfg.n_agents - 1
        aux_logits = jnp.zeros((n_teammates, cfg.n_actions))
        return logits, new_belief, aux_logits


class Critic(nn.Module):
    """Centralized critic: global state -> value (3-layer MLP)."""
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# Transition / rollout storage
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Single-step transition data for PPO training."""
    obs: jnp.ndarray        # [n_agents, obs_dim]
    actions: jnp.ndarray    # [n_agents]
    log_probs: jnp.ndarray  # [n_agents]
    rewards: jnp.ndarray    # scalar (shared reward)
    dones: jnp.ndarray      # scalar
    values: jnp.ndarray     # scalar (centralized value)
    state: jnp.ndarray      # global state for critic


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

class TarMAC:
    """TarMAC algorithm with JAX training."""

    def __init__(self, config: TarMACConfig):
        self.config = config
        self._tarmac_agent = TarMACAgent(config)
        self.agent = TarMACAgentWrapper(config)
        self.critic = Critic(config.critic_hidden_dim)

    def init(self, rng):
        """Initialize parameters and optimizer states.

        Args:
            rng: JAX PRNGKey

        Returns:
            agent_state: TrainState for the TarMAC agent wrapper
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

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, agent_state, critic_state, obs, state, rng):
        """Select actions for all agents via targeted communication.

        Args:
            agent_state: TrainState with agent params
            critic_state: TrainState with critic params
            obs: [n_agents, obs_dim]
            state: global state vector for critic
            rng: PRNGKey

        Returns:
            actions: [n_agents] sampled actions (int32)
            log_probs: [n_agents] log probabilities of sampled actions
            value: scalar state value
            attn_weights: [n_agents, n_agents] attention from last comm round
        """
        # Forward pass: targeted communication + policy
        logits, attn_weights = agent_state.apply_fn(agent_state.params, obs)

        # Sample actions independently per agent
        def sample_agent(logits_i, rng_i):
            action = jax.random.categorical(rng_i, logits_i)
            log_prob = jax.nn.log_softmax(logits_i)[action]
            return action, log_prob

        rngs = jax.random.split(rng, self.config.n_agents)
        actions, log_probs = jax.vmap(sample_agent)(logits, rngs)

        # Centralized value
        value = critic_state.apply_fn(critic_state.params, state)

        return actions, log_probs, value, attn_weights

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_actions(self, agent_state, critic_state, obs, state, actions):
        """Evaluate log probs, entropy, and values for given obs/actions.

        Used during PPO training epochs to recompute log probs under
        the current policy.

        Args:
            agent_state: TrainState with agent params
            critic_state: TrainState with critic params
            obs: [n_agents, obs_dim]
            state: global state vector
            actions: [n_agents] int actions

        Returns:
            log_probs: [n_agents]
            entropy: [n_agents]
            value: scalar
        """
        logits, _ = agent_state.apply_fn(agent_state.params, obs)
        log_probs_all = jax.nn.log_softmax(logits)  # [n_agents, n_actions]
        probs_all = jax.nn.softmax(logits)

        log_probs = jax.vmap(lambda lp, a: lp[a])(log_probs_all, actions)
        entropy = -jnp.sum(probs_all * log_probs_all, axis=-1)  # [n_agents]

        value = critic_state.apply_fn(critic_state.params, state)
        return log_probs, entropy, value

    @partial(jax.jit, static_argnums=(0,))
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: [T] reward per step
            values: [T] value estimates
            dones: [T] done flags

        Returns:
            returns: [T]
            advantages: [T]
        """
        cfg = self.config
        T = rewards.shape[0]

        def _gae_step(carry, t_rev):
            last_gae, next_value = carry
            t = T - 1 - t_rev
            reward_t = rewards[t]
            value_t = values[t]
            done_t = dones[t]

            delta = reward_t + cfg.gamma * next_value * (1.0 - done_t) - value_t
            gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done_t) * last_gae
            ret = gae + value_t
            return (gae, value_t), (ret, gae)

        init_carry = (jnp.float32(0.0), jnp.float32(0.0))
        _, (returns_rev, advantages_rev) = jax.lax.scan(
            _gae_step, init_carry, jnp.arange(T)
        )

        # Reverse back to chronological order
        returns = jnp.flip(returns_rev)
        advantages = jnp.flip(advantages_rev)
        return returns, advantages

    @partial(jax.jit, static_argnums=(0,))
    def update(self, agent_state, critic_state, batch):
        """Perform one full PPO update (multiple epochs) on a batch.

        Args:
            agent_state: TrainState for agent
            critic_state: TrainState for critic
            batch: dictionary-like with fields:
                obs: [T, n_agents, obs_dim]
                actions: [T, n_agents]
                log_probs: [T, n_agents]
                rewards: [T]
                dones: [T]
                values: [T]
                state: [T, state_dim]

        Returns:
            agent_state: updated TrainState
            critic_state: updated TrainState
            metrics: dict of scalar training metrics
        """
        cfg = self.config

        obs = batch["obs"]            # [T, n_agents, obs_dim]
        actions = batch["actions"]    # [T, n_agents]
        old_log_probs = batch["log_probs"]  # [T, n_agents]
        rewards = batch["rewards"]    # [T]
        dones = batch["dones"]        # [T]
        old_values = batch["values"]  # [T]
        states = batch["state"]       # [T, state_dim]

        # Compute GAE
        returns, advantages = self.compute_gae(rewards, old_values, dones)

        # Normalize advantages
        adv_mean = jnp.mean(advantages)
        adv_std = jnp.maximum(jnp.std(advantages), 1e-8)
        advantages = jnp.clip((advantages - adv_mean) / adv_std, -10.0, 10.0)

        # Sum old log probs across agents for joint ratio
        old_joint_log_probs = jnp.sum(old_log_probs, axis=-1)  # [T]

        def ppo_epoch(carry, _):
            a_state, c_state = carry

            def actor_loss_fn(agent_params):
                """Compute PPO clipped policy loss + entropy bonus."""
                def step_fn(obs_t, actions_t):
                    logits, _ = self.agent.apply(agent_params, obs_t)
                    log_probs_all = jax.nn.log_softmax(logits)
                    probs_all = jax.nn.softmax(logits)
                    lp = jax.vmap(lambda lp_i, a_i: lp_i[a_i])(log_probs_all, actions_t)
                    ent = -jnp.sum(probs_all * log_probs_all, axis=-1)
                    return lp, ent

                new_log_probs, entropy = jax.vmap(step_fn)(obs, actions)
                # new_log_probs: [T, n_agents], entropy: [T, n_agents]

                # Joint log probs
                new_joint_lp = jnp.sum(new_log_probs, axis=-1)  # [T]
                ratio = jnp.exp(new_joint_lp - old_joint_log_probs)
                ratio = jnp.clip(ratio, 0.0, 10.0)

                surr1 = ratio * advantages
                surr2 = jnp.clip(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages
                policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                # Mean entropy across agents and timesteps
                entropy_loss = -jnp.mean(entropy)

                total_loss = policy_loss + cfg.entropy_coef * entropy_loss
                return total_loss, (policy_loss, entropy_loss, jnp.mean(entropy))

            def critic_loss_fn(critic_params):
                """Compute value loss."""
                def value_step(state_t):
                    return self.critic.apply(critic_params, state_t)
                values = jax.vmap(value_step)(states)  # [T]
                value_loss = jnp.mean((values - returns) ** 2)
                return cfg.value_loss_coef * value_loss, value_loss

            # Actor gradient step
            (actor_total, (pol_loss, ent_loss, mean_ent)), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(a_state.params)
            a_state = a_state.apply_gradients(grads=actor_grads)

            # Critic gradient step
            (critic_total, val_loss), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(c_state.params)
            c_state = c_state.apply_gradients(grads=critic_grads)

            return (a_state, c_state), (pol_loss, val_loss, mean_ent)

        (agent_state, critic_state), (pol_losses, val_losses, entropies) = jax.lax.scan(
            ppo_epoch, (agent_state, critic_state), None, length=cfg.ppo_epochs
        )

        metrics = {
            "policy_loss": jnp.mean(pol_losses),
            "value_loss": jnp.mean(val_losses),
            "entropy": jnp.mean(entropies),
            "loss": jnp.mean(pol_losses) + jnp.mean(val_losses),
        }

        return agent_state, critic_state, metrics
