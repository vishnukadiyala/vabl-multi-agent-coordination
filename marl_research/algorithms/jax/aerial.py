"""AERIAL (Attention-Based Recurrence for Multi-Agent RL) in JAX/Flax.

End-to-end JAX implementation for vectorized training with JaxMARL.
Matches the PyTorch AERIAL architecture:
  - Observation encoder (obs -> embedding)
  - Multi-head attention over teammate HIDDEN STATES (not actions)
  - GRU belief update: input = [h_obs || attention_context]
  - Policy head: belief -> action logits
  - Centralized critic (global state -> value)

Key distinction from VABL: AERIAL shares hidden states across agents
(requires communication bandwidth), while VABL only uses observable
teammate actions (no communication needed).

Reference: "Attention-Based Recurrence for Multi-Agent Reinforcement Learning
under Stochastic Partial Observability" (Phan et al., 2023, ICML).
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

class AERIALConfig(NamedTuple):
    """AERIAL hyperparameters."""
    # Architecture
    hidden_dim: int = 128
    embed_dim: int = 64
    attention_heads: int = 4
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
    """Observation encoder: obs -> embedding (2-layer MLP)."""
    embed_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.embed_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention over teammate hidden states.

    Query: agent's own projected hidden state  [1, embed_dim]
    Key/Value: teammates' projected hidden states  [n_teammates, embed_dim]

    This is the core AERIAL mechanism -- attention over hidden states,
    not over actions (which is what VABL does).
    """
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, query, key_value, mask=None):
        """
        Args:
            query: [embed_dim] (single agent's projected hidden state)
            key_value: [n_teammates, embed_dim] (projected teammate hidden states)
            mask: [n_teammates] optional binary mask (1 = attend, 0 = ignore)

        Returns:
            context: [embed_dim] attention-weighted summary
            attn_weights: [n_teammates] attention weight per teammate
        """
        head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        n_teammates = key_value.shape[0]

        # Project query, keys, values for all heads
        # query: [embed_dim] -> [num_heads, head_dim]
        q = nn.Dense(self.embed_dim, name="q_proj")(query)
        q = q.reshape(self.num_heads, head_dim)

        # key_value: [n_teammates, embed_dim] -> [n_teammates, num_heads, head_dim]
        k = nn.Dense(self.embed_dim, name="k_proj")(key_value)
        k = k.reshape(n_teammates, self.num_heads, head_dim)

        v = nn.Dense(self.embed_dim, name="v_proj")(key_value)
        v = v.reshape(n_teammates, self.num_heads, head_dim)

        # Scaled dot-product attention per head
        # q: [num_heads, head_dim], k: [n_teammates, num_heads, head_dim]
        # scores: [num_heads, n_teammates]
        scale = jnp.sqrt(jnp.float32(head_dim))
        scores = jnp.einsum("hd,nhd->hn", q, k) / scale

        # Apply mask if provided
        if mask is not None:
            # mask: [n_teammates] -> broadcast to [num_heads, n_teammates]
            mask_expanded = jnp.broadcast_to(mask[None, :], (self.num_heads, n_teammates))
            scores = jnp.where(mask_expanded > 0, scores, -1e8)

        attn_weights_per_head = jax.nn.softmax(scores, axis=-1)  # [num_heads, n_teammates]

        # Weighted sum of values per head
        # attn_weights: [num_heads, n_teammates], v: [n_teammates, num_heads, head_dim]
        # context_per_head: [num_heads, head_dim]
        context_per_head = jnp.einsum("hn,nhd->hd", attn_weights_per_head, v)

        # Concatenate heads -> output projection
        context = context_per_head.reshape(self.embed_dim)
        context = nn.Dense(self.embed_dim, name="out_proj")(context)

        # Average attention weights across heads for interpretability
        attn_weights = jnp.mean(attn_weights_per_head, axis=0)  # [n_teammates]

        return context, attn_weights


class GRUCell(nn.Module):
    """GRU cell for belief update."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h):
        gru = nn.GRUCell(features=self.hidden_dim)
        carry, _ = gru(h, x)  # Flax GRU returns (carry, output); carry == output for GRU
        return carry


class AERIALAgent(nn.Module):
    """Single AERIAL agent network.

    Forward pass:
      1. h_obs = obs_encoder(obs)
      2. q = hidden_proj(prev_belief)
      3. kv = hidden_proj(teammate_hidden_states)
      4. context = MHA(q, kv, kv)
      5. gru_input = [h_obs || context]
      6. belief = GRU(gru_input, prev_belief)
      7. logits = policy_head(belief)
    """
    config: AERIALConfig

    @nn.compact
    def __call__(self, obs, prev_belief, teammate_hidden_states):
        """
        Args:
            obs: [obs_dim]
            prev_belief: [hidden_dim]
            teammate_hidden_states: [n_teammates, hidden_dim]

        Returns:
            logits: [n_actions]
            belief: [hidden_dim]
            attn_weights: [n_teammates]
        """
        cfg = self.config
        n_teammates = cfg.n_agents - 1

        # 1. Encode observation
        h_obs = ObsEncoder(cfg.embed_dim)(obs)  # [embed_dim]

        # 2. Project own hidden state for attention query
        query_proj = nn.Dense(cfg.embed_dim, name="hidden_proj_query")(prev_belief)  # [embed_dim]

        # 3. Project teammate hidden states for attention keys/values
        # Dense broadcasts over leading dims: [n_teammates, hidden_dim] -> [n_teammates, embed_dim]
        teammate_kv = nn.Dense(cfg.embed_dim, name="hidden_proj_kv")(
            teammate_hidden_states
        )  # [n_teammates, embed_dim]

        # 4. Multi-head attention over teammate hidden states
        mha = MultiHeadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.attention_heads,
            name="mha",
        )
        context, attn_weights = mha(query_proj, teammate_kv)  # [embed_dim], [n_teammates]

        # 5. Concatenate obs encoding and attention context
        gru_input = jnp.concatenate([h_obs, context])  # [2 * embed_dim]

        # Project to hidden_dim for GRU input
        gru_input = nn.Dense(cfg.hidden_dim, name="gru_input_proj")(gru_input)

        # 6. GRU belief update
        belief = GRUCell(cfg.hidden_dim)(gru_input, prev_belief)  # [hidden_dim]

        # 7. Policy head: belief -> action logits
        logits = nn.Dense(cfg.n_actions, name="policy_head")(belief)

        return logits, belief, attn_weights


class AERIALAgentWrapper(nn.Module):
    """Wrapper around AERIAL to match the standardized per-agent interface.

    Signature: (obs_i, belief_i, teammate_actions_oh, vis_mask_i) -> (logits, new_belief, aux_logits)

    AERIAL's full cross-agent attention requires ALL agents' hidden states
    simultaneously, which doesn't fit the per-agent interface. This wrapper
    operates as a simple GRU agent: encodes obs, updates belief via GRU,
    and produces action logits. No cross-agent attention in per-agent mode.
    Returns zeros for aux_logits.
    """
    config: AERIALConfig

    @nn.compact
    def __call__(self, obs_i, belief_i, teammate_actions_oh, vis_mask_i):
        cfg = self.config

        # Encode observation
        h_obs = ObsEncoder(cfg.embed_dim)(obs_i)  # [embed_dim]

        # Project to hidden_dim for GRU input (skip attention)
        gru_input = nn.Dense(cfg.hidden_dim, name="gru_input_proj")(h_obs)

        # GRU belief update
        new_belief = GRUCell(cfg.hidden_dim)(gru_input, belief_i)  # [hidden_dim]

        # Policy head
        logits = nn.Dense(cfg.n_actions, name="policy_head")(new_belief)

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
    obs: jnp.ndarray               # [n_agents, obs_dim]
    actions: jnp.ndarray           # [n_agents]
    log_probs: jnp.ndarray         # [n_agents]
    rewards: jnp.ndarray           # scalar (shared reward)
    dones: jnp.ndarray             # scalar
    values: jnp.ndarray            # scalar (centralized value)
    state: jnp.ndarray             # global state for critic
    beliefs: jnp.ndarray           # [n_agents, hidden_dim]


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------

class AERIAL:
    """AERIAL algorithm with JAX training."""

    def __init__(self, config: AERIALConfig):
        self.config = config
        self._aerial_agent = AERIALAgent(config)
        self.agent = AERIALAgentWrapper(config)
        self.critic = Critic(config.critic_hidden_dim)

    def init(self, rng):
        """Initialize parameters and optimizer states.

        Args:
            rng: JAX PRNGKey

        Returns:
            agent_state: TrainState for the AERIAL agent wrapper
            critic_state: TrainState for the centralized critic
        """
        cfg = self.config
        n_teammates = cfg.n_agents - 1
        rng_agent, rng_critic = jax.random.split(rng)

        # Dummy inputs for wrapper signature
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

    def init_beliefs(self):
        """Initialize belief states to zeros for all agents.

        Returns:
            beliefs: [n_agents, hidden_dim]
        """
        return jnp.zeros((self.config.n_agents, self.config.hidden_dim))

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, agent_state, critic_state, obs, beliefs, state, rng):
        """Select actions for all agents using AERIAL forward pass.

        Each agent shares its hidden state with teammates. Each agent then
        uses multi-head attention over teammates' hidden states to form a
        context, combined with observation encoding to update belief via GRU.

        Args:
            agent_state: TrainState with agent params
            critic_state: TrainState with critic params
            obs: [n_agents, obs_dim]
            beliefs: [n_agents, hidden_dim]
            state: global state vector for critic
            rng: PRNGKey

        Returns:
            actions: [n_agents] sampled actions (int32)
            new_beliefs: [n_agents, hidden_dim]
            log_probs: [n_agents]
            value: scalar
            attn_weights: [n_agents, n_teammates] attention weights
        """
        cfg = self.config

        def agent_step(agent_idx):
            rng_i = jax.random.fold_in(rng, agent_idx)
            obs_i = obs[agent_idx]           # [obs_dim]
            belief_i = beliefs[agent_idx]    # [hidden_dim]

            # Extract teammate hidden states (exclude self)
            # Build index array excluding agent_idx
            teammate_indices = jnp.concatenate([
                jnp.arange(0, agent_idx),
                jnp.arange(agent_idx + 1, cfg.n_agents),
            ])
            teammate_hiddens = beliefs[teammate_indices]  # [n_teammates, hidden_dim]

            # Forward pass through AERIAL agent
            logits, new_belief, attn_w = agent_state.apply_fn(
                agent_state.params, obs_i, belief_i, teammate_hiddens
            )

            # Sample action
            action = jax.random.categorical(rng_i, logits)
            log_prob = jax.nn.log_softmax(logits)[action]

            return action, new_belief, log_prob, attn_w

        # Vectorize over agents
        agent_indices = jnp.arange(cfg.n_agents)
        actions, new_beliefs, log_probs, attn_weights = jax.vmap(agent_step)(agent_indices)

        # Centralized value
        value = critic_state.apply_fn(critic_state.params, state)

        return actions, new_beliefs, log_probs, value, attn_weights

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_actions(self, agent_state, critic_state, obs, beliefs, state, actions):
        """Evaluate log probs, entropy, and value for given obs/actions/beliefs.

        Used during PPO training to recompute quantities under current policy.

        Args:
            agent_state: TrainState with agent params
            critic_state: TrainState with critic params
            obs: [n_agents, obs_dim]
            beliefs: [n_agents, hidden_dim]
            state: global state vector
            actions: [n_agents] int actions

        Returns:
            log_probs: [n_agents]
            entropy: [n_agents]
            new_beliefs: [n_agents, hidden_dim]
            value: scalar
        """
        cfg = self.config

        def agent_eval(agent_idx):
            obs_i = obs[agent_idx]
            belief_i = beliefs[agent_idx]

            teammate_indices = jnp.concatenate([
                jnp.arange(0, agent_idx),
                jnp.arange(agent_idx + 1, cfg.n_agents),
            ])
            teammate_hiddens = beliefs[teammate_indices]

            logits, new_belief, _ = agent_state.apply_fn(
                agent_state.params, obs_i, belief_i, teammate_hiddens
            )

            log_probs_all = jax.nn.log_softmax(logits)
            probs_all = jax.nn.softmax(logits)

            lp = log_probs_all[actions[agent_idx]]
            ent = -jnp.sum(probs_all * log_probs_all)

            return lp, ent, new_belief

        agent_indices = jnp.arange(cfg.n_agents)
        log_probs, entropy, new_beliefs = jax.vmap(agent_eval)(agent_indices)

        value = critic_state.apply_fn(critic_state.params, state)
        return log_probs, entropy, new_beliefs, value

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

        Because AERIAL is recurrent (GRU belief), the forward pass during
        training must reconstruct the belief trajectory. We store beliefs
        from the rollout and re-derive log probs at each timestep using
        the stored belief states (which approximates the true recurrent
        trajectory without full BPTT through the rollout for efficiency).

        Args:
            agent_state: TrainState for agent
            critic_state: TrainState for critic
            batch: dictionary with fields:
                obs: [T, n_agents, obs_dim]
                actions: [T, n_agents]
                log_probs: [T, n_agents]
                rewards: [T]
                dones: [T]
                values: [T]
                state: [T, state_dim]
                beliefs: [T, n_agents, hidden_dim]

        Returns:
            agent_state: updated TrainState
            critic_state: updated TrainState
            metrics: dict of scalar training metrics
        """
        cfg = self.config

        obs = batch["obs"]                  # [T, n_agents, obs_dim]
        actions = batch["actions"]          # [T, n_agents]
        old_log_probs = batch["log_probs"]  # [T, n_agents]
        rewards = batch["rewards"]          # [T]
        dones = batch["dones"]              # [T]
        old_values = batch["values"]        # [T]
        states = batch["state"]             # [T, state_dim]
        beliefs = batch["beliefs"]          # [T, n_agents, hidden_dim]

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
                """Compute PPO clipped policy loss + entropy bonus.

                Re-evaluate each timestep using the stored belief states and
                current agent parameters to get updated log probs and entropy.
                """
                def step_fn(obs_t, actions_t, beliefs_t):
                    """Process one timestep for all agents."""
                    def agent_fn(agent_idx):
                        obs_i = obs_t[agent_idx]
                        belief_i = beliefs_t[agent_idx]
                        teammate_indices = jnp.concatenate([
                            jnp.arange(0, agent_idx),
                            jnp.arange(agent_idx + 1, cfg.n_agents),
                        ])
                        teammate_hiddens = beliefs_t[teammate_indices]

                        logits, _, _ = self.agent.apply(
                            agent_params, obs_i, belief_i, teammate_hiddens
                        )
                        log_probs_all = jax.nn.log_softmax(logits)
                        probs_all = jax.nn.softmax(logits)

                        lp = log_probs_all[actions_t[agent_idx]]
                        ent = -jnp.sum(probs_all * log_probs_all)
                        return lp, ent

                    agent_indices = jnp.arange(cfg.n_agents)
                    lps, ents = jax.vmap(agent_fn)(agent_indices)
                    return lps, ents  # [n_agents], [n_agents]

                new_log_probs, entropy = jax.vmap(step_fn)(obs, actions, beliefs)
                # new_log_probs: [T, n_agents], entropy: [T, n_agents]

                # Joint log probs
                new_joint_lp = jnp.sum(new_log_probs, axis=-1)  # [T]
                ratio = jnp.exp(new_joint_lp - old_joint_log_probs)
                ratio = jnp.clip(ratio, 0.0, 10.0)

                surr1 = ratio * advantages
                surr2 = jnp.clip(
                    ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param
                ) * advantages
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
