"""VABL (Variational Attention-based Belief Learning) in JAX/Flax.

End-to-end JAX implementation for vectorized training with JaxMARL.
Matches the PyTorch VABL architecture:
  - Feature encoder (obs → embedding)
  - Action encoder (actions → embedding) + identity embeddings
  - Multi-head attention (belief query, action keys/values)
  - GRU belief update
  - Policy head + auxiliary prediction head
  - Centralized critic (global state → value)
"""

from typing import NamedTuple, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import chex


class VABLConfig(NamedTuple):
    """VABL hyperparameters."""
    # Architecture
    embed_dim: int = 64
    hidden_dim: int = 128
    attention_heads: int = 4
    aux_hidden_dim: int = 64
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
    aux_lambda: float = 0.05
    grad_clip: float = 10.0

    # Ablation flags
    use_attention: bool = True   # False = mean pooling over teammate actions
    use_aux_loss: bool = True    # False = no auxiliary prediction (aux_lambda ignored)

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520  # flattened overcooked obs


class FeatureEncoder(nn.Module):
    """Observation encoder: obs → embedding."""
    embed_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.embed_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        return x


class ActionEncoder(nn.Module):
    """Teammate action encoder: one-hot action → embedding."""
    embed_dim: int
    n_actions: int

    @nn.compact
    def __call__(self, action_onehot):
        x = nn.Dense(self.embed_dim)(action_onehot)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim)(x)
        x = nn.relu(x)
        return x


class GRUCell(nn.Module):
    """GRU cell for belief update."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x, h):
        gru = nn.GRUCell(features=self.hidden_dim)
        new_carry, _ = gru(h, x)  # Flax GRU returns (new_carry, output)
        return new_carry


class VABLAgent(nn.Module):
    """Single VABL agent network.

    Forward pass:
      1. h_obs = encode(obs)
      2. h_actions = encode(teammate_actions) + identity_emb
      3. context = MHA(project(belief), h_actions, vis_mask)
      4. belief = GRU([h_obs || context], prev_belief)
      5. logits = policy_head(belief)
      6. aux_logits = aux_head(belief)
    """
    config: VABLConfig

    @nn.compact
    def __call__(self, obs, prev_belief, prev_teammate_actions, visibility_mask):
        """
        Args:
            obs: [embed_dim] or [batch, obs_dim]
            prev_belief: [hidden_dim]
            prev_teammate_actions: [n_teammates, n_actions] one-hot
            visibility_mask: [n_teammates] binary
        Returns:
            logits: [n_actions]
            belief: [hidden_dim]
            aux_logits: [n_teammates, n_actions]
        """
        cfg = self.config
        n_teammates = cfg.n_agents - 1

        # 1. Encode observation
        h_obs = FeatureEncoder(cfg.embed_dim)(obs)

        # 2. Encode teammate actions
        action_enc = ActionEncoder(cfg.embed_dim, cfg.n_actions)
        h_actions = jax.vmap(action_enc)(prev_teammate_actions)  # [n_teammates, embed_dim]

        # Identity embeddings
        identity_emb = self.param(
            'identity_emb',
            nn.initializers.normal(stddev=0.01),
            (cfg.n_agents, cfg.embed_dim),
        )
        # Add identity for each teammate (skip self = agent 0 convention)
        h_actions = h_actions + identity_emb[1:n_teammates + 1]

        # 3. Context: attention or mean pooling
        if cfg.use_attention:
            belief_proj = nn.Dense(cfg.embed_dim)(prev_belief)
            query = belief_proj[None, :]  # [1, embed_dim]
            attn_logits = jnp.matmul(query, h_actions.T) / jnp.sqrt(cfg.embed_dim)
            attn_logits = attn_logits.squeeze(0)  # [n_teammates]
            attn_mask = jnp.where(visibility_mask > 0, 0.0, -1e8)
            attn_logits = attn_logits + attn_mask
            attn_weights = jax.nn.softmax(attn_logits)
            context = jnp.dot(attn_weights, h_actions)  # [embed_dim]
        else:
            # Mean pooling over visible teammates
            vis_expanded = visibility_mask[:, None]  # [n_teammates, 1]
            context = (h_actions * vis_expanded).sum(axis=0) / (visibility_mask.sum() + 1e-8)

        # 4. GRU belief update
        gru_input = jnp.concatenate([h_obs, context])  # [2 * embed_dim]
        # Project to hidden_dim for GRU input
        gru_input = nn.Dense(cfg.hidden_dim)(gru_input)
        belief = GRUCell(cfg.hidden_dim)(gru_input, prev_belief)

        # 5. Policy head
        logits = nn.Dense(cfg.n_actions)(belief)

        # 6. Auxiliary prediction head
        aux_h = nn.Dense(cfg.aux_hidden_dim)(belief)
        aux_h = nn.relu(aux_h)
        aux_logits = nn.Dense(cfg.n_actions * n_teammates)(aux_h)
        aux_logits = aux_logits.reshape(n_teammates, cfg.n_actions)

        return logits, belief, aux_logits


class Critic(nn.Module):
    """Centralized critic: global state → value."""
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class VABL:
    """VABL algorithm with JAX training."""

    def __init__(self, config: VABLConfig):
        self.config = config
        self.agent = VABLAgent(config)
        self.critic = Critic(config.critic_hidden_dim)

    def init(self, rng):
        """Initialize parameters."""
        cfg = self.config
        n_teammates = cfg.n_agents - 1
        rng_agent, rng_critic = jax.random.split(rng)

        # Dummy inputs for init
        dummy_obs = jnp.zeros(cfg.obs_dim)
        dummy_belief = jnp.zeros(cfg.hidden_dim)
        dummy_actions = jnp.zeros((n_teammates, cfg.n_actions))
        dummy_vis = jnp.ones(n_teammates)
        dummy_state = jnp.zeros(cfg.obs_dim * cfg.n_agents)

        agent_params = self.agent.init(rng_agent, dummy_obs, dummy_belief, dummy_actions, dummy_vis)
        critic_params = self.critic.init(rng_critic, dummy_state)

        agent_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.actor_lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.critic_lr, eps=1e-5),
        )

        agent_state = TrainState.create(apply_fn=self.agent.apply, params=agent_params, tx=agent_tx)
        critic_state = TrainState.create(apply_fn=self.critic.apply, params=critic_params, tx=critic_tx)

        return agent_state, critic_state

    def init_belief(self):
        """Initial belief state (zeros)."""
        return jnp.zeros((self.config.n_agents, self.config.hidden_dim))

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, agent_state, obs, beliefs, prev_actions, vis_masks, rng):
        """Select actions for all agents.

        Args:
            agent_state: TrainState with agent params
            obs: [n_agents, obs_dim]
            beliefs: [n_agents, hidden_dim]
            prev_actions: [n_agents] int actions from last step
            vis_masks: [n_agents, n_teammates]
            rng: PRNGKey

        Returns:
            actions: [n_agents] sampled actions
            new_beliefs: [n_agents, hidden_dim]
            log_probs: [n_agents]
        """
        cfg = self.config

        # Precompute static teammate indices
        _teammate_idx = jnp.array([
            [j for j in range(cfg.n_agents) if j != i]
            for i in range(cfg.n_agents)
        ])  # [n_agents, n_teammates]

        def agent_step(agent_idx, carry):
            rng_i = jax.random.fold_in(rng, agent_idx)
            obs_i = obs[agent_idx]
            belief_i = beliefs[agent_idx]

            # Build teammate action one-hots using static indices
            teammate_actions_int = prev_actions[_teammate_idx[agent_idx]]
            teammate_actions_oh = jax.nn.one_hot(teammate_actions_int, cfg.n_actions)
            vis_mask_i = vis_masks[agent_idx]

            logits, new_belief, aux_logits = agent_state.apply_fn(
                agent_state.params, obs_i, belief_i, teammate_actions_oh, vis_mask_i
            )

            # Sample action
            action = jax.random.categorical(rng_i, logits)
            log_prob = jax.nn.log_softmax(logits)[action]

            return action, new_belief, log_prob

        # Vectorize over agents
        agent_indices = jnp.arange(cfg.n_agents)
        actions, new_beliefs, log_probs = jax.vmap(
            lambda idx: agent_step(idx, None)
        )(agent_indices)

        return actions, new_beliefs, log_probs
