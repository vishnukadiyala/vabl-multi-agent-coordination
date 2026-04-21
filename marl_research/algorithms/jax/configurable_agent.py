"""Configurable MARL Agent for architecture sweep.

Swappable components to isolate which architectural choices cause
the gradient-interference pathology:

  Recurrence:  {GRU, LSTM, none}
  Attention:   {cross_attn, self_attn, additive, mean_pool}
  Aux loss:    {on, off}

Each combination uses the same training loop (train_vabl_vec_fast.py
pattern) so the only variable is the architecture.

Configurations tested:
  1. LSTM + cross_attn (+ aux / - aux)     — is recurrence type the factor?
  2. GRU + additive_attn (+ aux / - aux)   — is attention type the factor?
  3. GRU + self_attn (+ aux / - aux)       — cross-attn vs self-attn?
  4. none + cross_attn (+ aux / - aux)     — is recurrence required?
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class ConfigurableConfig(NamedTuple):
    """Config with swappable architecture components."""
    # Architecture choices
    recurrence: str = "gru"        # "gru", "lstm", "none"
    attention: str = "cross_attn"  # "cross_attn", "self_attn", "additive", "mean_pool"

    # Dimensions
    embed_dim: int = 64
    hidden_dim: int = 128
    attention_heads: int = 4
    aux_hidden_dim: int = 64
    critic_hidden_dim: int = 128

    # Training (same as VABL for fair comparison)
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    aux_lambda: float = 0.05
    grad_clip: float = 10.0

    # Ablation flags
    use_aux_loss: bool = True
    stop_gradient_belief_to_aux: bool = False
    aux_anneal_fraction: float = 0.0

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520


class ConfigurableAgent(nn.Module):
    """Agent with swappable recurrence and attention.

    Forward: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
           -> (logits, new_belief, aux_logits)
    """
    config: ConfigurableConfig

    @nn.compact
    def __call__(self, obs, prev_belief, teammate_actions_oh, teammate_indices,
                 visibility_mask):
        cfg = self.config
        n_teammates = cfg.n_agents - 1
        d = cfg.embed_dim

        # 1. Encode observation
        h_obs = nn.Dense(d, name="obs_enc_1")(obs)
        h_obs = nn.relu(h_obs)
        h_obs = nn.Dense(d, name="obs_enc_2")(h_obs)
        h_obs = nn.relu(h_obs)

        # 2. Encode teammate actions
        h_actions = nn.Dense(d, name="act_enc_1")(teammate_actions_oh)
        h_actions = nn.relu(h_actions)
        h_actions = nn.Dense(d, name="act_enc_2")(h_actions)
        h_actions = nn.relu(h_actions)  # [n_teammates, d]

        # Add identity embeddings
        identity_emb = self.param(
            "identity_emb",
            nn.initializers.normal(stddev=0.01),
            (cfg.n_agents, d),
        )
        h_actions = h_actions + identity_emb[teammate_indices]

        # Mask invisible teammates
        vis_expanded = visibility_mask[:, None]
        masked_h_actions = h_actions * vis_expanded

        # 3. Compute social context based on attention type
        if cfg.attention == "cross_attn":
            # VABL-style: belief queries, action keys/values
            belief_proj = nn.Dense(d, name="belief_proj")(prev_belief)
            query = belief_proj[None, :]
            mha = nn.MultiHeadDotProductAttention(
                num_heads=cfg.attention_heads,
                qkv_features=d,
                out_features=d,
                name="cross_mha",
            )
            attn_mask = visibility_mask[None, :].astype(jnp.bool_)
            any_visible = visibility_mask.sum() > 0
            attn_mask = jnp.where(any_visible, attn_mask, attn_mask.at[0, 0].set(True))
            context = mha(
                inputs_q=query,
                inputs_k=masked_h_actions,
                inputs_v=masked_h_actions,
                mask=attn_mask,
            ).squeeze(0)

        elif cfg.attention == "self_attn":
            # Self-attention over [obs_embed, action_1, ..., action_N]
            tokens = jnp.concatenate([h_obs[None, :], masked_h_actions], axis=0)
            mha = nn.MultiHeadDotProductAttention(
                num_heads=cfg.attention_heads,
                qkv_features=d,
                out_features=d,
                name="self_mha",
            )
            attended = mha(inputs_q=tokens, inputs_k=tokens, inputs_v=tokens)
            context = attended[0]  # first token (obs position)

        elif cfg.attention == "additive":
            # Bahdanau-style additive attention
            belief_proj = nn.Dense(d, name="belief_proj_add")(prev_belief)
            action_proj = nn.Dense(d, name="action_proj_add")(masked_h_actions)
            combined = jnp.tanh(belief_proj[None, :] + action_proj)  # [n_teammates, d]
            scores = nn.Dense(1, name="score_proj")(combined).squeeze(-1)  # [n_teammates]
            attn_mask = jnp.where(visibility_mask > 0, 0.0, -1e8)
            weights = jax.nn.softmax(scores + attn_mask)
            context = jnp.dot(weights, masked_h_actions)

        elif cfg.attention == "mean_pool":
            n_visible = visibility_mask.sum() + 1e-8
            context = masked_h_actions.sum(axis=0) / n_visible

        else:
            raise ValueError(f"Unknown attention type: {cfg.attention}")

        # 4. Belief update based on recurrence type
        gru_input = jnp.concatenate([h_obs, context])

        if cfg.recurrence == "gru":
            gru_input_proj = nn.Dense(cfg.hidden_dim, name="gru_proj")(gru_input)
            gru = nn.GRUCell(features=cfg.hidden_dim, name="gru_cell")
            new_belief, _ = gru(prev_belief, gru_input_proj)

        elif cfg.recurrence == "lstm":
            # LSTM: split prev_belief into h and c
            half = cfg.hidden_dim // 2
            lstm_h = prev_belief[:half]
            lstm_c = prev_belief[half:]
            lstm_input_proj = nn.Dense(half, name="lstm_proj")(gru_input)
            lstm = nn.OptimizedLSTMCell(features=half, name="lstm_cell")
            new_carry, _ = lstm((lstm_h, lstm_c), lstm_input_proj)
            new_belief = jnp.concatenate([new_carry[0], new_carry[1]])

        elif cfg.recurrence == "none":
            # No recurrence: project input directly to belief-sized vector
            new_belief = nn.Dense(cfg.hidden_dim, name="ff_proj")(gru_input)
            new_belief = nn.relu(new_belief)
            new_belief = nn.Dense(cfg.hidden_dim, name="ff_proj2")(new_belief)

        else:
            raise ValueError(f"Unknown recurrence type: {cfg.recurrence}")

        # 5. Policy head
        logits = nn.Dense(cfg.n_actions, name="policy_head")(new_belief)

        # 6. Aux head
        belief_for_aux = (jax.lax.stop_gradient(new_belief)
                          if cfg.stop_gradient_belief_to_aux else new_belief)
        aux_h = nn.Dense(cfg.aux_hidden_dim, name="aux_h1")(belief_for_aux)
        aux_h = nn.relu(aux_h)
        aux_logits = nn.Dense(cfg.n_actions * n_teammates, name="aux_h2")(aux_h)
        aux_logits = aux_logits.reshape(n_teammates, cfg.n_actions)

        return logits, new_belief, aux_logits


class ConfigurableCritic(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class ConfigurableCriticWithAux(nn.Module):
    """Centralized critic with auxiliary teammate-prediction head.

    MAAC-style experiment: tests whether the gradient-interference pathology
    appears when the aux loss is on the CRITIC (shared value+aux parameters)
    rather than the ACTOR (shared belief+aux parameters).

    Forward: (state) -> (value, aux_logits)
      state: concatenated observations of all agents [obs_dim * n_agents]
      value: scalar value estimate
      aux_logits: [n_teammates, n_actions] teammate action predictions
    """
    hidden_dim: int
    n_actions: int
    n_teammates: int

    @nn.compact
    def __call__(self, state):
        # Shared trunk
        x = nn.Dense(self.hidden_dim, name="trunk_1")(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="trunk_2")(x)
        x = nn.relu(x)

        # Value head
        value = nn.Dense(1, name="value_head")(x).squeeze(-1)

        # Aux head (predicts teammate actions from critic features)
        aux_h = nn.Dense(self.hidden_dim // 2, name="aux_h1")(x)
        aux_h = nn.relu(aux_h)
        aux_logits = nn.Dense(
            self.n_actions * self.n_teammates, name="aux_h2")(aux_h)
        aux_logits = aux_logits.reshape(self.n_teammates, self.n_actions)

        return value, aux_logits
