"""Generic Transformer MARL Agent with Auxiliary Prediction Head.

A simple, independent architecture that is NOT VABL and NOT AERIAL, used to
test whether the gradient-interference pathology is architectural-class
(attention + aux loss) rather than VABL-specific.

Key differences from VABL:
  - No GRU: the "belief" is a CLS token from a standard transformer encoder
  - Standard self-attention over all input tokens (obs + teammate actions),
    not cross-attention from a learned query
  - Positional encoding, not identity embeddings
  - Different parameterization entirely (transformer blocks vs GRU + MHA)

Key differences from AERIAL:
  - Uses observable actions as tokens, not shared hidden states
  - Has an explicit auxiliary prediction head (AERIAL doesn't)
  - No recurrence at all

Architecture:
  Input tokens: [CLS, obs_embed, teammate_action_1, ..., teammate_action_N-1]
  Encoder: 2-layer transformer with self-attention
  Output: CLS token → policy head + aux head
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


class TransformerConfig(NamedTuple):
    """Config for the generic transformer MARL agent."""
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: float = 2.0
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

    # Ablation flags (identical interface to VABLv2Config)
    use_attention: bool = True
    use_aux_loss: bool = True
    stop_gradient_belief_to_aux: bool = False
    aux_anneal_fraction: float = 0.0

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    use_attention: bool

    @nn.compact
    def __call__(self, x):
        if self.use_attention:
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                out_features=self.embed_dim,
            )(h, h)
            x = x + h
        else:
            # No-attention ablation: per-token MLP (same param count, no mixing)
            h = nn.LayerNorm()(x)
            h = nn.Dense(self.embed_dim)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.embed_dim)(h)
            x = x + h

        # FFN
        h = nn.LayerNorm()(x)
        hidden = int(self.embed_dim * self.mlp_ratio)
        h = nn.Dense(hidden)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.embed_dim)(h)
        x = x + h
        return x


class TransformerMARLAgent(nn.Module):
    """Generic transformer agent for cooperative MARL.

    Input: obs (per-agent) + teammate previous actions (one-hot)
    Output: policy logits + aux teammate-action prediction logits

    Forward pass:
      1. Embed obs → obs_token
      2. Embed each teammate action → action_tokens
      3. Prepend CLS token
      4. Add positional encoding
      5. Pass through N transformer blocks
      6. CLS token → policy head + aux head
    """
    config: TransformerConfig

    @nn.compact
    def __call__(self, obs, teammate_actions_oh, visibility_mask):
        """
        Args:
            obs: [obs_dim]
            teammate_actions_oh: [n_teammates, n_actions]
            visibility_mask: [n_teammates]
        Returns:
            logits: [n_actions]
            cls_token: [embed_dim] (the "belief")
            aux_logits: [n_teammates, n_actions]
        """
        cfg = self.config
        n_teammates = cfg.n_agents - 1
        d = cfg.embed_dim

        # 1. Embed observation
        obs_token = nn.Dense(d, name="obs_embed")(obs)  # [d]

        # 2. Embed teammate actions
        action_tokens = nn.Dense(d, name="action_embed")(
            teammate_actions_oh)  # [n_teammates, d]
        # Mask invisible teammates
        action_tokens = action_tokens * visibility_mask[:, None]

        # 3. CLS token
        cls = self.param("cls_token", nn.initializers.normal(stddev=0.02), (d,))

        # 4. Assemble sequence: [CLS, obs, action_1, ..., action_N-1]
        tokens = jnp.concatenate([
            cls[None, :],           # [1, d]
            obs_token[None, :],     # [1, d]
            action_tokens,          # [n_teammates, d]
        ], axis=0)  # [1 + 1 + n_teammates, d]

        seq_len = tokens.shape[0]
        pos_embed = self.param("pos_embed",
                               nn.initializers.normal(stddev=0.02),
                               (seq_len, d))
        tokens = tokens + pos_embed

        # 5. Transformer blocks
        for i in range(cfg.num_layers):
            tokens = TransformerBlock(
                embed_dim=d,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                use_attention=cfg.use_attention,
                name=f"block_{i}",
            )(tokens)

        tokens = nn.LayerNorm()(tokens)

        # 6. CLS token is the "belief"
        if cfg.use_attention:
            belief = tokens[0]  # CLS attended to everything
        else:
            belief = tokens.mean(axis=0)  # No attention → mean pool all tokens

        # 7. Policy head
        logits = nn.Dense(cfg.n_actions, name="policy_head")(belief)

        # 8. Aux head (with optional stop-gradient)
        belief_for_aux = jax.lax.stop_gradient(belief) if cfg.stop_gradient_belief_to_aux else belief
        aux_h = nn.Dense(cfg.aux_hidden_dim, name="aux_h1")(belief_for_aux)
        aux_h = nn.gelu(aux_h)
        aux_logits = nn.Dense(cfg.n_actions * n_teammates, name="aux_h2")(aux_h)
        aux_logits = aux_logits.reshape(n_teammates, cfg.n_actions)

        return logits, belief, aux_logits


class TransformerCritic(nn.Module):
    """Centralized critic (same as VABL's)."""
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)
