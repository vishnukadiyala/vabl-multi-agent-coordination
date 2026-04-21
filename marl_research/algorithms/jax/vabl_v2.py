"""VABL v2 — Fully aligned with PyTorch reference implementation.

Fixes from v1:
1. Identity embedding now correctly indexes by teammate global indices
2. Removed extra Dense projection in GRU input (matches PyTorch direct concat)
3. Multi-head attention with configurable heads (matches PyTorch's nn.MultiheadAttention)
4. Visibility mask actually used (passed in as argument, not hardcoded)
5. Aux loss uses next-step teammate actions
6. Orthogonal init with appropriate gains
7. Belief projection layer named for proper Flax param sharing

Still missing (low priority):
- Value normalization (PopArt)
- Target critic network
- KL early stopping
- Aux annealing / entropy decay schedules
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.algo_interface import RolloutBatch


class VABLv2Config(NamedTuple):
    embed_dim: int = 64
    hidden_dim: int = 128
    attention_heads: int = 4
    aux_hidden_dim: int = 64
    critic_hidden_dim: int = 128
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    aux_lambda: float = 0.05  # initial/constant aux loss weight
    grad_clip: float = 10.0
    use_attention: bool = True
    use_aux_loss: bool = True  # honored: if False, aux loss term is zeroed regardless of aux_lambda
    use_identity_encoding: bool = True
    # Camera-ready fix-path knobs (added 2026-04-09 after post-fix 10M ablation
    # revealed aux+attention interaction). See wiki/concepts/aux_loss_bug.md.
    stop_gradient_belief_to_aux: bool = False  # if True, aux head only trains itself, not the belief encoder
    aux_anneal_fraction: float = 0.0  # 0.0 = constant lambda; 0.X = anneal lambda to 0 over first X fraction of training
    # Intra-actor control (added 2026-04-20): when True, the auxiliary predictor
    # uses its OWN parallel encoder pipeline (phi_aux, psi_aux, belief_aux_dense).
    # The policy's belief encoder is entirely shielded from auxiliary gradients.
    # Tests whether the pathology is specifically due to the SHARED encoder, as
    # opposed to attention-plus-aux being harmful anywhere.
    separate_aux_encoder: bool = False
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520


def orthogonal_init(scale=1.414):
    return nn.initializers.orthogonal(scale=scale)


class FeatureEncoder(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal_init(1.414))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal_init(1.414))(x)
        x = nn.relu(x)
        return x


class ActionEncoder(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, action_onehot):
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal_init(1.414))(action_onehot)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal_init(1.414))(x)
        x = nn.relu(x)
        return x


class VABLv2Agent(nn.Module):
    """VABL agent network — properly aligned with PyTorch reference.

    Forward:
      obs (per-agent), prev_belief, teammate_actions, teammate_indices, vis_mask
        → logits, new_belief, aux_logits
    """
    config: VABLv2Config

    @nn.compact
    def __call__(self, obs, prev_belief, teammate_actions_oh, teammate_indices, visibility_mask):
        """
        Args:
            obs: [obs_dim]
            prev_belief: [hidden_dim]
            teammate_actions_oh: [n_teammates, n_actions]
            teammate_indices: [n_teammates] int — global indices of teammates
            visibility_mask: [n_teammates] binary
        Returns:
            logits: [n_actions]
            new_belief: [hidden_dim]
            aux_logits: [n_teammates, n_actions]
        """
        cfg = self.config
        n_teammates = cfg.n_agents - 1

        # 1. Encode observation
        h_obs = FeatureEncoder(cfg.embed_dim, name="phi_net")(obs)

        # 2. Encode teammate actions (one-hot → embed_dim)
        action_encoder = ActionEncoder(cfg.embed_dim, name="psi_net")
        h_actions = jax.vmap(action_encoder)(teammate_actions_oh)  # [n_teammates, embed_dim]

        # 3. Add identity embeddings indexed by ACTUAL teammate indices
        if cfg.use_identity_encoding:
            identity_emb = self.param(
                "identity_emb",
                nn.initializers.normal(stddev=0.01),
                (cfg.n_agents, cfg.embed_dim),
            )
            # Index by teammate_indices, NOT hardcoded slice
            h_actions = h_actions + identity_emb[teammate_indices]

        # 4. Apply visibility mask to embeddings (zero out invisible)
        # Match PyTorch: mask multiplies embeddings BEFORE attention
        vis_expanded = visibility_mask[:, None]  # [n_teammates, 1]
        masked_h_actions = h_actions * vis_expanded

        # 5. Context: attention or mean pooling
        if cfg.use_attention:
            # Multi-head attention with belief query
            belief_proj = nn.Dense(cfg.embed_dim, name="belief_proj",
                                    kernel_init=orthogonal_init(1.414))(prev_belief)
            query = belief_proj[None, :]  # [1, embed_dim]

            # Use Flax's MultiHeadDotProductAttention
            mha = nn.MultiHeadDotProductAttention(
                num_heads=cfg.attention_heads,
                qkv_features=cfg.embed_dim,
                out_features=cfg.embed_dim,
                kernel_init=orthogonal_init(1.414),
                name="mha",
            )

            # All inputs must have same rank: [seq_len, embed_dim]
            # query: [1, embed_dim], k/v: [n_teammates, embed_dim]
            # Mask shape: [1, n_teammates] for [query_len=1, key_len=n_teammates]
            attn_mask = visibility_mask[None, :].astype(jnp.bool_)  # [1, n_teammates]
            any_visible = visibility_mask.sum() > 0
            attn_mask = jnp.where(any_visible, attn_mask, attn_mask.at[0, 0].set(True))

            context = mha(
                inputs_q=query,                # [1, embed_dim]
                inputs_k=masked_h_actions,     # [n_teammates, embed_dim]
                inputs_v=masked_h_actions,
                mask=attn_mask,                # [1, n_teammates]
            ).squeeze(0)  # [embed_dim]
        else:
            # Mean pooling over visible teammates
            n_visible = visibility_mask.sum() + 1e-8
            context = masked_h_actions.sum(axis=0) / n_visible

        # 6. GRU belief update — DIRECT concat to GRU, no extra projection
        gru_input = jnp.concatenate([h_obs, context])  # [2 * embed_dim]
        gru_cell = nn.GRUCell(features=cfg.hidden_dim, name="gru",
                               kernel_init=orthogonal_init(1.0),
                               recurrent_kernel_init=orthogonal_init(1.0))
        new_belief, _ = gru_cell(prev_belief, gru_input)

        # 7. Policy head — small init for exploration
        logits = nn.Dense(cfg.n_actions, name="policy_head",
                           kernel_init=orthogonal_init(0.01))(new_belief)

        # 8. Aux head.
        #
        # Three mutually-exclusive variants (in priority order):
        #   (i)  separate_aux_encoder: aux has its OWN parallel encoder
        #        pipeline (phi_aux, psi_aux, pooling, belief projection). The
        #        policy's belief encoder is entirely shielded from the
        #        auxiliary gradient.
        #   (ii) stop_gradient_belief_to_aux: aux reads from the shared belief
        #        but gradients from aux_loss are blocked before they reach the
        #        belief encoder.
        #   (iii) default: aux reads from the shared belief with full gradient
        #        flow back into the encoder pipeline (the configuration the
        #        pathology manifests in).
        if cfg.separate_aux_encoder:
            # Independent feature encoders. Params are NOT shared with phi_net
            # or psi_net used by the policy.
            aux_h_obs = FeatureEncoder(cfg.embed_dim, name="phi_aux")(obs)
            aux_action_encoder = ActionEncoder(cfg.embed_dim, name="psi_aux")
            aux_h_actions = jax.vmap(aux_action_encoder)(teammate_actions_oh)
            # Masked mean pooling (teammate actions in the aux pipeline are
            # aggregated without the pathological attention path; the control
            # tests the encoder sharing, not attention on the aux side).
            aux_masked = aux_h_actions * vis_expanded
            aux_n_visible = visibility_mask.sum() + 1e-8
            aux_context = aux_masked.sum(axis=0) / aux_n_visible
            # Project to a belief-dim representation for the aux head. Note:
            # no recurrence here; the aux predictor sees only current-step
            # features, which isolates the "shared encoder" question from
            # belief propagation. GRU-for-aux could be added if needed.
            aux_belief = nn.Dense(cfg.hidden_dim, name="belief_aux_proj",
                                  kernel_init=orthogonal_init(1.414))(
                jnp.concatenate([aux_h_obs, aux_context]))
            aux_belief = nn.relu(aux_belief)
            belief_for_aux = aux_belief
        elif cfg.stop_gradient_belief_to_aux:
            belief_for_aux = jax.lax.stop_gradient(new_belief)
        else:
            belief_for_aux = new_belief

        aux_h = nn.Dense(cfg.aux_hidden_dim, name="aux_h1",
                          kernel_init=orthogonal_init(1.414))(belief_for_aux)
        aux_h = nn.relu(aux_h)
        aux_logits_flat = nn.Dense(cfg.n_actions * n_teammates, name="aux_h2",
                                    kernel_init=orthogonal_init(0.01))(aux_h)
        aux_logits = aux_logits_flat.reshape(n_teammates, cfg.n_actions)

        return logits, new_belief, aux_logits


class VABLv2Critic(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal_init(1.414))(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal_init(1.414))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal_init(1.0))(x).squeeze(-1)


class VABLv2Impl:
    """VABL v2 implementing the JaxMARLAlgo interface."""

    def __init__(self, config: VABLv2Config):
        self.config = config
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.obs_dim = config.obs_dim
        self.state_dim = config.obs_dim * config.n_agents
        self.n_teammates = config.n_agents - 1

        self.agent_net = VABLv2Agent(config)
        self.critic_net = VABLv2Critic(config.critic_hidden_dim)

        # Per-agent teammate indices [n_agents, n_teammates]
        self.teammate_idx = jnp.array([
            [j for j in range(config.n_agents) if j != i]
            for i in range(config.n_agents)
        ], dtype=jnp.int32)

    def init(self, rng):
        cfg = self.config
        rng_a, rng_c = jax.random.split(rng)

        # Init with agent 0's teammate indices as the dummy
        dummy_teammate_idx = self.teammate_idx[0]
        agent_params = self.agent_net.init(
            rng_a,
            jnp.zeros(self.obs_dim),
            jnp.zeros(cfg.hidden_dim),
            jnp.zeros((self.n_teammates, self.n_actions)),
            dummy_teammate_idx,
            jnp.ones(self.n_teammates),
        )
        critic_params = self.critic_net.init(rng_c, jnp.zeros(self.state_dim))

        agent_state = TrainState.create(
            apply_fn=self.agent_net.apply, params=agent_params,
            tx=optax.chain(
                optax.clip_by_global_norm(cfg.grad_clip),
                optax.adam(cfg.actor_lr, eps=1e-5),
            ),
        )
        critic_state = TrainState.create(
            apply_fn=self.critic_net.apply, params=critic_params,
            tx=optax.chain(
                optax.clip_by_global_norm(cfg.grad_clip),
                optax.adam(cfg.critic_lr, eps=1e-5),
            ),
        )
        return agent_state, critic_state

    def init_carry(self, n_envs):
        return jnp.zeros((n_envs, self.n_agents, self.config.hidden_dim))

    def step(self, agent_params, obs, carry, prev_actions, rng):
        """Forward pass for all agents in N parallel envs."""
        n_envs = obs.shape[0]

        def per_env(env_idx, env_obs, env_carry, env_prev):
            rng_env = jax.random.fold_in(rng, env_idx)

            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                # Get this agent's teammate indices
                t_idx = self.teammate_idx[i]
                # Get teammate actions
                t_acts = env_prev[t_idx]
                t_oh = jax.nn.one_hot(t_acts, self.n_actions)
                # All teammates visible during rollout
                vis = jnp.ones(self.n_teammates)

                logits, new_b, _ = self.agent_net.apply(
                    agent_params, env_obs[i], env_carry[i], t_oh, t_idx, vis
                )
                action = jax.random.categorical(rng_i, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, new_b, lp

            return jax.vmap(per_agent)(jnp.arange(self.n_agents))

        return jax.vmap(per_env)(
            jnp.arange(n_envs), obs, carry, prev_actions
        )

    def get_value(self, critic_params, state):
        return jax.vmap(lambda s: self.critic_net.apply(critic_params, s))(state)

    def actor_loss(self, params, batch, clip_param, entropy_coef, aux_lambda_eff=None):
        """Actor loss with configurable aux_lambda.

        Args:
            aux_lambda_eff: effective aux lambda to use (for annealing schedules).
                If None, falls back to self.config.aux_lambda. If config.use_aux_loss
                is False, aux loss is zeroed regardless of this value.
        """
        B = batch.obs.shape[0]
        n_a = self.n_agents
        n_t = self.n_teammates
        n_act = self.n_actions
        if aux_lambda_eff is None:
            aux_lambda_eff = self.config.aux_lambda

        # Build flat (B*n_agents) inputs
        flat_obs = batch.obs.reshape(B * n_a, self.obs_dim)
        flat_carry = batch.carry.reshape(B * n_a, self.config.hidden_dim)

        # For each (b, i): teammate actions = batch.actions[b, teammate_idx[i]]
        t_acts = batch.actions[:, self.teammate_idx]  # [B, n_agents, n_teammates]
        flat_t_oh = jax.nn.one_hot(t_acts, n_act).reshape(B * n_a, n_t, n_act)

        # teammate_idx repeated for each (b, i)
        flat_t_idx = jnp.tile(self.teammate_idx[None, :, :], (B, 1, 1)).reshape(B * n_a, n_t)

        flat_vis = jnp.ones((B * n_a, n_t))

        def forward_one(o, c, t_oh, t_idx, vis):
            logits, _, aux_logits = self.agent_net.apply(params, o, c, t_oh, t_idx, vis)
            return logits, aux_logits

        flat_logits, flat_aux = jax.vmap(forward_one)(
            flat_obs, flat_carry, flat_t_oh, flat_t_idx, flat_vis
        )

        logits = flat_logits.reshape(B, n_a, n_act)

        lp = jax.nn.log_softmax(logits)
        nlp = jnp.take_along_axis(lp, batch.actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
        old_lp_sum = batch.log_probs.sum(axis=-1)

        ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
        s1 = ratio * batch.advantages
        s2 = jnp.clip(ratio, 1 - clip_param, 1 + clip_param) * batch.advantages
        p_loss = -jnp.minimum(s1, s2).mean()

        pr = jax.nn.softmax(logits)
        ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
        e_loss = -ent.mean()

        # Aux loss: predict NEXT-step teammate actions from current belief.
        # Gated by use_aux_loss (hard on/off) and scaled by aux_lambda_eff
        # (the effective lambda for this training iteration, possibly annealed).
        aux_logits = flat_aux.reshape(B, n_a, n_t, n_act)
        aux_lp = jax.nn.log_softmax(aux_logits)
        aux_targets = batch.next_actions[:, self.teammate_idx]  # [B, n_agents, n_teammates]
        aux_taken = jnp.take_along_axis(aux_lp, aux_targets[..., None], axis=-1).squeeze(-1)
        aux_loss = -aux_taken.mean()

        aux_term = aux_lambda_eff * aux_loss if self.config.use_aux_loss else jnp.zeros_like(aux_loss)
        return p_loss + entropy_coef * e_loss + aux_term

    def critic_loss(self, params, batch):
        vals = jax.vmap(lambda s: self.critic_net.apply(params, s))(batch.states)
        return ((vals - batch.returns) ** 2).mean()
