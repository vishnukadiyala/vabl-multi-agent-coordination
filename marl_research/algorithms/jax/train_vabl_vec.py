"""Vectorized VABL JAX training — N parallel envs via jax.vmap.

Canonical camera-ready entry point (2026-04-09). Uses `VABLv2Agent` from
`vabl_v2.py`. The v1 `VABLAgent` from `vabl.py` is deprecated: v2 fixes bugs in
identity-embedding indexing, visibility-mask application, and adds multi-head
attention and orthogonal init. See wiki/concepts/aux_loss_bug.md for the
rationale.

This runner supports aux-loss knobs exposed by VABLv2Config:
  * use_aux_loss (bool): hard on/off switch for the aux loss term.
  * aux_lambda (float): the initial/constant aux loss weight.
  * stop_gradient_belief_to_aux (bool): when True, aux gradients never reach
    the belief encoder.
  * aux_anneal_fraction (float): 0.0 means constant lambda; values in (0, 1]
    linearly decay aux_lambda → 0 over the first `aux_anneal_fraction` of
    training iterations, held at 0 afterwards.
  * aux_frozen_target_policy (runtime flag): when True, aux targets come from
    a snapshot of the initial agent policy instead of the current teammates.
    This removes co-learning non-stationarity from aux targets while
    preserving aux capacity cost and the aux-to-encoder gradient pathway.
    Used to distinguish the directional-interference hypothesis from a
    generic aux-capacity-cost hypothesis.

Usage:
    python -m marl_research.algorithms.jax.train_vabl_vec --layout cramped_room --episodes 5000 --n-envs 32
"""

import time
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.vabl_v2 import (
    VABLv2Config as VABLConfig,
    VABLv2Agent as VABLAgent,
    VABLv2Critic as Critic,
    orthogonal_init,
)
import flax.linen as nn


class VariantAuxHead(nn.Module):
    """Small MLP head for the Q4 auxiliary-task variants (latent / recon).

    Separate parameter tree from the agent so the default action-prediction
    path is untouched; gradients still reach the agent's belief encoder
    because the head consumes the (non-detached) belief."""
    out_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, belief):
        x = nn.Dense(self.hidden, kernel_init=orthogonal_init(1.414))(belief)
        x = nn.relu(x)
        return nn.Dense(self.out_dim, kernel_init=orthogonal_init(0.01))(x)


def _import_overcooked():
    import sys, types
    if "jaxmarl.environments.mabrax" not in sys.modules:
        fake = types.ModuleType("jaxmarl.environments.mabrax")
        fake.Ant = fake.Humanoid = fake.Hopper = fake.Walker2d = fake.HalfCheetah = None
        sys.modules["jaxmarl.environments.mabrax"] = fake
    from jaxmarl.environments.overcooked.overcooked import Overcooked, layouts
    return Overcooked, layouts


LAYOUT_MAP = {
    "cramped_room": "cramped_room",
    "asymmetric_advantages": "asymm_advantages",
    "coordination_ring": "coord_ring",
    "forced_coordination": "forced_coord",
    "counter_circuit": "counter_circuit",
}


def train_vabl_vec(
    config: VABLConfig = None,
    layout: str = "cramped_room",
    n_episodes: int = 5000,
    horizon: int = 400,
    n_envs: int = 32,
    seed: int = 0,
    log_interval: int = 10,
    save_path: str = None,
    aux_frozen_target_policy: bool = False,
    aux_noise_targets: bool = False,
    log_gradient_decomp: bool = False,
    grad_log_interval: int = 25,
    log_policy_kl: bool = False,
    aux_snapshot_refresh: int = 0,
    aux_soft_targets: bool = False,
    grad_surgery: str = "none",
    drift_gate_tau: float = 0.0,
    drift_gate_window: int = 10,
    log_jpi: bool = False,
    aux_task: str = "action",
    aux_schedule: str = "constant",
    drift_gate_random: float = 0.0,
    aux_ema_alpha: float = 0.0,
):
    if config is None:
        config = VABLConfig()

    # Rebuttal round 2 (2026-07-26): Q3 J_pi estimator, Q4 aux-task variants,
    # Q8 schedules, Q6 random-gate control. Same isolation rule as before: the
    # default path (all new flags off) is unchanged, including its RNG stream.
    if aux_task not in ("action", "latent", "recon"):
        raise ValueError(f"aux_task must be action|latent|recon, got {aux_task}")
    if aux_schedule not in ("constant", "cosine", "exp", "kl_adaptive"):
        raise ValueError(f"aux_schedule must be constant|cosine|exp|kl_adaptive, got {aux_schedule}")
    if aux_task != "action":
        assert config.use_aux_loss and grad_surgery == "none" \
            and not log_gradient_decomp and not log_jpi \
            and drift_gate_tau == 0.0 and drift_gate_random == 0.0 \
            and not aux_frozen_target_policy and not aux_noise_targets \
            and aux_snapshot_refresh == 0, \
            "aux_task variants only support the plain training path (+ policy-KL logging)"
    if aux_schedule == "kl_adaptive":
        log_policy_kl = True  # the controller consumes the measurement
    if drift_gate_random > 0.0:
        assert drift_gate_tau == 0.0 and config.use_aux_loss \
            and not config.stop_gradient_belief_to_aux
    if log_jpi:
        assert log_gradient_decomp, "--log-jpi rides on the gradient-decomp logging points"
    if aux_ema_alpha > 0.0:
        # EMA-distilled auxiliary targets: a target-DESIGN intervention (root
        # cause) rather than a gradient-path one. Predict soft actions of an
        # exponential-moving-average copy of the policy instead of the live
        # co-learning policy; drift is slowed at the target source by design.
        assert 0.0 < aux_ema_alpha < 1.0
        assert aux_snapshot_refresh == 0 and not aux_frozen_target_policy \
            and not aux_noise_targets and aux_task == "action" \
            and grad_surgery == "none" and drift_gate_tau == 0.0 \
            and drift_gate_random == 0.0
        aux_soft_targets = True

    # Rebuttal instrumentation (2026-07-24). Modes are mutually exclusive with
    # each other and with the frozen/noise target experiments; the default path
    # (all flags off) is unchanged, including its RNG stream.
    if grad_surgery not in ("none", "pcgrad", "gradnorm"):
        raise ValueError(f"grad_surgery must be none|pcgrad|gradnorm, got {grad_surgery}")
    if grad_surgery != "none":
        assert config.use_aux_loss, "grad_surgery requires the aux loss to be on"
        assert aux_snapshot_refresh == 0 and drift_gate_tau == 0.0
    if drift_gate_tau > 0.0:
        assert config.use_aux_loss and not config.stop_gradient_belief_to_aux
        assert aux_snapshot_refresh == 0
    if aux_snapshot_refresh != 0:
        assert not aux_frozen_target_policy and not aux_noise_targets

    Overcooked, available_layouts = _import_overcooked()
    jax_layout = LAYOUT_MAP.get(layout, layout)
    env = Overcooked(layout=available_layouts[jax_layout])

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    test_obs, _ = env.reset(reset_rng)
    agent_names = sorted(env.agents)
    obs_dim = int(np.prod(test_obs[agent_names[0]].shape))
    n_agents = len(agent_names)
    n_actions = env.action_space(agent_names[0]).n
    n_teammates = n_agents - 1

    config = config._replace(obs_dim=obs_dim, n_agents=n_agents, n_actions=n_actions)

    print(f"VABL JAX Vectorized Training on {layout}")
    print(f"  obs={obs_dim}, agents={n_agents}, actions={n_actions}")
    print(f"  N envs={n_envs}, horizon={horizon}, episodes={n_episodes}, seed={seed}")
    print(f"  Total iterations needed: {n_episodes // n_envs}")

    # Init networks
    agent_net = VABLAgent(config)
    critic_net = Critic(config.critic_hidden_dim)

    rng, rng_a, rng_c, rng_v = jax.random.split(rng, 4)
    # v2 agent signature: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
    dummy_t_idx = jnp.arange(1, n_agents, dtype=jnp.int32)  # teammate indices for agent 0
    # When use_vae_belief=True the agent calls self.make_rng('vae') inside the
    # forward; Flax needs a 'vae' rng stream at init time even if the VAE
    # branch is inactive, for parameter-shape determination.
    init_rngs = {"params": rng_a, "vae": rng_v}
    agent_params = agent_net.init(init_rngs, jnp.zeros(obs_dim), jnp.zeros(config.hidden_dim),
                                   jnp.zeros((n_teammates, n_actions)), dummy_t_idx, jnp.ones(n_teammates))
    critic_params = critic_net.init(rng_c, jnp.zeros(obs_dim * n_agents))

    agent_state = TrainState.create(
        apply_fn=agent_net.apply, params=agent_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.actor_lr, eps=1e-5)))
    critic_state = TrainState.create(
        apply_fn=critic_net.apply, params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.critic_lr, eps=1e-5)))

    teammate_idx = jnp.array([[j for j in range(n_agents) if j != i] for i in range(n_agents)])

    # JIT vectorized env: vmap over N envs
    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    # ---- JIT functions ----

    @jax.jit
    def step_agents_vec(params, obs_batch, beliefs_batch, prev_actions_batch, rng):
        """Run agents for N envs in parallel.
        obs_batch: [N, n_agents, obs_dim]
        beliefs_batch: [N, n_agents, hidden_dim]
        prev_actions_batch: [N, n_agents]
        Returns: actions [N, n_agents], beliefs [N, n_agents, hidden_dim], log_probs [N, n_agents]
        """
        def per_env(env_idx, env_obs, env_beliefs, env_prev_acts):
            rng_env = jax.random.fold_in(rng, env_idx)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                rng_i_act, rng_i_vae = jax.random.split(rng_i)
                t_idx = teammate_idx[i]
                t_oh = jax.nn.one_hot(env_prev_acts[t_idx], n_actions)
                # v2 signature: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
                logits, new_b, _, _ = agent_net.apply(
                    params, env_obs[i], env_beliefs[i], t_oh, t_idx, jnp.ones(n_teammates),
                    rngs={"vae": rng_i_vae})
                action = jax.random.categorical(rng_i_act, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, new_b, lp
            return jax.vmap(per_agent)(jnp.arange(n_agents))

        env_indices = jnp.arange(n_envs)
        return jax.vmap(per_env)(env_indices, obs_batch, beliefs_batch, prev_actions_batch)

    @jax.jit
    def get_value_vec(critic_params, states_batch):
        """states_batch: [N, state_dim]"""
        return jax.vmap(lambda s: critic_net.apply(critic_params, s))(states_batch)

    @jax.jit
    def compute_gae_vec(rewards, values, dones):
        """All shape [N, H]. Returns advantages and returns of same shape."""
        def per_env(rew, val, dn):
            def body(gae, t):
                idx = horizon - 1 - t
                next_val = jnp.where(idx + 1 < horizon, val[jnp.minimum(idx + 1, horizon - 1)], 0.0)
                delta = rew[idx] + config.gamma * next_val * (1 - dn[idx]) - val[idx]
                gae = delta + config.gamma * config.gae_lambda * (1 - dn[idx]) * gae
                return gae, gae
            _, adv_rev = jax.lax.scan(body, 0.0, jnp.arange(horizon))
            return jnp.flip(adv_rev)
        advantages = jax.vmap(per_env)(rewards, values, dones)
        returns = advantages + values
        # Normalize advantages globally across all N*H samples
        adv_mean = advantages.mean()
        adv_std = jnp.maximum(advantages.std(), 1e-8)
        advantages = (advantages - adv_mean) / adv_std
        return advantages, returns

    @jax.jit
    def compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng):
        """Returns (logits [B*n_agents, n_actions], aux_logits [B*n_agents, n_teammates, n_actions],
        kl_total scalar — sum of per-sample KL terms; zero when use_vae_belief=False).
        vae_rng is the per-batch VAE rng; split per-sample via fold_in for determinism.
        """
        def forward_one(idx, obs_i, belief_i, t_oh_i, t_idx_i):
            # v2 signature: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
            rng_i = jax.random.fold_in(vae_rng, idx)
            logits, _, aux, kl = agent_net.apply(
                params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates),
                rngs={"vae": rng_i})
            return logits, aux, kl
        idx = jnp.arange(flat_obs.shape[0])
        logits, aux, kls = jax.vmap(forward_one)(idx, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
        return logits, aux, kls.sum()

    @jax.jit
    def compute_separate_gradients(
        params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
        flat_actions, flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff, vae_rng,
    ):
        """Compute policy-only and aux-only gradients separately for the diagnostic.

        Matches the terms inside `actor_update.loss_fn` but splits the PPO+entropy
        loss from the aux loss so we can measure norms and cosine between the two
        gradient directions in the full param tree. VAE KL is not included in
        either split (it's added at the combined-loss level by actor_update).
        """
        def policy_only_loss(p):
            flat_logits, _, _ = compute_logits_and_aux(
                p, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
            B = flat_actions.shape[0]
            logits = flat_logits.reshape(B, n_agents, n_actions)
            lp = jax.nn.log_softmax(logits)
            nlp = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
            ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
            s1 = ratio * advantages_flat
            s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
            p_loss = -jnp.minimum(s1, s2).mean()
            pr = jax.nn.softmax(logits)
            ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
            e_loss = -ent.mean()
            return p_loss + config.entropy_coef * e_loss

        def aux_only_loss(p):
            _, flat_aux, _ = compute_logits_and_aux(
                p, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
            aux_lp = jax.nn.log_softmax(flat_aux, axis=-1)
            aux_taken = jnp.take_along_axis(
                aux_lp, flat_next_t_actions[..., None], axis=-1
            ).squeeze(-1)
            aux_loss = -aux_taken.mean()
            return aux_lambda_eff * aux_loss

        g_policy = jax.grad(policy_only_loss)(params)
        g_aux = jax.grad(aux_only_loss)(params)

        # Flatten both gradient trees to single vectors for inner products.
        flat_gp, _ = jax.flatten_util.ravel_pytree(g_policy)
        flat_ga, _ = jax.flatten_util.ravel_pytree(g_aux)

        norm_p = jnp.linalg.norm(flat_gp)
        norm_a = jnp.linalg.norm(flat_ga)
        dot = jnp.sum(flat_gp * flat_ga)
        cos = dot / (norm_p * norm_a + 1e-12)
        # Also return the flattened task gradients so the host can track each
        # gradient's OWN temporal direction stability (self-cosine between
        # consecutive logged iterations). This separates "the aux gradient
        # direction drifts" (the mechanism's signature) from "the policy
        # gradient direction drifts" (present in every condition).
        return norm_p, norm_a, cos, flat_gp, flat_ga

    @jax.jit
    def frozen_policy_argmax_actions(frozen_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng):
        """Run a FROZEN policy snapshot on rollout obs/beliefs and return deterministic
        argmax actions for each agent. Used to compute stationary aux targets that do
        not drift with co-learning teammate policies.

        Returns flat actions of shape [B*n_agents] (int32). The vae_rng argument is
        threaded for signature compatibility; the argmax makes the choice independent
        of the VAE noise (argmax(mu + sigma*eps) is not exactly argmax(mu), but for
        a distinguishing experiment we accept the stochasticity).
        """
        logits, _, _ = compute_logits_and_aux(
            frozen_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    @jax.jit
    def compute_policy_kl(params_new, params_old, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng):
        """Mean KL(pi_new || pi_old) over the batch's visited states.

        Direct measurement of consecutive-policy drift (the Sigma_pi quantity
        the paper previously proxied via cosine-temporal-std), as in
        trust-region decomposition. Both forward passes share one vae_rng so
        the comparison is between parameter sets, not noise draws.
        """
        logits_new, _, _ = compute_logits_and_aux(
            params_new, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
        logits_old, _, _ = compute_logits_and_aux(
            params_old, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
        lp_new = jax.nn.log_softmax(logits_new)
        lp_old = jax.nn.log_softmax(logits_old)
        p_new = jnp.exp(lp_new)
        return (p_new * (lp_new - lp_old)).sum(axis=-1).mean()

    @jax.jit
    def snapshot_policy_probs(lag_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng):
        """Soft action distributions of a lagged policy snapshot on rollout inputs.

        Used by the snapshot-lag continuum: targets share label smoothness and
        state dependence with the co-learning condition; only temporal drift
        (set by the refresh interval) differs.
        """
        logits, _, _ = compute_logits_and_aux(
            lag_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
        return jax.nn.softmax(logits, axis=-1)

    @jax.jit
    def actor_update_soft(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_actions,
                          flat_soft_targets, old_lp_sum, advantages_flat, aux_lambda_eff, vae_rng):
        """actor_update with distribution-valued aux targets (cross-entropy vs
        a soft target distribution instead of a hard action index)."""
        def loss_fn(params):
            flat_logits, flat_aux, kl_total = compute_logits_and_aux(
                params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
            B = flat_actions.shape[0]
            logits = flat_logits.reshape(B, n_agents, n_actions)
            lp = jax.nn.log_softmax(logits)
            nlp = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
            ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
            s1 = ratio * advantages_flat
            s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
            p_loss = -jnp.minimum(s1, s2).mean()
            pr = jax.nn.softmax(logits)
            ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
            e_loss = -ent.mean()
            aux_lp = jax.nn.log_softmax(flat_aux, axis=-1)
            aux_loss = -(flat_soft_targets * aux_lp).sum(axis=-1).mean()
            aux_term = aux_lambda_eff * aux_loss if _use_aux_loss else jnp.zeros_like(aux_loss)
            if _use_vae_belief:
                kl_term = _vae_kl_weight * (kl_total / B)
            else:
                kl_term = jnp.zeros_like(aux_loss)
            return p_loss + config.entropy_coef * e_loss + aux_term + kl_term
        loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
        return agent_state.apply_gradients(grads=grads), loss

    _grad_surgery = str(grad_surgery)

    @jax.jit
    def actor_update_surgery(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_actions,
                             flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff, vae_rng,
                             gn_w_policy, gn_w_aux):
        """PCGrad / GradNorm variants of actor_update.

        pcgrad: project the lambda-weighted aux gradient and the policy
          gradient against each other when they conflict (2-task PCGrad).
        gradnorm: combine the raw (unweighted) task gradients with the
          learnable weights maintained python-side by the norm-balancing rule;
          aux_lambda_eff is unused (the learned weight replaces it).
        Returns the raw task-gradient norms and their cosine so the caller can
        both update GradNorm weights and log the diagnostic.
        """
        def policy_loss_fn(p):
            flat_logits, _, _ = compute_logits_and_aux(
                p, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
            B = flat_actions.shape[0]
            logits = flat_logits.reshape(B, n_agents, n_actions)
            lp = jax.nn.log_softmax(logits)
            nlp = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
            ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
            s1 = ratio * advantages_flat
            s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
            p_loss = -jnp.minimum(s1, s2).mean()
            pr = jax.nn.softmax(logits)
            ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
            return p_loss - config.entropy_coef * ent.mean()

        def aux_loss_fn(p):
            _, flat_aux, _ = compute_logits_and_aux(
                p, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
            aux_lp = jax.nn.log_softmax(flat_aux, axis=-1)
            aux_taken = jnp.take_along_axis(
                aux_lp, flat_next_t_actions[..., None], axis=-1).squeeze(-1)
            return -aux_taken.mean()

        p_loss, g_p = jax.value_and_grad(policy_loss_fn)(agent_state.params)
        a_loss, g_a = jax.value_and_grad(aux_loss_fn)(agent_state.params)
        gp, unravel = jax.flatten_util.ravel_pytree(g_p)
        ga, _ = jax.flatten_util.ravel_pytree(g_a)
        norm_p = jnp.linalg.norm(gp)
        norm_a = jnp.linalg.norm(ga)
        cos = jnp.sum(gp * ga) / (norm_p * norm_a + 1e-12)

        if _grad_surgery == "pcgrad":
            ga_w = aux_lambda_eff * ga
            dot = jnp.sum(gp * ga_w)
            coef = jnp.minimum(dot, 0.0)
            gp_proj = gp - coef / (jnp.sum(ga_w * ga_w) + 1e-12) * ga_w
            ga_proj = ga_w - coef / (norm_p ** 2 + 1e-12) * gp
            g = gp_proj + ga_proj
        else:
            g = gn_w_policy * gp + gn_w_aux * ga

        new_state = agent_state.apply_gradients(grads=unravel(g))
        return new_state, p_loss, a_loss, norm_p, norm_a, cos

    # Drift-gated controller: a stop-gradient view of the SAME parameter tree
    # (stop_gradient adds no params), so per-iteration gating is a choice
    # between two jitted updates over identical params.
    if drift_gate_tau > 0.0 or drift_gate_random > 0.0:
        agent_net_sg = VABLAgent(config._replace(stop_gradient_belief_to_aux=True))

        @jax.jit
        def actor_update_sg(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_actions,
                            flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff, vae_rng):
            def forward_sg(params):
                def forward_one(idx, obs_i, belief_i, t_oh_i, t_idx_i):
                    rng_i = jax.random.fold_in(vae_rng, idx)
                    logits, _, aux, kl = agent_net_sg.apply(
                        params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates),
                        rngs={"vae": rng_i})
                    return logits, aux, kl
                idx = jnp.arange(flat_obs.shape[0])
                logits, aux, kls = jax.vmap(forward_one)(idx, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
                return logits, aux, kls.sum()

            def loss_fn(params):
                flat_logits, flat_aux, kl_total = forward_sg(params)
                B = flat_actions.shape[0]
                logits = flat_logits.reshape(B, n_agents, n_actions)
                lp = jax.nn.log_softmax(logits)
                nlp = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
                ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
                s1 = ratio * advantages_flat
                s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
                p_loss = -jnp.minimum(s1, s2).mean()
                pr = jax.nn.softmax(logits)
                ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
                e_loss = -ent.mean()
                aux_lp = jax.nn.log_softmax(flat_aux, axis=-1)
                aux_taken = jnp.take_along_axis(
                    aux_lp, flat_next_t_actions[..., None], axis=-1).squeeze(-1)
                aux_loss = -aux_taken.mean()
                aux_term = aux_lambda_eff * aux_loss if _use_aux_loss else jnp.zeros_like(aux_loss)
                if _use_vae_belief:
                    kl_term = _vae_kl_weight * (kl_total / B)
                else:
                    kl_term = jnp.zeros_like(aux_loss)
                return p_loss + config.entropy_coef * e_loss + aux_term + kl_term
            loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
            return agent_state.apply_gradients(grads=grads), loss

    # Static scalar constants the jit should bake in from config
    _use_aux_loss = bool(config.use_aux_loss)
    _use_vae_belief = bool(config.use_vae_belief)
    _vae_kl_weight = float(config.vae_kl_weight)

    @jax.jit
    def actor_update(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_actions,
                     flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff, vae_rng):
        def loss_fn(params):
            flat_logits, flat_aux, kl_total = compute_logits_and_aux(
                params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, vae_rng)
            # flat_logits: [N*H*n_agents, n_actions]
            # flat_aux: [N*H*n_agents, n_teammates, n_actions]
            B = flat_actions.shape[0]  # N*H
            logits = flat_logits.reshape(B, n_agents, n_actions)
            lp = jax.nn.log_softmax(logits)
            nlp = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)  # [B]
            ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
            s1 = ratio * advantages_flat
            s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
            p_loss = -jnp.minimum(s1, s2).mean()
            pr = jax.nn.softmax(logits)
            ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
            e_loss = -ent.mean()

            # Auxiliary loss: predict next-step teammate actions from current beliefs.
            # Gated by use_aux_loss (static config) and scaled by aux_lambda_eff (runtime).
            aux_lp = jax.nn.log_softmax(flat_aux, axis=-1)  # [B*n_agents, n_teammates, n_actions]
            aux_taken = jnp.take_along_axis(
                aux_lp, flat_next_t_actions[..., None], axis=-1
            ).squeeze(-1)  # [B*n_agents, n_teammates]
            aux_loss = -aux_taken.mean()

            aux_term = aux_lambda_eff * aux_loss if _use_aux_loss else jnp.zeros_like(aux_loss)
            # VAE KL term (per-sample average so magnitude is independent of batch).
            if _use_vae_belief:
                kl_term = _vae_kl_weight * (kl_total / B)
            else:
                kl_term = jnp.zeros_like(aux_loss)
            return p_loss + config.entropy_coef * e_loss + aux_term + kl_term
        loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
        return agent_state.apply_gradients(grads=grads), loss

    @jax.jit
    def critic_update(critic_state, flat_states, flat_returns):
        def loss_fn(params):
            vals = jax.vmap(lambda s: critic_net.apply(params, s))(flat_states)
            return ((vals - flat_returns) ** 2).mean()
        loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    # Snapshot initial (pre-training) agent params for the frozen-target-policy
    # experiment. When `aux_frozen_target_policy` is True, the aux targets are
    # computed by running this frozen snapshot on rollout obs; the snapshot
    # never updates, so aux targets remain stationary even as the live policy
    # and its teammates co-learn.
    frozen_agent_params = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), agent_state.params)
    if aux_frozen_target_policy:
        print("  [aux-frozen-targets] snapshot taken; aux targets are stationary from iter 0")

    # Q4 auxiliary-task variants: separate head + optimizer; agent encoder
    # still receives aux gradients through the belief.
    variant_head = None
    vh_state = None
    if aux_task != "action":
        vh_out = (n_teammates * config.hidden_dim) if aux_task == "latent" else obs_dim
        variant_head = VariantAuxHead(out_dim=vh_out)
        vh_params = variant_head.init(
            jax.random.PRNGKey(seed + 555001), jnp.zeros(config.hidden_dim))
        vh_state = TrainState.create(
            apply_fn=variant_head.apply, params=vh_params,
            tx=optax.chain(optax.clip_by_global_norm(config.grad_clip),
                           optax.adam(config.actor_lr, eps=1e-5)))
        print(f"  [aux-task={aux_task}] variant head out_dim={vh_out}")

        @jax.jit
        def actor_update_variant(agent_state, vh_state, flat_obs, flat_beliefs, flat_t_oh,
                                 flat_t_idx, flat_actions, flat_var_targets, old_lp_sum,
                                 advantages_flat, aux_lambda_eff, vae_rng):
            def loss_fn(params, hparams):
                def forward_one(idx, obs_i, belief_i, t_oh_i, t_idx_i):
                    rng_i = jax.random.fold_in(vae_rng, idx)
                    logits, new_b, _aux, kl = agent_net.apply(
                        params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates),
                        rngs={"vae": rng_i})
                    return logits, new_b, kl
                idx = jnp.arange(flat_obs.shape[0])
                flat_logits, flat_newb, kls = jax.vmap(forward_one)(
                    idx, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
                B = flat_actions.shape[0]
                logits = flat_logits.reshape(B, n_agents, n_actions)
                lp = jax.nn.log_softmax(logits)
                nlp = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
                ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
                s1 = ratio * advantages_flat
                s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
                p_loss = -jnp.minimum(s1, s2).mean()
                pr = jax.nn.softmax(logits)
                ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
                e_loss = -ent.mean()
                pred = jax.vmap(lambda b: variant_head.apply(hparams, b))(flat_newb)
                aux_loss = jnp.mean((pred - flat_var_targets) ** 2)
                if _use_vae_belief:
                    kl_term = _vae_kl_weight * (kls.sum() / B)
                else:
                    kl_term = jnp.zeros(())
                return p_loss + config.entropy_coef * e_loss + aux_lambda_eff * aux_loss + kl_term
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(
                agent_state.params, vh_state.params)
            g_agent, g_head = grads
            return (agent_state.apply_gradients(grads=g_agent),
                    vh_state.apply_gradients(grads=g_head), loss)

    # ---- Training loop ----
    rewards_history = []
    best_reward = float("-inf")
    grad_log = []  # list of {iteration, norm_policy, norm_aux, cosine}

    # Rebuttal-instrumentation state. side_rng is an independent stream so the
    # new features never consume from the canonical rng sequence.
    side_rng = jax.random.PRNGKey(seed + 777001)
    prev_iter_params = agent_state.params  # for consecutive-policy KL
    lagged_params = agent_state.params     # for snapshot-lag targets
    kl_log = []      # {iteration, policy_kl}
    gate_log = []    # {iteration, cosine, gate_active}
    gate_cosines = []
    gn_log = []      # {iteration, w_policy, w_aux}
    gn_w_policy, gn_w_aux = 1.0, 1.0
    GN_LR = 0.025  # GradNorm weight learning rate (Chen et al. 2018)
    prev_grad_vecs = None  # (g_policy, g_aux) at the previous logged iteration

    print("  Starting (compiling on first iteration)...")
    t0 = time.time()

    n_iterations = (n_episodes + n_envs - 1) // n_envs
    total_episodes = 0

    for iteration in range(n_iterations):
        rng, reset_rng = jax.random.split(rng)
        env_keys = jax.random.split(reset_rng, n_envs)
        obs_dict_batch, env_state_batch = vmap_reset(env_keys)
        beliefs = jnp.zeros((n_envs, n_agents, config.hidden_dim))
        prev_actions = jnp.zeros((n_envs, n_agents), dtype=jnp.int32)

        # Stack initial obs: [N, n_agents, obs_dim]
        obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
        states_batch = obs_batch.reshape(n_envs, -1)

        # Rollout buffers (preallocated)
        buf_obs = jnp.zeros((horizon, n_envs, n_agents, obs_dim))
        buf_actions = jnp.zeros((horizon, n_envs, n_agents), dtype=jnp.int32)
        buf_log_probs = jnp.zeros((horizon, n_envs, n_agents))
        buf_values = jnp.zeros((horizon, n_envs))
        buf_beliefs = jnp.zeros((horizon, n_envs, n_agents, config.hidden_dim))
        buf_states = jnp.zeros((horizon, n_envs, obs_dim * n_agents))
        buf_rewards = jnp.zeros((horizon, n_envs))
        buf_dones = jnp.zeros((horizon, n_envs))

        ep_rewards = jnp.zeros(n_envs)

        # Roll out N envs in parallel for `horizon` steps
        for step in range(horizon):
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            values_batch = get_value_vec(critic_state.params, states_batch)
            actions_batch, new_beliefs, log_probs_batch = step_agents_vec(
                agent_state.params, obs_batch, beliefs, prev_actions, act_rng)

            buf_obs = buf_obs.at[step].set(obs_batch)
            buf_actions = buf_actions.at[step].set(actions_batch)
            buf_log_probs = buf_log_probs.at[step].set(log_probs_batch)
            buf_values = buf_values.at[step].set(values_batch)
            buf_beliefs = buf_beliefs.at[step].set(beliefs)
            buf_states = buf_states.at[step].set(states_batch)

            # Step all envs in parallel
            actions_dict_batch = {a: actions_batch[:, i] for i, a in enumerate(agent_names)}
            step_keys = jax.random.split(step_rng, n_envs)
            obs_dict_batch, env_state_batch, reward_dict_batch, done_dict_batch, info = vmap_step(
                step_keys, env_state_batch, actions_dict_batch)

            # Reward (sparse + shaped)
            rewards = reward_dict_batch[agent_names[0]]
            if "shaped_reward" in info:
                for a in agent_names:
                    if a in info["shaped_reward"]:
                        rewards = rewards + info["shaped_reward"][a]
            dones = done_dict_batch["__all__"].astype(jnp.float32)

            buf_rewards = buf_rewards.at[step].set(rewards)
            buf_dones = buf_dones.at[step].set(dones)
            ep_rewards = ep_rewards + rewards

            # Update for next step
            obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
            states_batch = obs_batch.reshape(n_envs, -1)
            beliefs = new_beliefs
            prev_actions = actions_batch

        # Episode rewards (one per env)
        for env_i in range(n_envs):
            r = float(ep_rewards[env_i])
            rewards_history.append(r)
            best_reward = max(best_reward, r)
            total_episodes += 1

        # GAE per env, then flatten
        # Reshape buffers from [H, N, ...] to [N, H, ...]
        rewards_NH = buf_rewards.transpose(1, 0)  # [N, H]
        values_NH = buf_values.transpose(1, 0)
        dones_NH = buf_dones.transpose(1, 0)

        advantages_NH, returns_NH = compute_gae_vec(rewards_NH, values_NH, dones_NH)

        # Flatten for PPO update: [H, N, n_agents, ...] -> [N*H, n_agents, ...]
        flat_obs = buf_obs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, obs_dim)
        flat_beliefs = buf_beliefs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, config.hidden_dim)
        flat_actions = buf_actions.transpose(1, 0, 2).reshape(n_envs * horizon, n_agents)

        # Teammate one-hots for each (env, t, agent)
        # acts at [env, t]: [n_envs, horizon, n_agents]
        acts_NH = buf_actions.transpose(1, 0, 2)  # [N, H, n_agents]
        # For each (env, t, agent_i), need teammate actions
        # Use teammate_idx: [n_agents, n_teammates]
        # acts_NH[:, :, teammate_idx]: [N, H, n_agents, n_teammates]
        t_acts_NHA = acts_NH[:, :, teammate_idx]  # broadcast indexing
        flat_t_oh = jax.nn.one_hot(t_acts_NHA, n_actions).reshape(
            n_envs * horizon * n_agents, n_teammates, n_actions)

        # Per-(env, t, agent_i) teammate indices — static tile of teammate_idx
        # replicated across the (n_envs, horizon) batch dimension.
        flat_t_idx = jnp.tile(teammate_idx[None, None, :, :],
                               (n_envs, horizon, 1, 1)).reshape(
            n_envs * horizon * n_agents, n_teammates).astype(jnp.int32)

        # Next-step teammate actions for auxiliary loss
        # Shift acts_NH by 1 along time axis: predict t+1 from belief at t
        flat_soft_targets = None
        if aux_noise_targets:
            # Replace aux targets with uniform random integers resampled each
            # iteration. Aux network capacity + gradient pathway preserved; the
            # co-learning SIGNAL is destroyed (targets carry no teammate info).
            # Predicts different things per hypothesis:
            #   paper's story (directional co-learning noise): unclear — noise
            #     is IID, not teammate-drift-correlated; Sigma_eps may differ
            #     in structure from co-learning case.
            #   capacity story: pathology persists (aux still consumes capacity).
            rng, noise_rng = jax.random.split(rng)
            noise_targets = jax.random.randint(
                noise_rng, shape=(n_envs, horizon, n_agents, n_teammates),
                minval=0, maxval=n_actions, dtype=jnp.int32)
            next_t_acts_NHA = noise_targets
        elif aux_frozen_target_policy:
            # Replace actual teammate-action targets with the FROZEN-POLICY's
            # deterministic (argmax) actions on the same rollout obs/beliefs at
            # step t+1. This keeps aux capacity, architecture, and the
            # aux-to-encoder gradient pathway identical to Full, but decouples
            # the *targets* from co-learning teammate policies.
            rng, rng_frozen_vae = jax.random.split(rng)
            frozen_actions_flat = frozen_policy_argmax_actions(
                frozen_agent_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                rng_frozen_vae,
            )  # [N*H*n_agents]
            frozen_acts_NH_agents = frozen_actions_flat.reshape(n_envs, horizon, n_agents)
            next_frozen_NH = jnp.concatenate(
                [frozen_acts_NH_agents[:, 1:], jnp.zeros((n_envs, 1, n_agents), dtype=jnp.int32)], axis=1
            )
            next_t_acts_NHA = next_frozen_NH[:, :, teammate_idx]
        elif aux_snapshot_refresh != 0 or aux_ema_alpha > 0.0:
            # Snapshot-lag continuum: aux targets come from a policy snapshot
            # refreshed every `aux_snapshot_refresh` iterations (-1 = never,
            # the frozen endpoint). With aux_soft_targets, targets are the
            # snapshot policy's full action DISTRIBUTION, so all lags share
            # label smoothness/state dependence and differ only in drift rate.
            # EMA mode instead updates the target policy as a slow moving
            # average of the live policy every iteration (target design that
            # reduces drift at the source while tracking task relevance).
            if aux_ema_alpha > 0.0:
                lagged_params = jax.tree_util.tree_map(
                    lambda e, p: aux_ema_alpha * e + (1.0 - aux_ema_alpha) * p,
                    lagged_params, agent_state.params)
            elif aux_snapshot_refresh > 0 and iteration % aux_snapshot_refresh == 0:
                lagged_params = agent_state.params
            side_rng, rng_snap = jax.random.split(side_rng)
            if aux_soft_targets:
                probs_flat = snapshot_policy_probs(
                    lagged_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, rng_snap)
                probs_NHA = probs_flat.reshape(n_envs, horizon, n_agents, n_actions)
                next_probs = jnp.concatenate(
                    [probs_NHA[:, 1:],
                     jnp.full((n_envs, 1, n_agents, n_actions), 1.0 / n_actions)], axis=1)
                soft_NHA = next_probs[:, :, teammate_idx]
                flat_soft_targets = soft_NHA.reshape(
                    n_envs * horizon * n_agents, n_teammates, n_actions)
                # Hard argmax view so the gradient-decomp diagnostic still works.
                next_t_acts_NHA = jnp.argmax(soft_NHA, axis=-1).astype(jnp.int32)
            else:
                lag_actions_flat = frozen_policy_argmax_actions(
                    lagged_params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, rng_snap)
                lag_acts = lag_actions_flat.reshape(n_envs, horizon, n_agents)
                next_lag = jnp.concatenate(
                    [lag_acts[:, 1:], jnp.zeros((n_envs, 1, n_agents), dtype=jnp.int32)], axis=1)
                next_t_acts_NHA = next_lag[:, :, teammate_idx]
        else:
            next_acts_NH = jnp.concatenate(
                [acts_NH[:, 1:], jnp.zeros((n_envs, 1, n_agents), dtype=jnp.int32)], axis=1
            )  # [N, H, n_agents]
            next_t_acts_NHA = next_acts_NH[:, :, teammate_idx]  # [N, H, n_agents, n_teammates]
        flat_next_t_actions = next_t_acts_NHA.reshape(
            n_envs * horizon * n_agents, n_teammates).astype(jnp.int32)

        old_log_probs_NH = buf_log_probs.transpose(1, 0, 2)  # [N, H, n_agents]
        old_lp_sum_flat = old_log_probs_NH.sum(axis=-1).reshape(n_envs * horizon)  # [N*H]

        advantages_flat = advantages_NH.reshape(n_envs * horizon)
        returns_flat = returns_NH.reshape(n_envs * horizon)
        flat_states = buf_states.transpose(1, 0, 2).reshape(n_envs * horizon, obs_dim * n_agents)

        # Q4 variant targets. latent: each teammate's NEXT-step belief
        # (structured, drifting as teammates learn). recon: own current obs
        # (structured, stationary mapping). Both are buffer data (detached).
        flat_var_targets = None
        if aux_task == "latent":
            bel_NH = buf_beliefs.transpose(1, 0, 2, 3)  # [N, H, agents, hidden]
            next_bel = jnp.concatenate(
                [bel_NH[:, 1:], jnp.zeros((n_envs, 1, n_agents, config.hidden_dim))], axis=1)
            t_bel = next_bel[:, :, teammate_idx]  # [N, H, agents, n_teammates, hidden]
            flat_var_targets = t_bel.reshape(
                n_envs * horizon * n_agents, n_teammates * config.hidden_dim)
        elif aux_task == "recon":
            flat_var_targets = flat_obs

        # Compute effective aux_lambda for this iteration.
        # Q8 schedules (cosine / exp / kl_adaptive) take precedence; otherwise
        # the original behavior: aux_anneal_fraction=0.0 -> constant lambda,
        # else linear decay over the first fraction of training, held at 0.
        if aux_schedule == "cosine":
            aux_lambda_eff = float(config.aux_lambda * 0.5 *
                                   (1.0 + np.cos(np.pi * iteration / max(1, n_iterations - 1))))
        elif aux_schedule == "exp":
            rate = 0.01 ** (1.0 / max(1, n_iterations - 1))
            aux_lambda_eff = float(config.aux_lambda * (rate ** iteration))
        elif aux_schedule == "kl_adaptive":
            # yGKw Q8: adaptive controller keyed to the measured Sigma_pi
            # proxy. lambda is scaled down when consecutive-policy KL exceeds
            # the reference late-training drift (2.5e-3 nats).
            KL_REF = 2.5e-3
            if kl_log:
                last_kl = kl_log[-1]["policy_kl"]
                aux_lambda_eff = float(config.aux_lambda * min(1.0, KL_REF / max(last_kl, 1e-8)))
            else:
                aux_lambda_eff = float(config.aux_lambda)
        elif config.aux_anneal_fraction > 0.0:
            anneal_iters = max(1, int(config.aux_anneal_fraction * n_iterations))
            frac_remaining = max(0.0, 1.0 - iteration / anneal_iters)
            aux_lambda_eff = float(config.aux_lambda * frac_remaining)
        else:
            aux_lambda_eff = float(config.aux_lambda)

        # Consecutive-policy KL: drift of the (shared-parameter) target policy
        # between iteration starts, measured on this iteration's visited states.
        # Direct Sigma_pi measurement for the rebuttal (yGKw Q1, fXvf Q6).
        if log_policy_kl and iteration > 0:
            side_rng, rng_kl = jax.random.split(side_rng)
            kl_val = compute_policy_kl(
                agent_state.params, prev_iter_params, flat_obs, flat_beliefs,
                flat_t_oh, flat_t_idx, rng_kl)
            kl_log.append({"iteration": int(iteration), "policy_kl": float(kl_val)})
        prev_iter_params = agent_state.params

        # Drift-gated controller (yGKw Q6): compute the gradient cosine every
        # iteration; gate the aux->encoder pathway (stop-gradient) whenever the
        # rolling cosine-std exceeds the threshold.
        gate_active = False
        if drift_gate_random > 0.0:
            # Q6 matched-duty random-gating control: gate with fixed
            # probability, independent of any drift signal.
            side_rng, r_g = jax.random.split(side_rng)
            gate_active = bool(jax.random.uniform(r_g) < drift_gate_random)
            gate_log.append({"iteration": int(iteration),
                             "gate_active": gate_active, "random": True})
        elif drift_gate_tau > 0.0:
            side_rng, rng_gate = jax.random.split(side_rng)
            _np_g, _na_g, co_g, _gp_v, _ga_v = compute_separate_gradients(
                agent_state.params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                jnp.asarray(aux_lambda_eff), rng_gate)
            gate_cosines.append(float(co_g))
            if len(gate_cosines) >= drift_gate_window:
                gate_active = bool(
                    float(np.std(gate_cosines[-drift_gate_window:])) > drift_gate_tau)
            gate_log.append({"iteration": int(iteration), "cosine": float(co_g),
                             "gate_active": gate_active})

        # Gradient decomposition diagnostic (before the PPO epochs so the
        # snapshot is taken at the same params we're about to update from).
        if log_gradient_decomp and (iteration % grad_log_interval == 0):
            rng, rng_gd_vae = jax.random.split(rng)
            np_, na_, co_, gp_vec, ga_vec = compute_separate_gradients(
                agent_state.params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                jnp.asarray(aux_lambda_eff), rng_gd_vae,
            )
            # Self-cosines: direction stability of each task gradient between
            # consecutive logged iterations (added 2026-07-24; the between-task
            # cosine conflates the two and does not separate frozen from full).
            gp_np = np.asarray(gp_vec)
            ga_np = np.asarray(ga_vec)
            entry = {
                "iteration": int(iteration),
                "norm_policy": float(np_),
                "norm_aux": float(na_),
                "cosine": float(co_),
                "aux_lambda_eff": float(aux_lambda_eff),
            }
            if prev_grad_vecs is not None:
                pgp, pga = prev_grad_vecs
                denom_p = float(np.linalg.norm(gp_np) * np.linalg.norm(pgp) + 1e-12)
                denom_a = float(np.linalg.norm(ga_np) * np.linalg.norm(pga) + 1e-12)
                entry["policy_self_cos"] = float(np.dot(gp_np, pgp) / denom_p)
                entry["aux_self_cos"] = float(np.dot(ga_np, pga) / denom_a)
            prev_grad_vecs = (gp_np, ga_np)
            # Belief effective rank (participation ratio of singular values) on
            # a fixed-size belief sample: representation-collapse diagnostic
            # (PYCT Q1), logged wherever gradient decomp is logged.
            bel = np.asarray(flat_beliefs[:2048])
            sv = np.linalg.svd(bel, compute_uv=False)
            entry["belief_effective_rank"] = float((sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12))
            # J_pi finite-difference estimator (yGKw Q3): perturb the aux
            # TARGETS (flip each to uniform-random with prob eps) and measure
            # the relative aux-gradient response. Three eps values give a
            # linearity check; the same vae rng is reused so the response is
            # purely target-driven.
            if log_jpi and float(np.linalg.norm(ga_np)) > 0:
                for eps in (0.05, 0.1, 0.2):
                    side_rng, r_flip, r_act = jax.random.split(side_rng, 3)
                    flip = jax.random.bernoulli(r_flip, eps, flat_next_t_actions.shape)
                    rand_a = jax.random.randint(
                        r_act, flat_next_t_actions.shape, 0, n_actions, dtype=jnp.int32)
                    pert = jnp.where(flip, rand_a, flat_next_t_actions)
                    _n1, _n2, _c, _gp2, ga_pert = compute_separate_gradients(
                        agent_state.params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                        flat_actions, pert, old_lp_sum_flat, advantages_flat,
                        jnp.asarray(aux_lambda_eff), rng_gd_vae)
                    ga_pert = np.asarray(ga_pert)
                    entry[f"jpi_rel_eps{eps}"] = float(
                        np.linalg.norm(ga_pert - ga_np) / (np.linalg.norm(ga_np) + 1e-12))
            grad_log.append(entry)

        # PPO epochs. Dispatch order: soft targets > gradient surgery >
        # drift gate > canonical. The rng split is identical in every branch
        # so the canonical path's rng stream is unchanged by the new modes.
        for _ in range(config.ppo_epochs):
            rng, rng_au_vae = jax.random.split(rng)
            if aux_task != "action":
                agent_state, vh_state, a_loss = actor_update_variant(
                    agent_state, vh_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_actions, flat_var_targets, old_lp_sum_flat, advantages_flat,
                    jnp.asarray(aux_lambda_eff), rng_au_vae)
            elif flat_soft_targets is not None:
                agent_state, a_loss = actor_update_soft(
                    agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_actions, flat_soft_targets, old_lp_sum_flat, advantages_flat,
                    jnp.asarray(aux_lambda_eff), rng_au_vae)
            elif _grad_surgery != "none":
                agent_state, p_l, a_l, np_r, na_r, cos_r = actor_update_surgery(
                    agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                    jnp.asarray(aux_lambda_eff), rng_au_vae,
                    jnp.asarray(gn_w_policy), jnp.asarray(gn_w_aux))
                a_loss = p_l
                if _grad_surgery == "gradnorm":
                    # Norm-balancing GradNorm update (inverse-training-rate
                    # term dropped: PPO losses can be negative, so L(t)/L(0)
                    # ratios are ill-defined; this is the standard RL
                    # adaptation and is reported as such in the rebuttal).
                    G_p = gn_w_policy * float(np_r)
                    G_a = gn_w_aux * float(na_r)
                    G_bar = 0.5 * (G_p + G_a)
                    gn_w_policy -= GN_LR * float(np.sign(G_p - G_bar)) * float(np_r)
                    gn_w_aux -= GN_LR * float(np.sign(G_a - G_bar)) * float(na_r)
                    gn_w_policy = max(gn_w_policy, 1e-3)
                    gn_w_aux = max(gn_w_aux, 1e-3)
                    scale = 2.0 / (gn_w_policy + gn_w_aux)
                    gn_w_policy *= scale
                    gn_w_aux *= scale
            elif drift_gate_tau > 0.0 and gate_active:
                agent_state, a_loss = actor_update_sg(
                    agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                    jnp.asarray(aux_lambda_eff), rng_au_vae)
            else:
                agent_state, a_loss = actor_update(
                    agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                    jnp.asarray(aux_lambda_eff), rng_au_vae)
            critic_state, c_loss = critic_update(critic_state, flat_states, returns_flat)

        if _grad_surgery == "gradnorm" and (iteration % 5 == 0):
            gn_log.append({"iteration": int(iteration),
                           "w_policy": float(gn_w_policy), "w_aux": float(gn_w_aux)})

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            avg_r = np.mean(rewards_history[-n_envs * log_interval:])
            elapsed = time.time() - t0
            eps = total_episodes / elapsed
            print(f"  Iter {iteration+1:5d}/{n_iterations} | Eps {total_episodes:6d} | "
                  f"R: {avg_r:7.1f} | Best: {best_reward:.0f} | {eps:.1f} ep/s")

    elapsed = time.time() - t0
    final = float(np.mean(rewards_history[-min(50, len(rewards_history)):]))
    print(f"\nDone in {elapsed:.0f}s ({total_episodes/elapsed:.1f} ep/s)")
    print(f"Final: {final:.1f}, Best: {best_reward:.0f}")

    # Record the exact config (including fix-path knobs) with the results so
    # we can tell runs apart post hoc without relying on filenames.
    cfg_record = {
        "use_attention": bool(config.use_attention),
        "use_aux_loss": bool(config.use_aux_loss),
        "aux_lambda": float(config.aux_lambda),
        "stop_gradient_belief_to_aux": bool(config.stop_gradient_belief_to_aux),
        "aux_anneal_fraction": float(config.aux_anneal_fraction),
        "separate_aux_encoder": bool(config.separate_aux_encoder),
        "aux_frozen_target_policy": bool(aux_frozen_target_policy),
        "aux_noise_targets": bool(aux_noise_targets),
        "log_gradient_decomp": bool(log_gradient_decomp),
        "log_policy_kl": bool(log_policy_kl),
        "aux_snapshot_refresh": int(aux_snapshot_refresh),
        "aux_soft_targets": bool(aux_soft_targets),
        "grad_surgery": str(grad_surgery),
        "drift_gate_tau": float(drift_gate_tau),
        "drift_gate_window": int(drift_gate_window),
        "log_jpi": bool(log_jpi),
        "aux_task": str(aux_task),
        "aux_schedule": str(aux_schedule),
        "drift_gate_random": float(drift_gate_random),
        "aux_ema_alpha": float(aux_ema_alpha),
        "use_vae_belief": bool(config.use_vae_belief),
        "vae_kl_weight": float(config.vae_kl_weight),
        "n_agents": int(config.n_agents),
        "n_actions": int(config.n_actions),
        "obs_dim": int(config.obs_dim),
        "hidden_dim": int(config.hidden_dim),
        "aux_hidden_dim": int(config.aux_hidden_dim),
        "attention_heads": int(config.attention_heads),
        "ppo_epochs": int(config.ppo_epochs),
        "actor_lr": float(config.actor_lr),
        "critic_lr": float(config.critic_lr),
        "layout": str(layout),
        "n_envs": int(n_envs),
        "horizon": int(horizon),
        "n_episodes": int(n_episodes),
        "seed": int(seed),
        "vabl_version": "v2",
    }
    results = {
        "rewards": rewards_history,
        "final_reward": final,
        "best_reward": best_reward,
        "elapsed": elapsed,
        "config": cfg_record,
        "gradient_decomp": grad_log,  # [] if disabled
        "policy_kl": kl_log,          # [] if disabled
        "drift_gate": gate_log,       # [] if disabled
        "gradnorm_weights": gn_log,   # [] if disabled
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save", default=None)
    # Ablation / fix-path knobs. Launch scripts set these to select the
    # config being run; defaults reproduce the canonical Full VABL.
    parser.add_argument("--no-attention", action="store_true",
                        help="Replace attention with mean pooling (VABL No-Attn ablation).")
    parser.add_argument("--no-aux-loss", action="store_true",
                        help="Hard-disable the auxiliary loss regardless of aux_lambda.")
    parser.add_argument("--aux-lambda", type=float, default=0.05,
                        help="Initial/constant auxiliary loss weight.")
    parser.add_argument("--stop-gradient-belief", action="store_true",
                        help="Stop aux gradients from flowing into the belief encoder.")
    parser.add_argument("--aux-anneal-fraction", type=float, default=0.0,
                        help="Linearly anneal aux_lambda to 0 over the first F of training (0 = constant).")
    parser.add_argument("--separate-aux-encoder", action="store_true",
                        help="Aux predictor uses its own parallel encoder (intra-actor control).")
    parser.add_argument("--aux-frozen-target-policy", action="store_true",
                        help="Use a snapshot of the initial agent policy to generate aux targets; "
                             "keeps aux capacity + gradient pathway but removes co-learning "
                             "non-stationarity from targets (distinguishing experiment).")
    parser.add_argument("--aux-noise-targets", action="store_true",
                        help="Replace aux targets with uniform random integers (resampled each "
                             "iteration). Tests whether aux CAPACITY or co-learning SIGNAL matters.")
    parser.add_argument("--aux-hidden-dim", type=int, default=None,
                        help="Override aux-MLP hidden dim (default 64). Used to scale aux capacity "
                             "independently of target source, e.g. --aux-hidden-dim 16.")
    parser.add_argument("--use-vae-belief", action="store_true",
                        help="Use Dynamic-Belief-style VAE belief encoder (GRU -> mu/log_sigma2 -> "
                             "reparameterized sample) with KL-to-N(0, I) regularizer.")
    parser.add_argument("--vae-kl-weight", type=float, default=0.005,
                        help="KL weight for VAE belief. Default 0.005 matches Zhai et al.")
    parser.add_argument("--log-gradient-decomp", action="store_true",
                        help="Log per-iteration policy/aux gradient norms + cosine (diagnostic).")
    parser.add_argument("--grad-log-interval", type=int, default=25,
                        help="If --log-gradient-decomp, log every N iterations.")
    # Rebuttal instrumentation (2026-07-24).
    parser.add_argument("--log-policy-kl", action="store_true",
                        help="Log per-iteration consecutive-policy KL on visited states "
                             "(direct Sigma_pi drift measurement).")
    parser.add_argument("--aux-snapshot-refresh", type=int, default=0,
                        help="Snapshot-lag continuum: refresh the aux-target policy snapshot "
                             "every L iterations (0 = off/co-learning, -1 = never/frozen).")
    parser.add_argument("--aux-soft-targets", action="store_true",
                        help="With --aux-snapshot-refresh: use the snapshot policy's full "
                             "action distribution as the aux target (matched label smoothness).")
    parser.add_argument("--grad-surgery", choices=["none", "pcgrad", "gradnorm"], default="none",
                        help="Apply PCGrad projection or GradNorm adaptive weighting between "
                             "the policy and aux gradients (reviewer-requested baselines).")
    parser.add_argument("--drift-gate-tau", type=float, default=0.0,
                        help="Drift-gated controller: stop-gradient the aux->encoder pathway "
                             "when rolling cosine-std exceeds tau (0 = off).")
    parser.add_argument("--drift-gate-window", type=int, default=10,
                        help="Rolling window (iterations) for the drift gate cosine-std.")
    # Rebuttal round 2 (2026-07-26).
    parser.add_argument("--log-jpi", action="store_true",
                        help="Finite-difference J_pi estimator: perturb aux targets at "
                             "eps in {0.05,0.1,0.2} at grad-log points and record the "
                             "relative aux-gradient response (yGKw Q3).")
    parser.add_argument("--aux-task", choices=["action", "latent", "recon"], default="action",
                        help="Auxiliary task variant (yGKw Q4): action = canonical "
                             "teammate-action prediction; latent = predict teammate next "
                             "belief (drifting); recon = reconstruct own obs (stationary).")
    parser.add_argument("--aux-schedule", choices=["constant", "cosine", "exp", "kl_adaptive"],
                        default="constant",
                        help="Aux-lambda schedule beyond linear annealing (yGKw Q8).")
    parser.add_argument("--drift-gate-random", type=float, default=0.0,
                        help="Random-gating control at fixed duty cycle P (Q6 control).")
    parser.add_argument("--aux-ema-alpha", type=float, default=0.0,
                        help="EMA-distilled auxiliary targets: predict soft actions of an "
                             "EMA copy of the policy (target-design intervention; implies "
                             "--aux-soft-targets). 0 = off; typical 0.995.")
    args = parser.parse_args()

    # Build config from CLI knobs
    cfg_kwargs = dict(
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
        aux_lambda=args.aux_lambda,
        stop_gradient_belief_to_aux=args.stop_gradient_belief,
        aux_anneal_fraction=args.aux_anneal_fraction,
        separate_aux_encoder=args.separate_aux_encoder,
    )
    if args.aux_hidden_dim is not None:
        cfg_kwargs["aux_hidden_dim"] = int(args.aux_hidden_dim)
    if args.use_vae_belief:
        cfg_kwargs["use_vae_belief"] = True
        cfg_kwargs["vae_kl_weight"] = float(args.vae_kl_weight)
    base_config = VABLConfig()._replace(**cfg_kwargs)

    train_vabl_vec(
        config=base_config,
        layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, log_interval=args.log_interval,
        save_path=args.save,
        aux_frozen_target_policy=args.aux_frozen_target_policy,
        aux_noise_targets=args.aux_noise_targets,
        log_gradient_decomp=args.log_gradient_decomp,
        grad_log_interval=args.grad_log_interval,
        log_policy_kl=args.log_policy_kl,
        aux_snapshot_refresh=args.aux_snapshot_refresh,
        aux_soft_targets=args.aux_soft_targets,
        grad_surgery=args.grad_surgery,
        drift_gate_tau=args.drift_gate_tau,
        drift_gate_window=args.drift_gate_window,
        log_jpi=args.log_jpi,
        aux_task=args.aux_task,
        aux_schedule=args.aux_schedule,
        drift_gate_random=args.drift_gate_random,
        aux_ema_alpha=args.aux_ema_alpha,
    )
