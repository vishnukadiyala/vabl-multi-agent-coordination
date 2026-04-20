"""Vectorized VABL JAX training — N parallel envs via jax.vmap.

Canonical camera-ready entry point (2026-04-09). Uses `VABLv2Agent` from
`vabl_v2.py`. The v1 `VABLAgent` from `vabl.py` is deprecated: v2 fixes bugs in
identity-embedding indexing, visibility-mask application, and adds multi-head
attention and orthogonal init. See wiki/concepts/aux_loss_bug.md for the
rationale.

This runner supports three aux-loss knobs exposed by VABLv2Config:
  * use_aux_loss (bool): hard on/off switch for the aux loss term.
  * aux_lambda (float): the initial/constant aux loss weight.
  * stop_gradient_belief_to_aux (bool): when True, aux gradients never reach
    the belief encoder.
  * aux_anneal_fraction (float): 0.0 means constant lambda; values in (0, 1]
    linearly decay aux_lambda → 0 over the first `aux_anneal_fraction` of
    training iterations, held at 0 afterwards.

Usage:
    python -m marl_research.algorithms.jax.train_vabl_vec --layout cramped_room --episodes 5000 --n-envs 32
"""

import time
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.vabl_v2 import (
    VABLv2Config as VABLConfig,
    VABLv2Agent as VABLAgent,
    VABLv2Critic as Critic,
)


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
):
    if config is None:
        config = VABLConfig()

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

    rng, rng_a, rng_c = jax.random.split(rng, 3)
    # v2 agent signature: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
    dummy_t_idx = jnp.arange(1, n_agents, dtype=jnp.int32)  # teammate indices for agent 0
    agent_params = agent_net.init(rng_a, jnp.zeros(obs_dim), jnp.zeros(config.hidden_dim),
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
                t_idx = teammate_idx[i]
                t_oh = jax.nn.one_hot(env_prev_acts[t_idx], n_actions)
                # v2 signature: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
                logits, new_b, _ = agent_net.apply(
                    params, env_obs[i], env_beliefs[i], t_oh, t_idx, jnp.ones(n_teammates))
                action = jax.random.categorical(rng_i, logits)
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
    def compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx):
        """Returns (logits [B*n_agents, n_actions], aux_logits [B*n_agents, n_teammates, n_actions])."""
        def forward_one(obs_i, belief_i, t_oh_i, t_idx_i):
            # v2 signature: (obs, prev_belief, teammate_actions_oh, teammate_indices, vis_mask)
            logits, _, aux = agent_net.apply(
                params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates))
            return logits, aux
        return jax.vmap(forward_one)(flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)

    # Static scalar constants the jit should bake in from config
    _use_aux_loss = bool(config.use_aux_loss)

    @jax.jit
    def actor_update(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_actions,
                     flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff):
        def loss_fn(params):
            flat_logits, flat_aux = compute_logits_and_aux(
                params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
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
            return p_loss + config.entropy_coef * e_loss + aux_term
        loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
        return agent_state.apply_gradients(grads=grads), loss

    @jax.jit
    def critic_update(critic_state, flat_states, flat_returns):
        def loss_fn(params):
            vals = jax.vmap(lambda s: critic_net.apply(params, s))(flat_states)
            return ((vals - flat_returns) ** 2).mean()
        loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    # ---- Training loop ----
    rewards_history = []
    best_reward = float("-inf")

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

        # Compute effective aux_lambda for this iteration (annealing schedule).
        # aux_anneal_fraction=0.0 -> constant lambda. Else linear decay over
        # the first `aux_anneal_fraction * n_iterations` iterations, held at 0.
        if config.aux_anneal_fraction > 0.0:
            anneal_iters = max(1, int(config.aux_anneal_fraction * n_iterations))
            frac_remaining = max(0.0, 1.0 - iteration / anneal_iters)
            aux_lambda_eff = float(config.aux_lambda * frac_remaining)
        else:
            aux_lambda_eff = float(config.aux_lambda)

        # PPO epochs
        for _ in range(config.ppo_epochs):
            agent_state, a_loss = actor_update(
                agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                jnp.asarray(aux_lambda_eff))
            critic_state, c_loss = critic_update(critic_state, flat_states, returns_flat)

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
        "n_agents": int(config.n_agents),
        "n_actions": int(config.n_actions),
        "obs_dim": int(config.obs_dim),
        "hidden_dim": int(config.hidden_dim),
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
    args = parser.parse_args()

    # Build config from CLI knobs
    base_config = VABLConfig()._replace(
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
        aux_lambda=args.aux_lambda,
        stop_gradient_belief_to_aux=args.stop_gradient_belief,
        aux_anneal_fraction=args.aux_anneal_fraction,
        separate_aux_encoder=args.separate_aux_encoder,
    )

    train_vabl_vec(
        config=base_config,
        layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, log_interval=args.log_interval,
        save_path=args.save,
    )
