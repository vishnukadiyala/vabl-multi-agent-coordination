"""Gradient-diagnostic variant of train_vabl_vec.py.

Runs a single A_full training run and decomposes the actor gradient into
PPO and auxiliary components at the shared encoder level every LOG_EVERY
iterations. Logs:

    ||grad_PPO||   — L2 norm of grad(L_PPO + L_entropy) over encoder params
    ||grad_aux||   — L2 norm of grad(lambda * L_aux) over encoder params
    cosine         — cosine similarity between the two gradient vectors
    ratio          — ||grad_aux|| / ||grad_PPO||

Produces a JSON with per-iteration diagnostic arrays + the usual reward
history. A separate plotting script (paper/plot_gradient_diagnostics.py)
turns this into the paper figure.

This script is intentionally a COPY of the core training loop (not a
modification) so it doesn't risk breaking the canonical training code.
It reproduces the same training dynamics as train_vabl_vec.py with
identical hyperparameters.

Usage:
    python -m marl_research.algorithms.jax.train_vabl_vec_diagnostic \
        --layout asymmetric_advantages --episodes 25000 --n-envs 64 \
        --seed 0 --save results/gradient_diagnostics.json
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

LOG_EVERY = 50  # Log gradient diagnostics every N iterations


def _flatten_params(pytree):
    """Flatten a param pytree into a single 1D vector."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.concatenate([l.reshape(-1) for l in leaves])


def _grad_norm(grad_pytree):
    """L2 norm of a gradient pytree."""
    flat = _flatten_params(grad_pytree)
    return jnp.sqrt(jnp.sum(flat ** 2))


def _cosine_sim(grad_a, grad_b):
    """Cosine similarity between two gradient pytrees."""
    a = _flatten_params(grad_a)
    b = _flatten_params(grad_b)
    dot = jnp.sum(a * b)
    norm_a = jnp.sqrt(jnp.sum(a ** 2)) + 1e-8
    norm_b = jnp.sqrt(jnp.sum(b ** 2)) + 1e-8
    return dot / (norm_a * norm_b)


def run_diagnostic(
    config: VABLConfig = None,
    layout: str = "asymmetric_advantages",
    n_episodes: int = 25000,
    horizon: int = 400,
    n_envs: int = 64,
    seed: int = 0,
    save_path: str = None,
):
    if config is None:
        config = VABLConfig()  # Default: Full VABL (attention + aux at constant lambda=0.05)

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

    print(f"Gradient Diagnostic Run on {layout}")
    print(f"  obs={obs_dim}, agents={n_agents}, actions={n_actions}")
    print(f"  N envs={n_envs}, horizon={horizon}, episodes={n_episodes}, seed={seed}")
    print(f"  Logging gradient decomposition every {LOG_EVERY} iterations")

    # Init networks (identical to train_vabl_vec.py)
    agent_net = VABLAgent(config)
    critic_net = Critic(config.critic_hidden_dim)

    rng, rng_a, rng_c = jax.random.split(rng, 3)
    dummy_t_idx = jnp.arange(1, n_agents, dtype=jnp.int32)
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

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    @jax.jit
    def step_agents_vec(params, obs_batch, beliefs_batch, prev_actions_batch, rng):
        def per_env(env_idx, env_obs, env_beliefs, env_prev_acts):
            rng_env = jax.random.fold_in(rng, env_idx)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                t_idx = teammate_idx[i]
                t_oh = jax.nn.one_hot(env_prev_acts[t_idx], n_actions)
                logits, new_b, _ = agent_net.apply(
                    params, env_obs[i], env_beliefs[i], t_oh, t_idx, jnp.ones(n_teammates))
                action = jax.random.categorical(rng_i, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, new_b, lp
            return jax.vmap(per_agent)(jnp.arange(n_agents))
        return jax.vmap(per_env)(jnp.arange(n_envs), obs_batch, beliefs_batch, prev_actions_batch)

    @jax.jit
    def get_value_vec(critic_params, states_batch):
        return jax.vmap(lambda s: critic_net.apply(critic_params, s))(states_batch)

    @jax.jit
    def compute_gae_vec(rewards, values, dones):
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
        adv_mean = advantages.mean()
        adv_std = jnp.maximum(advantages.std(), 1e-8)
        advantages = (advantages - adv_mean) / adv_std
        return advantages, returns

    @jax.jit
    def compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx):
        def forward_one(obs_i, belief_i, t_oh_i, t_idx_i):
            logits, _, aux = agent_net.apply(
                params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates))
            return logits, aux
        return jax.vmap(forward_one)(flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)

    # ---- DIAGNOSTIC: separate PPO and aux loss functions ----

    def ppo_loss_fn(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_actions, old_lp_sum, advantages_flat):
        """PPO + entropy loss only (no auxiliary)."""
        flat_logits, _ = compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
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

    def aux_loss_fn(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                    flat_next_t_actions, aux_lambda_eff):
        """Auxiliary loss only (scaled by lambda)."""
        _, flat_aux = compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
        aux_lp = jax.nn.log_softmax(flat_aux, axis=-1)
        aux_taken = jnp.take_along_axis(
            aux_lp, flat_next_t_actions[..., None], axis=-1
        ).squeeze(-1)
        aux_loss = -aux_taken.mean()
        return aux_lambda_eff * aux_loss

    @jax.jit
    def compute_gradient_diagnostics(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                                      flat_actions, flat_next_t_actions, old_lp_sum,
                                      advantages_flat, aux_lambda_eff):
        """Compute PPO and aux gradients separately, return norms + cosine."""
        grad_ppo = jax.grad(ppo_loss_fn)(
            params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
            flat_actions, old_lp_sum, advantages_flat)
        grad_aux = jax.grad(aux_loss_fn)(
            params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
            flat_next_t_actions, aux_lambda_eff)

        norm_ppo = _grad_norm(grad_ppo)
        norm_aux = _grad_norm(grad_aux)
        cosine = _cosine_sim(grad_ppo, grad_aux)
        ratio = norm_aux / (norm_ppo + 1e-8)

        return norm_ppo, norm_aux, cosine, ratio

    # ---- Standard actor update (same as train_vabl_vec.py) ----

    @jax.jit
    def actor_update(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_actions,
                     flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff):
        def loss_fn(params):
            flat_logits, flat_aux = compute_logits_and_aux(
                params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)
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
            return p_loss + config.entropy_coef * e_loss + aux_lambda_eff * aux_loss
        loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
        return agent_state.apply_gradients(grads=grads), loss

    @jax.jit
    def critic_update(critic_state, flat_states, flat_returns):
        def loss_fn(params):
            vals = jax.vmap(lambda s: critic_net.apply(params, s))(flat_states)
            return ((vals - flat_returns) ** 2).mean()
        loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    # ---- Training loop with diagnostic logging ----
    rewards_history = []
    best_reward = float("-inf")
    diagnostic_log = {
        'iterations': [],
        'norm_ppo': [],
        'norm_aux': [],
        'cosine_sim': [],
        'grad_ratio': [],
        'episode_reward': [],
    }

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
        obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
        states_batch = obs_batch.reshape(n_envs, -1)

        buf_obs = jnp.zeros((horizon, n_envs, n_agents, obs_dim))
        buf_actions = jnp.zeros((horizon, n_envs, n_agents), dtype=jnp.int32)
        buf_log_probs = jnp.zeros((horizon, n_envs, n_agents))
        buf_values = jnp.zeros((horizon, n_envs))
        buf_beliefs = jnp.zeros((horizon, n_envs, n_agents, config.hidden_dim))
        buf_states = jnp.zeros((horizon, n_envs, obs_dim * n_agents))
        buf_rewards = jnp.zeros((horizon, n_envs))
        buf_dones = jnp.zeros((horizon, n_envs))
        ep_rewards = jnp.zeros(n_envs)

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

            actions_dict_batch = {a: actions_batch[:, i] for i, a in enumerate(agent_names)}
            step_keys = jax.random.split(step_rng, n_envs)
            obs_dict_batch, env_state_batch, reward_dict_batch, done_dict_batch, info = vmap_step(
                step_keys, env_state_batch, actions_dict_batch)

            rewards = reward_dict_batch[agent_names[0]]
            if "shaped_reward" in info:
                for a in agent_names:
                    if a in info["shaped_reward"]:
                        rewards = rewards + info["shaped_reward"][a]
            dones = done_dict_batch["__all__"].astype(jnp.float32)

            buf_rewards = buf_rewards.at[step].set(rewards)
            buf_dones = buf_dones.at[step].set(dones)
            ep_rewards = ep_rewards + rewards

            obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
            states_batch = obs_batch.reshape(n_envs, -1)
            beliefs = new_beliefs
            prev_actions = actions_batch

        for env_i in range(n_envs):
            r = float(ep_rewards[env_i])
            rewards_history.append(r)
            best_reward = max(best_reward, r)
            total_episodes += 1

        # GAE
        rewards_NH = buf_rewards.transpose(1, 0)
        values_NH = buf_values.transpose(1, 0)
        dones_NH = buf_dones.transpose(1, 0)
        advantages_NH, returns_NH = compute_gae_vec(rewards_NH, values_NH, dones_NH)

        # Flatten for PPO
        flat_obs = buf_obs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, obs_dim)
        flat_beliefs = buf_beliefs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, config.hidden_dim)
        flat_actions = buf_actions.transpose(1, 0, 2).reshape(n_envs * horizon, n_agents)
        acts_NH = buf_actions.transpose(1, 0, 2)
        t_acts_NHA = acts_NH[:, :, teammate_idx]
        flat_t_oh = jax.nn.one_hot(t_acts_NHA, n_actions).reshape(
            n_envs * horizon * n_agents, n_teammates, n_actions)
        flat_t_idx = jnp.tile(teammate_idx[None, None, :, :],
                               (n_envs, horizon, 1, 1)).reshape(
            n_envs * horizon * n_agents, n_teammates).astype(jnp.int32)
        next_acts_NH = jnp.concatenate(
            [acts_NH[:, 1:], jnp.zeros((n_envs, 1, n_agents), dtype=jnp.int32)], axis=1)
        next_t_acts_NHA = next_acts_NH[:, :, teammate_idx]
        flat_next_t_actions = next_t_acts_NHA.reshape(
            n_envs * horizon * n_agents, n_teammates).astype(jnp.int32)
        old_log_probs_NH = buf_log_probs.transpose(1, 0, 2)
        old_lp_sum_flat = old_log_probs_NH.sum(axis=-1).reshape(n_envs * horizon)
        advantages_flat = advantages_NH.reshape(n_envs * horizon)
        returns_flat = returns_NH.reshape(n_envs * horizon)
        flat_states = buf_states.transpose(1, 0, 2).reshape(n_envs * horizon, obs_dim * n_agents)

        aux_lambda_eff = jnp.asarray(float(config.aux_lambda))

        # ---- DIAGNOSTIC: decompose gradients before the PPO update ----
        if iteration % LOG_EVERY == 0:
            norm_ppo, norm_aux, cosine, ratio = compute_gradient_diagnostics(
                agent_state.params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_actions, flat_next_t_actions, old_lp_sum_flat,
                advantages_flat, aux_lambda_eff)
            avg_r = float(np.mean(rewards_history[-n_envs:]))
            diagnostic_log['iterations'].append(int(iteration))
            diagnostic_log['norm_ppo'].append(float(norm_ppo))
            diagnostic_log['norm_aux'].append(float(norm_aux))
            diagnostic_log['cosine_sim'].append(float(cosine))
            diagnostic_log['grad_ratio'].append(float(ratio))
            diagnostic_log['episode_reward'].append(avg_r)

            print(f"  Iter {iteration:5d}/{n_iterations} | Eps {total_episodes:6d} | "
                  f"R: {avg_r:7.1f} | ||PPO||: {float(norm_ppo):.4f} | "
                  f"||aux||: {float(norm_aux):.4f} | cos: {float(cosine):.4f} | "
                  f"ratio: {float(ratio):.2f}")

        # PPO epochs (standard training, identical to train_vabl_vec.py)
        for _ in range(config.ppo_epochs):
            agent_state, a_loss = actor_update(
                agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_actions, flat_next_t_actions, old_lp_sum_flat, advantages_flat,
                aux_lambda_eff)
            critic_state, c_loss = critic_update(critic_state, flat_states, returns_flat)

    elapsed = time.time() - t0
    final = float(np.mean(rewards_history[-min(50, len(rewards_history)):]))
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Final: {final:.1f}, Best: {best_reward:.0f}")
    print(f"Logged {len(diagnostic_log['iterations'])} diagnostic checkpoints")

    result = {
        "rewards": rewards_history,
        "final_reward": final,
        "best_reward": best_reward,
        "elapsed": elapsed,
        "diagnostic": diagnostic_log,
        "config": {
            "layout": layout,
            "n_episodes": n_episodes,
            "horizon": horizon,
            "n_envs": n_envs,
            "seed": seed,
            "aux_lambda": float(config.aux_lambda),
            "use_attention": True,
            "use_aux_loss": True,
            "log_every": LOG_EVERY,
        },
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {save_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="asymmetric_advantages")
    parser.add_argument("--episodes", type=int, default=25000)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="results/gradient_diagnostics.json")
    parser.add_argument("--aux-lambda", type=float, default=0.05)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-aux-loss", action="store_true")
    args = parser.parse_args()

    cfg = VABLConfig()._replace(
        aux_lambda=args.aux_lambda,
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
    )

    run_diagnostic(
        config=cfg,
        layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, save_path=args.save,
    )
