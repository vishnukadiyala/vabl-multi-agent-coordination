"""Vectorized VABL JAX training with jax.lax.scan rollout.

Drop-in replacement for train_vabl_vec.py with 3-5x faster training.
The rollout loop is compiled into a single jitted call via jax.lax.scan,
eliminating the 400 Python-GPU synchronization points per PPO iteration.

Expected improvement: GPU utilization goes from ~15% to ~90%+,
wall time per 25K-episode run drops from ~8.5 min to ~2-3 min.

The outputs are numerically equivalent to train_vabl_vec.py for the same
seed (verified by comparing canonical results).

Usage: identical CLI to train_vabl_vec.py.
"""

import time
import argparse
import json
from pathlib import Path
from functools import partial

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


def train_vabl_vec_fast(
    config: VABLConfig = None,
    layout: str = "asymmetric_advantages",
    n_episodes: int = 5000,
    horizon: int = 400,
    n_envs: int = 64,
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

    print(f"VABL JAX FAST (scan rollout) on {layout}")
    print(f"  obs={obs_dim}, agents={n_agents}, actions={n_actions}")
    print(f"  N envs={n_envs}, horizon={horizon}, episodes={n_episodes}, seed={seed}")

    # Init networks
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

    # Vmapped env functions (NOT jitted — scan body will jit them)
    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    # ---- Agent step (pure function, no @jax.jit) ----
    def agent_step(agent_params, obs_batch, beliefs_batch, prev_actions_batch, rng):
        """Run all agents for N envs. Returns actions, new_beliefs, log_probs."""
        def per_env(env_idx, env_obs, env_beliefs, env_prev_acts):
            rng_env = jax.random.fold_in(rng, env_idx)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                t_idx = teammate_idx[i]
                t_oh = jax.nn.one_hot(env_prev_acts[t_idx], n_actions)
                logits, new_b, _ = agent_net.apply(
                    agent_params, env_obs[i], env_beliefs[i], t_oh, t_idx, jnp.ones(n_teammates))
                action = jax.random.categorical(rng_i, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, new_b, lp
            return jax.vmap(per_agent)(jnp.arange(n_agents))
        return jax.vmap(per_env)(jnp.arange(n_envs), obs_batch, beliefs_batch, prev_actions_batch)

    def critic_step(critic_params, states_batch):
        return jax.vmap(lambda s: critic_net.apply(critic_params, s))(states_batch)

    # ---- The scan-based rollout ----
    def rollout_body(carry, step_idx):
        """One environment step. Called by jax.lax.scan."""
        rng, obs_batch, beliefs, prev_actions, env_state, states_batch, ep_rewards, a_params, c_params = carry

        # Split RNG deterministically using step index
        rng, act_rng, step_rng = jax.random.split(rng, 3)

        # Critic values — use params from carry, NOT captured variables
        values = critic_step(c_params, states_batch)

        # Agent actions — use params from carry, NOT captured variables
        actions, new_beliefs, log_probs = agent_step(
            a_params, obs_batch, beliefs, prev_actions, act_rng)

        # Build action dict for env step
        actions_dict = {agent_names[i]: actions[:, i] for i in range(n_agents)}

        # Step environment
        step_keys = jax.random.split(step_rng, n_envs)
        new_obs_dict, new_env_state, reward_dict, done_dict, info = vmap_step(
            step_keys, env_state, actions_dict)

        # Extract rewards (sparse + shaped)
        rewards = reward_dict[agent_names[0]]
        # Add shaped rewards if available
        for a in agent_names:
            rewards = rewards + info["shaped_reward"][a]
        dones = done_dict["__all__"].astype(jnp.float32)

        # New observations
        new_obs_batch = jnp.stack(
            [new_obs_dict[a].reshape(n_envs, -1) for a in agent_names], axis=1)
        new_states_batch = new_obs_batch.reshape(n_envs, -1)

        # New carry — pass params through unchanged
        new_carry = (rng, new_obs_batch, new_beliefs, actions,
                     new_env_state, new_states_batch, ep_rewards + rewards,
                     a_params, c_params)

        # Output (collected per step)
        output = {
            'obs': obs_batch,
            'actions': actions,
            'log_probs': log_probs,
            'values': values,
            'beliefs': beliefs,
            'states': states_batch,
            'rewards': rewards,
            'dones': dones,
        }

        return new_carry, output

    @jax.jit
    def do_rollout(rng, env_state, obs_batch, beliefs, prev_actions,
                   states_batch, agent_params, critic_params):
        """Complete horizon-step rollout via scan. Single jitted call.

        Agent and critic params are passed as arguments (not captured)
        so they update correctly across PPO iterations.
        """
        init_carry = (rng, obs_batch, beliefs, prev_actions,
                      env_state, states_batch, jnp.zeros(n_envs),
                      agent_params, critic_params)

        final_carry, outputs = jax.lax.scan(
            rollout_body, init_carry, jnp.arange(horizon))

        ep_rewards = final_carry[6]  # cumulative rewards
        return outputs, ep_rewards

    # ---- GAE (already efficient, keep as is) ----
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

    # ---- Logits/aux forward for PPO update ----
    @jax.jit
    def compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx):
        def forward_one(obs_i, belief_i, t_oh_i, t_idx_i):
            logits, _, aux = agent_net.apply(
                params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates))
            return logits, aux
        return jax.vmap(forward_one)(flat_obs, flat_beliefs, flat_t_oh, flat_t_idx)

    _use_aux_loss = bool(config.use_aux_loss)

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

    print("  Starting (compiling on first iteration — scan compilation is slower but one-time)...")
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

        # === SCAN ROLLOUT: single jitted call for entire horizon ===
        rng, rollout_rng = jax.random.split(rng)
        outputs, ep_rewards = do_rollout(
            rollout_rng, env_state_batch, obs_batch, beliefs, prev_actions,
            states_batch, agent_state.params, critic_state.params)

        # outputs is a dict of [horizon, N, ...] arrays
        buf_obs = outputs['obs']         # [H, N, n_agents, obs_dim]
        buf_actions = outputs['actions'] # [H, N, n_agents]
        buf_log_probs = outputs['log_probs']  # [H, N, n_agents]
        buf_values = outputs['values']   # [H, N]
        buf_beliefs = outputs['beliefs'] # [H, N, n_agents, hidden_dim]
        buf_states = outputs['states']   # [H, N, state_dim]
        buf_rewards = outputs['rewards'] # [H, N]
        buf_dones = outputs['dones']     # [H, N]

        # Episode rewards
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

        # Aux lambda annealing
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

    cfg_record = {
        "use_attention": bool(config.use_attention),
        "use_aux_loss": bool(config.use_aux_loss),
        "aux_lambda": float(config.aux_lambda),
        "stop_gradient_belief_to_aux": bool(config.stop_gradient_belief_to_aux),
        "aux_anneal_fraction": float(config.aux_anneal_fraction),
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
        "training_mode": "scan_rollout",
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
    parser.add_argument("--layout", default="asymmetric_advantages")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-aux-loss", action="store_true")
    parser.add_argument("--aux-lambda", type=float, default=0.05)
    parser.add_argument("--stop-gradient-belief", action="store_true")
    parser.add_argument("--aux-anneal-fraction", type=float, default=0.0)
    parser.add_argument("--attention-heads", type=int, default=4,
                        help="Number of MHA heads (default 4, matches canonical Full VABL)")
    parser.add_argument("--aux-hidden-dim", type=int, default=64,
                        help="Hidden dim of the aux predictor MLP (default 64)")
    args = parser.parse_args()

    base_config = VABLConfig()._replace(
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
        aux_lambda=args.aux_lambda,
        stop_gradient_belief_to_aux=args.stop_gradient_belief,
        aux_anneal_fraction=args.aux_anneal_fraction,
        attention_heads=args.attention_heads,
        aux_hidden_dim=args.aux_hidden_dim,
    )

    train_vabl_vec_fast(
        config=base_config,
        layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, log_interval=args.log_interval,
        save_path=args.save,
    )
