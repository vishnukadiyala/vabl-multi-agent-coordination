"""Vectorized training for the generic Transformer MARL agent.

Tests the gradient-interference pathology on an independent architecture
(NOT VABL, NOT AERIAL). If the pathology reproduces here, it's proven to
be an architectural-class finding.

The training loop is identical to train_vabl_vec.py except:
  - Uses TransformerMARLAgent instead of VABLAgent
  - No GRU hidden state carried between timesteps — the transformer
    processes each timestep independently (CLS token is the "belief")
  - Teammate actions from the CURRENT step are used as input tokens
    (the transformer sees the present, not the past)

CLI flags mirror train_vabl_vec.py exactly.

Usage:
    python -m marl_research.algorithms.jax.train_transformer_vec \
        --layout asymmetric_advantages --episodes 25000 --n-envs 64 \
        --seed 0 --aux-lambda 0.05 --save results/transformer/full_seed0.json
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

from marl_research.algorithms.jax.transformer_agent import (
    TransformerConfig,
    TransformerMARLAgent,
    TransformerCritic,
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
}


def train_transformer_vec(
    config: TransformerConfig = None,
    layout: str = "asymmetric_advantages",
    n_episodes: int = 25000,
    horizon: int = 400,
    n_envs: int = 64,
    seed: int = 0,
    log_interval: int = 10,
    save_path: str = None,
):
    if config is None:
        config = TransformerConfig()

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

    print(f"Transformer MARL Agent on {layout}")
    print(f"  obs={obs_dim}, agents={n_agents}, actions={n_actions}")
    print(f"  N envs={n_envs}, horizon={horizon}, episodes={n_episodes}, seed={seed}")
    print(f"  use_attention={config.use_attention}, use_aux_loss={config.use_aux_loss}")
    print(f"  aux_lambda={config.aux_lambda}, aux_anneal={config.aux_anneal_fraction}")
    print(f"  Architecture: {config.num_layers}-layer transformer, {config.num_heads} heads, dim {config.embed_dim}")

    # Init
    agent_net = TransformerMARLAgent(config)
    critic_net = TransformerCritic(config.critic_hidden_dim)

    rng, rng_a, rng_c = jax.random.split(rng, 3)
    agent_params = agent_net.init(
        rng_a,
        jnp.zeros(obs_dim),
        jnp.zeros((n_teammates, n_actions)),
        jnp.ones(n_teammates),
    )
    critic_params = critic_net.init(rng_c, jnp.zeros(obs_dim * n_agents))

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(agent_params))
    print(f"  Agent params: {n_params:,}")

    agent_state = TrainState.create(
        apply_fn=agent_net.apply, params=agent_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.actor_lr, eps=1e-5)))
    critic_state = TrainState.create(
        apply_fn=critic_net.apply, params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.critic_lr, eps=1e-5)))

    teammate_idx = jnp.array([[j for j in range(n_agents) if j != i] for i in range(n_agents)])

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    # NOTE: No hidden state carried between steps — transformer is stateless
    @jax.jit
    def step_agents_vec(params, obs_batch, prev_actions_batch, rng):
        """No beliefs/carry — transformer processes each step independently."""
        def per_env(env_idx, env_obs, env_prev_acts):
            rng_env = jax.random.fold_in(rng, env_idx)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                t_idx = teammate_idx[i]
                t_oh = jax.nn.one_hot(env_prev_acts[t_idx], n_actions)
                logits, _, _ = agent_net.apply(
                    params, env_obs[i], t_oh, jnp.ones(n_teammates))
                action = jax.random.categorical(rng_i, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, lp
            return jax.vmap(per_agent)(jnp.arange(n_agents))
        return jax.vmap(per_env)(jnp.arange(n_envs), obs_batch, prev_actions_batch)

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

    _use_aux_loss = bool(config.use_aux_loss)

    @jax.jit
    def actor_update(agent_state, flat_obs, flat_t_oh, flat_actions,
                     flat_next_t_actions, old_lp_sum, advantages_flat, aux_lambda_eff):
        def loss_fn(params):
            def forward_one(obs_i, t_oh_i):
                logits, _, aux = agent_net.apply(
                    params, obs_i, t_oh_i, jnp.ones(n_teammates))
                return logits, aux
            flat_logits, flat_aux = jax.vmap(forward_one)(flat_obs, flat_t_oh)

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
    t0 = time.time()
    n_iterations = (n_episodes + n_envs - 1) // n_envs
    total_episodes = 0

    print("  Starting (compiling on first iteration)...")

    for iteration in range(n_iterations):
        rng, reset_rng = jax.random.split(rng)
        env_keys = jax.random.split(reset_rng, n_envs)
        obs_dict_batch, env_state_batch = vmap_reset(env_keys)
        prev_actions = jnp.zeros((n_envs, n_agents), dtype=jnp.int32)
        obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
        states_batch = obs_batch.reshape(n_envs, -1)

        buf_obs = jnp.zeros((horizon, n_envs, n_agents, obs_dim))
        buf_actions = jnp.zeros((horizon, n_envs, n_agents), dtype=jnp.int32)
        buf_log_probs = jnp.zeros((horizon, n_envs, n_agents))
        buf_values = jnp.zeros((horizon, n_envs))
        buf_states = jnp.zeros((horizon, n_envs, obs_dim * n_agents))
        buf_rewards = jnp.zeros((horizon, n_envs))
        buf_dones = jnp.zeros((horizon, n_envs))
        ep_rewards = jnp.zeros(n_envs)

        for step in range(horizon):
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            values_batch = get_value_vec(critic_state.params, states_batch)
            actions_batch, log_probs_batch = step_agents_vec(
                agent_state.params, obs_batch, prev_actions, act_rng)

            buf_obs = buf_obs.at[step].set(obs_batch)
            buf_actions = buf_actions.at[step].set(actions_batch)
            buf_log_probs = buf_log_probs.at[step].set(log_probs_batch)
            buf_values = buf_values.at[step].set(values_batch)
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

        # Flatten
        flat_obs = buf_obs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, obs_dim)
        flat_actions = buf_actions.transpose(1, 0, 2).reshape(n_envs * horizon, n_agents)
        acts_NH = buf_actions.transpose(1, 0, 2)
        t_acts_NHA = acts_NH[:, :, teammate_idx]
        flat_t_oh = jax.nn.one_hot(t_acts_NHA, n_actions).reshape(
            n_envs * horizon * n_agents, n_teammates, n_actions)
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

        # Aux lambda
        if config.aux_anneal_fraction > 0.0:
            anneal_iters = max(1, int(config.aux_anneal_fraction * n_iterations))
            frac_remaining = max(0.0, 1.0 - iteration / anneal_iters)
            aux_lambda_eff = float(config.aux_lambda * frac_remaining)
        else:
            aux_lambda_eff = float(config.aux_lambda)

        for _ in range(config.ppo_epochs):
            agent_state, a_loss = actor_update(
                agent_state, flat_obs, flat_t_oh, flat_actions,
                flat_next_t_actions, old_lp_sum_flat, advantages_flat,
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

    results = {
        "rewards": rewards_history, "final_reward": final,
        "best_reward": best_reward, "elapsed": elapsed,
        "config": {
            "architecture": "transformer_2layer",
            "use_attention": bool(config.use_attention),
            "use_aux_loss": bool(config.use_aux_loss),
            "aux_lambda": float(config.aux_lambda),
            "stop_gradient_belief_to_aux": bool(config.stop_gradient_belief_to_aux),
            "aux_anneal_fraction": float(config.aux_anneal_fraction),
            "embed_dim": config.embed_dim, "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "layout": layout, "n_envs": n_envs, "horizon": horizon,
            "n_episodes": n_episodes, "seed": seed,
        },
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
    parser.add_argument("--episodes", type=int, default=25000)
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
    args = parser.parse_args()

    cfg = TransformerConfig()._replace(
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
        aux_lambda=args.aux_lambda,
        stop_gradient_belief_to_aux=args.stop_gradient_belief,
        aux_anneal_fraction=args.aux_anneal_fraction,
    )

    train_transformer_vec(
        config=cfg,
        layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, log_interval=args.log_interval,
        save_path=args.save,
    )
