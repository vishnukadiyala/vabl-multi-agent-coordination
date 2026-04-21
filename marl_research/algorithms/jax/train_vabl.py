"""VABL JAX training — split JIT for fast compilation.

Each JIT function is small enough to compile in seconds, not minutes.
Rollout: env stepping in Python, agent inference JIT'd.
PPO update: per-epoch JIT'd (not nested inside lax.scan).

Usage:
    python -m marl_research.algorithms.jax.train_vabl --layout cramped_room --episodes 5000
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

from marl_research.algorithms.jax.vabl import VABLConfig, VABLAgent, Critic, GRUCell


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


def train_vabl(
    config: VABLConfig = None,
    layout: str = "cramped_room",
    n_episodes: int = 5000,
    horizon: int = 400,
    seed: int = 0,
    log_interval: int = 100,
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

    print(f"VABL JAX Training on {layout}")
    print(f"  obs={obs_dim}, agents={n_agents}, actions={n_actions}, horizon={horizon}")
    print(f"  episodes={n_episodes}, seed={seed}")

    # Init
    agent_net = VABLAgent(config)
    critic_net = Critic(config.critic_hidden_dim)

    rng, rng_a, rng_c = jax.random.split(rng, 3)
    agent_params = agent_net.init(rng_a, jnp.zeros(obs_dim), jnp.zeros(config.hidden_dim),
                                   jnp.zeros((n_teammates, n_actions)), jnp.ones(n_teammates))
    critic_params = critic_net.init(rng_c, jnp.zeros(obs_dim * n_agents))

    agent_state = TrainState.create(
        apply_fn=agent_net.apply, params=agent_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.actor_lr, eps=1e-5)))
    critic_state = TrainState.create(
        apply_fn=critic_net.apply, params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip), optax.adam(config.critic_lr, eps=1e-5)))

    teammate_idx = jnp.array([[j for j in range(n_agents) if j != i] for i in range(n_agents)])
    env_reset = jax.jit(env.reset)
    env_step = jax.jit(env.step)

    # --- Small JIT functions ---

    @jax.jit
    def step_agents(params, obs_all, beliefs, prev_actions, rng):
        def per_agent(i):
            rng_i = jax.random.fold_in(rng, i)
            t_oh = jax.nn.one_hot(prev_actions[teammate_idx[i]], n_actions)
            logits, new_b, _ = agent_net.apply(params, obs_all[i], beliefs[i], t_oh, jnp.ones(n_teammates))
            action = jax.random.categorical(rng_i, logits)
            lp = jax.nn.log_softmax(logits)[action]
            return action, new_b, lp
        return jax.vmap(per_agent)(jnp.arange(n_agents))

    @jax.jit
    def compute_value(params, state):
        return critic_net.apply(params, state)

    @jax.jit
    def compute_gae(rewards, values, dones, seq_len):
        mask = (jnp.arange(rewards.shape[0]) < seq_len).astype(jnp.float32)
        def body(gae, t):
            idx = jnp.clip(seq_len - 1 - t, 0, rewards.shape[0] - 1)
            nxt = jnp.clip(idx + 1, 0, rewards.shape[0] - 1)
            nv = jnp.where(idx + 1 < seq_len, values[nxt], 0.0)
            delta = rewards[idx] + config.gamma * nv * (1 - dones[idx]) - values[idx]
            gae = delta + config.gamma * config.gae_lambda * (1 - dones[idx]) * gae
            return gae, gae
        _, adv_rev = jax.lax.scan(body, 0.0, jnp.arange(rewards.shape[0]))
        adv = jnp.flip(adv_rev) * mask
        ret = (adv + values) * mask
        # Normalize
        am = (adv * mask).sum() / (mask.sum() + 1e-8)
        av = ((adv - am)**2 * mask).sum() / (mask.sum() + 1e-8)
        adv = (adv - am) / jnp.sqrt(av + 1e-8) * mask
        return adv, ret, mask

    # Precompute flat teammate action one-hots for all (t, agent) pairs
    # This avoids nested vmap — single flat batch instead
    @jax.jit
    def prepare_flat_inputs(obs_seq, acts_seq, beliefs_seq):
        """Flatten [T, n_agents, ...] to [T*n_agents, ...] with precomputed teammate actions."""
        T = obs_seq.shape[0]
        # For each (t, i), get teammate actions as one-hot
        # teammate_idx: [n_agents, n_teammates]
        # acts_seq: [T, n_agents]
        # For agent i at time t: teammate_acts = acts_seq[t, teammate_idx[i]]
        t_acts = acts_seq[:, teammate_idx]  # [T, n_agents, n_teammates]
        t_oh = jax.nn.one_hot(t_acts, n_actions)  # [T, n_agents, n_teammates, n_actions]
        vis = jnp.ones((T, n_agents, n_teammates))

        # Flatten to [T*n_agents, ...]
        flat_obs = obs_seq.reshape(T * n_agents, obs_dim)
        flat_beliefs = beliefs_seq.reshape(T * n_agents, config.hidden_dim)
        flat_t_oh = t_oh.reshape(T * n_agents, n_teammates, n_actions)
        flat_vis = vis.reshape(T * n_agents, n_teammates)
        return flat_obs, flat_beliefs, flat_t_oh, flat_vis

    @jax.jit
    def compute_all_logits(params, flat_obs, flat_beliefs, flat_t_oh, flat_vis):
        """Single vmap over flat batch of (T*n_agents) inputs."""
        def forward_one(obs_i, belief_i, t_oh_i, vis_i):
            logits, _, _ = agent_net.apply(params, obs_i, belief_i, t_oh_i, vis_i)
            return logits
        return jax.vmap(forward_one)(flat_obs, flat_beliefs, flat_t_oh, flat_vis)  # [T*n_agents, n_actions]

    @jax.jit
    def actor_update(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_vis,
                     acts_seq, old_lp_sum, advantages, mask):
        T = mask.shape[0]
        def loss_fn(params):
            flat_logits = compute_all_logits(params, flat_obs, flat_beliefs, flat_t_oh, flat_vis)
            all_logits = flat_logits.reshape(T, n_agents, n_actions)
            lp = jax.nn.log_softmax(all_logits)
            nlp = jnp.take_along_axis(lp, acts_seq[..., None], axis=-1).squeeze(-1).sum(axis=-1)
            ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
            s1 = ratio * advantages
            s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages
            p_loss = -(jnp.minimum(s1, s2) * mask).sum() / (mask.sum() + 1e-8)
            pr = jax.nn.softmax(all_logits)
            ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
            e_loss = -(ent * mask).sum() / (mask.sum() + 1e-8)
            return p_loss + config.entropy_coef * e_loss
        loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)
        return agent_state.apply_gradients(grads=grads), loss

    @jax.jit
    def critic_update(critic_state, states_seq, returns, mask):
        def loss_fn(params):
            vals = jax.vmap(lambda s: critic_net.apply(params, s))(states_seq)
            return ((vals - returns)**2 * mask).sum() / (mask.sum() + 1e-8)
        loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    # --- Training loop ---
    rewards_history = []
    best_reward = float("-inf")

    print("  Starting (first episode compiles JIT functions)...")
    t0 = time.time()

    for episode in range(n_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs_dict, env_state = env_reset(reset_rng)
        beliefs = jnp.zeros((n_agents, config.hidden_dim))
        prev_actions = jnp.zeros(n_agents, dtype=jnp.int32)

        buf_obs, buf_acts, buf_lps, buf_vals = [], [], [], []
        buf_beliefs, buf_states, buf_rewards, buf_dones = [], [], [], []
        ep_reward = 0.0
        seq_len = 0

        for step in range(horizon):
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            obs = jnp.stack([obs_dict[a].flatten() for a in agent_names])
            state = obs.flatten()

            value = compute_value(critic_state.params, state)
            actions, beliefs, log_probs = step_agents(agent_state.params, obs, beliefs, prev_actions, act_rng)

            buf_obs.append(obs)
            buf_acts.append(actions)
            buf_lps.append(log_probs)
            buf_vals.append(value)
            buf_beliefs.append(beliefs)
            buf_states.append(state)

            actions_dict = {a: actions[i] for i, a in enumerate(agent_names)}
            obs_dict, env_state, reward_dict, done_dict, info = env_step(step_rng, env_state, actions_dict)
            # Sparse + shaped reward
            reward = float(reward_dict[agent_names[0]])
            if "shaped_reward" in info:
                for a in agent_names:
                    if a in info["shaped_reward"]:
                        reward += float(info["shaped_reward"][a])
            done = bool(done_dict["__all__"])
            ep_reward += reward
            buf_rewards.append(reward)
            buf_dones.append(float(done))
            prev_actions = actions
            seq_len += 1
            if done:
                break

        rewards_history.append(ep_reward)
        best_reward = max(best_reward, ep_reward)

        # PPO update
        if seq_len > 1:
            def pad(lst, trail_shape):
                s = jnp.stack(lst)
                pad_n = horizon - len(lst)
                if pad_n > 0:
                    s = jnp.concatenate([s, jnp.zeros((pad_n,) + trail_shape, dtype=s.dtype)])
                return s

            r_obs = pad(buf_obs, (n_agents, obs_dim))
            r_acts = pad(buf_acts, (n_agents,))
            r_lps = pad(buf_lps, (n_agents,))
            r_vals = jnp.concatenate([jnp.array(buf_vals), jnp.zeros(horizon - seq_len)])
            r_beliefs = pad(buf_beliefs, (n_agents, config.hidden_dim))
            r_states = pad(buf_states, (obs_dim * n_agents,))
            r_rewards = jnp.concatenate([jnp.array(buf_rewards), jnp.zeros(horizon - seq_len)])
            r_dones = jnp.concatenate([jnp.array(buf_dones), jnp.zeros(horizon - seq_len)])

            advantages, returns, mask = compute_gae(r_rewards, r_vals, r_dones, jnp.int32(seq_len))
            old_lp_sum = r_lps.sum(axis=-1)

            # Precompute flat inputs once (outside gradient)
            flat_obs, flat_beliefs, flat_t_oh, flat_vis = prepare_flat_inputs(r_obs, r_acts, r_beliefs)

            for _ in range(config.ppo_epochs):
                agent_state, a_loss = actor_update(
                    agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_vis,
                    r_acts, old_lp_sum, advantages, mask)
                critic_state, c_loss = critic_update(critic_state, r_states, returns, mask)

        if (episode + 1) % log_interval == 0:
            avg_r = np.mean(rewards_history[-log_interval:])
            elapsed = time.time() - t0
            eps = (episode + 1) / elapsed
            print(f"  Ep {episode+1:6d}/{n_episodes} | R: {avg_r:7.1f} | Best: {best_reward:.0f} | {eps:.1f} ep/s")

    elapsed = time.time() - t0
    final = float(np.mean(rewards_history[-50:])) if len(rewards_history) >= 50 else float(np.mean(rewards_history))
    print(f"\nDone in {elapsed:.0f}s ({n_episodes/elapsed:.1f} ep/s)")
    print(f"Final: {final:.1f}, Best: {best_reward:.0f}")

    results = {"rewards": rewards_history, "final_reward": final, "best_reward": best_reward, "elapsed": elapsed}

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    train_vabl(layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
               seed=args.seed, log_interval=args.log_interval, save_path=args.save)
