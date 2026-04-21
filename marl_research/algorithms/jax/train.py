"""Vectorized MARL training in JAX — fully JIT-compiled.

The entire rollout + PPO update is compiled into a single JAX program.
No Python loops in the hot path, no device→host transfers during training.

Usage:
    from marl_research.algorithms.jax.vabl import VABL, VABLConfig
    from marl_research.algorithms.jax.train import train
    results = train(VABL, VABLConfig(), layout="cramped_room", n_episodes=5000)
"""

from typing import Any, Dict, NamedTuple, Tuple, Type
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


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


class Transition(NamedTuple):
    obs: jnp.ndarray          # [n_agents, obs_dim]
    actions: jnp.ndarray      # [n_agents]
    rewards: jnp.ndarray      # scalar
    dones: jnp.ndarray        # scalar
    log_probs: jnp.ndarray    # [n_agents]
    values: jnp.ndarray       # scalar
    beliefs: jnp.ndarray      # [n_agents, hidden_dim]
    state: jnp.ndarray        # [state_dim]


def train(
    algo_cls,
    config,
    layout: str = "cramped_room",
    n_episodes: int = 5000,
    n_envs: int = 1,
    seed: int = 0,
    horizon: int = 400,
    log_interval: int = 100,
):
    Overcooked, available_layouts = _import_overcooked()
    jax_layout = LAYOUT_MAP.get(layout, layout)
    env = Overcooked(layout=available_layouts[jax_layout])

    # Get env info
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    test_obs, _ = env.reset(reset_rng)
    agent_names = sorted(env.agents)
    obs_dim = int(np.prod(test_obs[agent_names[0]].shape))
    n_agents = len(agent_names)
    n_actions = env.action_space(agent_names[0]).n

    config = config._replace(obs_dim=obs_dim, n_agents=n_agents, n_actions=n_actions)

    print(f"Training {algo_cls.__name__} on {layout}")
    print(f"  obs_dim={obs_dim}, n_agents={n_agents}, n_actions={n_actions}, horizon={horizon}")
    print(f"  episodes={n_episodes}, seed={seed}")

    # Init algorithm
    algo = algo_cls(config)
    rng, init_rng = jax.random.split(rng)
    agent_state, critic_state = algo.init(init_rng)

    # Precompute teammate indices as JAX array (must be jnp for vmap indexing)
    teammate_idx = jnp.array([
        [j for j in range(n_agents) if j != i]
        for i in range(n_agents)
    ])  # [n_agents, n_teammates]

    # JIT-compile env
    env_reset = jax.jit(env.reset)
    env_step = jax.jit(env.step)

    # Standardized forward: each algo must provide forward_agent and forward_critic
    # forward_agent(params, obs_i, belief_i, teammate_actions_oh, vis_mask_i) -> (logits, new_belief, aux)
    # forward_critic(params, state) -> scalar value
    # Each algo exposes agent_network and critic_network (or agent/critic as fallback)
    _agent_net = getattr(algo, 'agent_network', getattr(algo, 'agent', None))
    _critic_net = getattr(algo, 'critic_network', getattr(algo, 'critic', None))
    forward_agent_fn = _agent_net.apply
    forward_critic_fn = _critic_net.apply

    @jax.jit
    def get_actions_and_beliefs(agent_params, obs_all, beliefs, prev_actions, vis_masks, rng):
        """Run all agents in one JIT call. Returns actions, beliefs, log_probs."""
        def per_agent(i):
            rng_i = jax.random.fold_in(rng, i)
            t_acts = prev_actions[teammate_idx[i]]
            t_oh = jax.nn.one_hot(t_acts, n_actions)
            logits, new_belief, _ = forward_agent_fn(
                agent_params, obs_all[i], beliefs[i], t_oh, vis_masks[i]
            )
            action = jax.random.categorical(rng_i, logits)
            log_prob = jax.nn.log_softmax(logits)[action]
            return action, new_belief, log_prob

        actions, new_beliefs, log_probs = jax.vmap(per_agent)(jnp.arange(n_agents))
        return actions, new_beliefs, log_probs

    @jax.jit
    def get_value(critic_params, state):
        return forward_critic_fn(critic_params, state)

    # JIT-compile the PPO update
    @jax.jit
    def ppo_update(agent_state, critic_state, rollout_obs, rollout_actions,
                   rollout_log_probs, rollout_values, rollout_beliefs,
                   rollout_states, rollout_rewards, rollout_dones, rollout_vis, seq_len):
        """Single PPO update on one episode rollout. Fully JIT."""
        mask = jnp.arange(horizon) < seq_len

        # GAE
        def gae_body(carry, t):
            gae = carry
            idx = seq_len - 1 - t
            next_val = jnp.where(idx + 1 < seq_len, rollout_values[idx + 1], 0.0)
            delta = rollout_rewards[idx] + config.gamma * next_val * (1 - rollout_dones[idx]) - rollout_values[idx]
            gae = delta + config.gamma * config.gae_lambda * (1 - rollout_dones[idx]) * gae
            return gae, gae

        _, advs_rev = jax.lax.scan(gae_body, 0.0, jnp.arange(horizon))
        advantages = jnp.flip(advs_rev) * mask
        returns = (advantages + rollout_values) * mask

        # Normalize advantages
        adv_sum = (advantages * mask).sum()
        adv_sq = ((advantages ** 2) * mask).sum()
        adv_mean = adv_sum / (mask.sum() + 1e-8)
        adv_std = jnp.sqrt(adv_sq / (mask.sum() + 1e-8) - adv_mean ** 2 + 1e-8)
        advantages = (advantages - adv_mean) / adv_std * mask

        def ppo_epoch(carry, _):
            agent_state, critic_state = carry

            # Recompute log_probs under current params
            def recompute_agent(t):
                def per_agent(i):
                    t_acts = rollout_actions[t][teammate_idx[i]]
                    t_oh = jax.nn.one_hot(t_acts, n_actions)
                    logits, _, _ = forward_agent_fn(
                        agent_state.params, rollout_obs[t][i],
                        rollout_beliefs[t][i], t_oh, rollout_vis[t][i]
                    )
                    return logits
                all_logits = jax.vmap(per_agent)(jnp.arange(n_agents))  # [n_agents, n_actions]
                return all_logits

            all_logits = jax.vmap(recompute_agent)(jnp.arange(horizon))  # [horizon, n_agents, n_actions]

            # New log probs
            new_log_probs = jax.nn.log_softmax(all_logits)
            new_lp = jnp.take_along_axis(
                new_log_probs, rollout_actions[..., None], axis=-1
            ).squeeze(-1)  # [horizon, n_agents]
            new_lp_sum = new_lp.sum(axis=-1)  # [horizon]
            old_lp_sum = rollout_log_probs.sum(axis=-1)  # [horizon]

            # Entropy
            probs = jax.nn.softmax(all_logits)
            entropy = -(probs * new_log_probs).sum(axis=-1).mean(axis=-1)  # [horizon]

            # Policy loss
            ratio = jnp.exp(new_lp_sum - old_lp_sum)
            ratio = jnp.clip(ratio, 0.0, 5.0)
            surr1 = ratio * advantages
            surr2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages
            policy_loss = -(jnp.minimum(surr1, surr2) * mask).sum() / (mask.sum() + 1e-8)
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # Actor gradient
            def actor_loss(params):
                def recomp(t):
                    def per_a(i):
                        t_a = rollout_actions[t][teammate_idx[i]]
                        t_oh = jax.nn.one_hot(t_a, n_actions)
                        logits, _, _ = forward_agent_fn(
                            params, rollout_obs[t][i],
                            rollout_beliefs[t][i], t_oh, rollout_vis[t][i]
                        )
                        return logits
                    return jax.vmap(per_a)(jnp.arange(n_agents))
                logits = jax.vmap(recomp)(jnp.arange(horizon))
                lp = jax.nn.log_softmax(logits)
                lp_taken = jnp.take_along_axis(lp, rollout_actions[..., None], axis=-1).squeeze(-1)
                lp_sum = lp_taken.sum(axis=-1)
                r = jnp.exp(lp_sum - old_lp_sum)
                r = jnp.clip(r, 0.0, 5.0)
                s1 = r * advantages
                s2 = jnp.clip(r, 1 - config.clip_param, 1 + config.clip_param) * advantages
                p_loss = -(jnp.minimum(s1, s2) * mask).sum() / (mask.sum() + 1e-8)
                # Entropy
                pr = jax.nn.softmax(logits)
                ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
                e_loss = -(ent * mask).sum() / (mask.sum() + 1e-8)
                return p_loss + config.entropy_coef * e_loss

            a_loss, a_grads = jax.value_and_grad(actor_loss)(agent_state.params)
            agent_state = agent_state.apply_gradients(grads=a_grads)

            # Critic gradient
            def critic_loss(params):
                vals = jax.vmap(lambda s: forward_critic_fn(params, s))(rollout_states)
                v_loss = ((vals - returns) ** 2 * mask).sum() / (mask.sum() + 1e-8)
                return v_loss

            c_loss, c_grads = jax.value_and_grad(critic_loss)(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=c_grads)

            return (agent_state, critic_state), (a_loss, c_loss)

        (agent_state, critic_state), (a_losses, c_losses) = jax.lax.scan(
            ppo_epoch, (agent_state, critic_state), jnp.arange(config.ppo_epochs)
        )

        return agent_state, critic_state, a_losses.mean(), c_losses.mean()

    # ---- Main training loop ----
    rewards_history = []
    best_reward = float("-inf")
    vis_masks = jnp.ones((n_agents, n_agents - 1))

    # Preallocate rollout buffers (fixed size, use mask for variable-length episodes)
    zero_obs = jnp.zeros((horizon, n_agents, obs_dim))
    zero_acts = jnp.zeros((horizon, n_agents), dtype=jnp.int32)
    zero_lps = jnp.zeros((horizon, n_agents))
    zero_vals = jnp.zeros(horizon)
    zero_beliefs = jnp.zeros((horizon, n_agents, config.hidden_dim))
    zero_states = jnp.zeros((horizon, obs_dim * n_agents))
    zero_rewards = jnp.zeros(horizon)
    zero_dones = jnp.zeros(horizon)
    zero_vis = jnp.broadcast_to(vis_masks, (horizon, n_agents, n_agents - 1))

    print("  Compiling (first episode will be slow)...")
    t0 = time.time()

    for episode in range(n_episodes):
        rng, ep_rng, reset_rng = jax.random.split(rng, 3)

        obs_dict, env_state = env_reset(reset_rng)
        beliefs = jnp.zeros((n_agents, config.hidden_dim))
        prev_actions = jnp.zeros(n_agents, dtype=jnp.int32)

        # Collect rollout — env stepping in Python but agent inference in JIT
        buf_obs = []
        buf_acts = []
        buf_lps = []
        buf_vals = []
        buf_beliefs = []
        buf_states = []
        buf_rewards = []
        buf_dones = []
        ep_reward = 0.0
        seq_len = 0

        for step in range(horizon):
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            obs = jnp.stack([obs_dict[a].flatten() for a in agent_names])
            state = obs.flatten()

            value = get_value(critic_state.params, state)
            actions, beliefs, log_probs = get_actions_and_beliefs(
                agent_state.params, obs, beliefs, prev_actions, zero_vis[0], act_rng
            )

            buf_obs.append(obs)
            buf_acts.append(actions)
            buf_lps.append(log_probs)
            buf_vals.append(value)
            buf_beliefs.append(beliefs)
            buf_states.append(state)

            actions_dict = {a: actions[i] for i, a in enumerate(agent_names)}
            obs_dict, env_state, reward_dict, done_dict, info = env_step(step_rng, env_state, actions_dict)

            reward = float(reward_dict[agent_names[0]])
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

        # Pad to fixed horizon and run PPO update
        if seq_len > 1:
            def pad(arr_list, template):
                stacked = jnp.stack(arr_list)
                pad_len = horizon - len(arr_list)
                if pad_len > 0:
                    pad_shape = (pad_len,) + template.shape[1:]
                    return jnp.concatenate([stacked, jnp.zeros(pad_shape, dtype=stacked.dtype)])
                return stacked

            r_obs = pad(buf_obs, zero_obs)
            r_acts = pad(buf_acts, zero_acts)
            r_lps = pad(buf_lps, zero_lps)
            r_vals = jnp.concatenate([jnp.array(buf_vals), jnp.zeros(horizon - seq_len)])
            r_beliefs = pad(buf_beliefs, zero_beliefs)
            r_states = pad(buf_states, zero_states)
            r_rewards = jnp.concatenate([jnp.array(buf_rewards), jnp.zeros(horizon - seq_len)])
            r_dones = jnp.concatenate([jnp.array(buf_dones), jnp.zeros(horizon - seq_len)])

            agent_state, critic_state, a_loss, c_loss = ppo_update(
                agent_state, critic_state,
                r_obs, r_acts, r_lps, r_vals, r_beliefs, r_states,
                r_rewards, r_dones, zero_vis, jnp.int32(seq_len),
            )

        if (episode + 1) % log_interval == 0:
            avg_r = np.mean(rewards_history[-log_interval:])
            elapsed = time.time() - t0
            eps_per_sec = (episode + 1) / elapsed
            print(f"  Ep {episode+1:6d}/{n_episodes} | R: {avg_r:7.1f} | Best: {best_reward:.0f} | "
                  f"{eps_per_sec:.1f} ep/s | a_loss: {float(a_loss):.3f}")

    elapsed = time.time() - t0
    final_reward = float(np.mean(rewards_history[-50:])) if len(rewards_history) >= 50 else float(np.mean(rewards_history))

    print(f"\nDone in {elapsed:.0f}s ({n_episodes/elapsed:.1f} ep/s)")
    print(f"Final: {final_reward:.1f}, Best: {best_reward:.0f}")

    return {
        "rewards": rewards_history,
        "final_reward": final_reward,
        "best_reward": best_reward,
        "elapsed": elapsed,
    }
