"""Unified vectorized training loop for any JaxMARLAlgo implementation.

Works with VABL, MAPPO, TarMAC, AERIAL, etc. — any algorithm implementing the
JaxMARLAlgo interface in algo_interface.py.

Usage:
    from marl_research.algorithms.jax.vabl_impl import VABLImpl
    from marl_research.algorithms.jax.vabl import VABLConfig
    from marl_research.algorithms.jax.train_unified import train_unified

    algo = VABLImpl(VABLConfig())
    results = train_unified(algo, layout="cramped_room", n_episodes=5000, n_envs=64)
"""

import time
import json
from pathlib import Path
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from marl_research.algorithms.jax.algo_interface import RolloutBatch


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


def train_unified(
    algo_factory,
    layout: str = "cramped_room",
    n_episodes: int = 5000,
    horizon: int = 400,
    n_envs: int = 64,
    seed: int = 0,
    log_interval: int = 5,
    save_path: str = None,
    clip_param: float = 0.2,
    entropy_coef: float = 0.01,
    ppo_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    env_type: str = "overcooked",
    n_agents_simple: int = 4,
):
    """Train any JaxMARLAlgo on JaxMARL Overcooked or Simple Coordination.

    Args:
        algo_factory: Callable taking (obs_dim, n_agents, n_actions) and returning
                      a JaxMARLAlgo instance. This lets us detect env dims first.
        layout: Overcooked layout name (ignored if env_type='simple')
        n_episodes: Total training episodes
        horizon: Max steps per episode
        n_envs: Parallel environments
        seed: Random seed
        log_interval: Print metrics every N iterations
        save_path: Save final results here
        env_type: 'overcooked' or 'simple'
        n_agents_simple: Number of agents for simple env
    """
    if env_type == "simple":
        from marl_research.environments.simple_jax_env import SimpleJaxEnv
        env = SimpleJaxEnv(n_agents=n_agents_simple, episode_limit=horizon)
    else:
        Overcooked, available_layouts = _import_overcooked()
        jax_layout = LAYOUT_MAP.get(layout, layout)
        env = Overcooked(layout=available_layouts[jax_layout])

    # Detect env dimensions BEFORE creating the algorithm
    rng = jax.random.PRNGKey(seed)
    rng, probe_rng = jax.random.split(rng)
    test_obs, _ = env.reset(probe_rng)
    agent_names = sorted(env.agents)
    detected_obs_dim = int(np.prod(test_obs[agent_names[0]].shape))
    detected_n_agents = len(agent_names)
    detected_n_actions = env.action_space(agent_names[0]).n

    # Build algo with correct dimensions
    algo = algo_factory(detected_obs_dim, detected_n_agents, detected_n_actions)

    rng, init_rng = jax.random.split(rng)
    agent_state, critic_state = algo.init(init_rng)

    n_agents = algo.n_agents
    obs_dim = algo.obs_dim
    state_dim = algo.state_dim

    print(f"Training {type(algo).__name__} on {layout}")
    print(f"  obs={obs_dim}, agents={n_agents}, n_envs={n_envs}, horizon={horizon}")
    print(f"  episodes={n_episodes}, seed={seed}")

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    # JIT the algorithm methods
    step_fn = jax.jit(algo.step)
    value_fn = jax.jit(algo.get_value)

    @jax.jit
    def actor_update(agent_state, batch):
        loss, grads = jax.value_and_grad(
            lambda p: algo.actor_loss(p, batch, clip_param, entropy_coef)
        )(agent_state.params)
        return agent_state.apply_gradients(grads=grads), loss

    @jax.jit
    def critic_update(critic_state, batch):
        loss, grads = jax.value_and_grad(
            lambda p: algo.critic_loss(p, batch)
        )(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    @jax.jit
    def compute_gae_vec(rewards, values, dones):
        """rewards/values/dones: [N, H]. Returns advantages, returns of same shape."""
        def per_env(rew, val, dn):
            def body(gae, t):
                idx = horizon - 1 - t
                next_val = jnp.where(idx + 1 < horizon, val[jnp.minimum(idx + 1, horizon - 1)], 0.0)
                delta = rew[idx] + gamma * next_val * (1 - dn[idx]) - val[idx]
                gae = delta + gamma * gae_lambda * (1 - dn[idx]) * gae
                return gae, gae
            _, adv_rev = jax.lax.scan(body, 0.0, jnp.arange(horizon))
            return jnp.flip(adv_rev)
        adv = jax.vmap(per_env)(rewards, values, dones)
        ret = adv + values
        adv_norm = (adv - adv.mean()) / jnp.maximum(adv.std(), 1e-8)
        return adv_norm, ret

    rewards_history = []
    best_reward = float("-inf")
    n_iterations = (n_episodes + n_envs - 1) // n_envs
    total_episodes = 0

    print("  Compiling first iteration...")
    t0 = time.time()

    for iteration in range(n_iterations):
        rng, reset_rng = jax.random.split(rng)
        env_keys = jax.random.split(reset_rng, n_envs)
        obs_dict_batch, env_state_batch = vmap_reset(env_keys)
        carry = algo.init_carry(n_envs)
        prev_actions = jnp.zeros((n_envs, n_agents), dtype=jnp.int32)

        obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
        states_batch = obs_batch.reshape(n_envs, -1)

        # Preallocated rollout buffers
        carry_dim_per_agent = carry.shape[-1]
        buf_obs = jnp.zeros((horizon, n_envs, n_agents, obs_dim))
        buf_actions = jnp.zeros((horizon, n_envs, n_agents), dtype=jnp.int32)
        buf_log_probs = jnp.zeros((horizon, n_envs, n_agents))
        buf_values = jnp.zeros((horizon, n_envs))
        buf_carry = jnp.zeros((horizon, n_envs, n_agents, carry_dim_per_agent))
        buf_states = jnp.zeros((horizon, n_envs, state_dim))
        buf_rewards = jnp.zeros((horizon, n_envs))
        buf_dones = jnp.zeros((horizon, n_envs))

        ep_rewards = jnp.zeros(n_envs)

        for step in range(horizon):
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            values_batch = value_fn(critic_state.params, states_batch)
            actions_batch, new_carry, log_probs_batch = step_fn(
                agent_state.params, obs_batch, carry, prev_actions, act_rng)

            buf_obs = buf_obs.at[step].set(obs_batch)
            buf_actions = buf_actions.at[step].set(actions_batch)
            buf_log_probs = buf_log_probs.at[step].set(log_probs_batch)
            buf_values = buf_values.at[step].set(values_batch)
            buf_carry = buf_carry.at[step].set(carry)
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
            carry = new_carry
            prev_actions = actions_batch

        # Episode rewards
        ep_rewards_np = np.array(ep_rewards)
        for env_i in range(n_envs):
            r = float(ep_rewards_np[env_i])
            rewards_history.append(r)
            best_reward = max(best_reward, r)
            total_episodes += 1

        # Compute GAE per env
        rewards_NH = buf_rewards.transpose(1, 0)
        values_NH = buf_values.transpose(1, 0)
        dones_NH = buf_dones.transpose(1, 0)
        advantages_NH, returns_NH = compute_gae_vec(rewards_NH, values_NH, dones_NH)

        # Compute next-step actions for VABL aux loss
        # buf_actions: [H, N, n_agents] → shift t by 1 along H axis
        next_actions_HN = jnp.concatenate([
            buf_actions[1:],
            jnp.zeros((1, n_envs, n_agents), dtype=jnp.int32),
        ], axis=0)  # [H, N, n_agents]

        # Flatten to (N*H, ...) for PPO update
        B = n_envs * horizon
        batch = RolloutBatch(
            obs=buf_obs.transpose(1, 0, 2, 3).reshape(B, n_agents, obs_dim),
            actions=buf_actions.transpose(1, 0, 2).reshape(B, n_agents),
            next_actions=next_actions_HN.transpose(1, 0, 2).reshape(B, n_agents),
            log_probs=buf_log_probs.transpose(1, 0, 2).reshape(B, n_agents),
            carry=buf_carry.transpose(1, 0, 2, 3).reshape(B, n_agents, carry_dim_per_agent),
            states=buf_states.transpose(1, 0, 2).reshape(B, state_dim),
            advantages=advantages_NH.reshape(B),
            returns=returns_NH.reshape(B),
        )

        for _ in range(ppo_epochs):
            agent_state, a_loss = actor_update(agent_state, batch)
            critic_state, c_loss = critic_update(critic_state, batch)

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
        "algorithm": type(algo).__name__,
        "layout": layout,
        "rewards": rewards_history,
        "final_reward": final,
        "best_reward": best_reward,
        "elapsed": elapsed,
        "n_episodes": total_episodes,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["vabl", "vabl_v2", "mappo", "tarmac", "aerial", "commnet"], default="vabl")
    parser.add_argument("--env-type", choices=["overcooked", "simple"], default="overcooked")
    parser.add_argument("--n-agents-simple", type=int, default=4)
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save", default=None)
    # AERIAL-specific aux-loss knobs (reviewer-requested fix-path experiment)
    parser.add_argument("--aux-lambda", type=float, default=0.0,
                        help="AERIAL aux-loss weight (0 = off)")
    parser.add_argument("--no-aux-loss", action="store_true",
                        help="Force aux loss off even if aux-lambda > 0")
    parser.add_argument("--stop-gradient-belief", action="store_true",
                        help="Stop aux gradients from flowing into the belief encoder (AERIAL)")
    args = parser.parse_args()

    if args.algo == "vabl":
        from marl_research.algorithms.jax.vabl import VABLConfig
        from marl_research.algorithms.jax.vabl_impl import VABLImpl
        factory = lambda od, na, nact: VABLImpl(VABLConfig()._replace(obs_dim=od, n_agents=na, n_actions=nact))
    elif args.algo == "vabl_v2":
        from marl_research.algorithms.jax.vabl_v2 import VABLv2Config, VABLv2Impl
        factory = lambda od, na, nact: VABLv2Impl(VABLv2Config()._replace(obs_dim=od, n_agents=na, n_actions=nact))
    elif args.algo == "mappo":
        from marl_research.algorithms.jax.mappo_impl import MAPPOConfig, MAPPOImpl
        factory = lambda od, na, nact: MAPPOImpl(MAPPOConfig()._replace(obs_dim=od, n_agents=na, n_actions=nact))
    elif args.algo == "tarmac":
        from marl_research.algorithms.jax.tarmac_impl import TarMACConfig, TarMACImpl
        factory = lambda od, na, nact: TarMACImpl(TarMACConfig()._replace(obs_dim=od, n_agents=na, n_actions=nact))
    elif args.algo == "aerial":
        from marl_research.algorithms.jax.aerial_impl import AERIALConfig, AERIALImpl
        _use_aux = (args.aux_lambda > 0.0) and not args.no_aux_loss
        factory = lambda od, na, nact: AERIALImpl(AERIALConfig()._replace(
            obs_dim=od, n_agents=na, n_actions=nact,
            aux_lambda=args.aux_lambda,
            use_aux_loss=_use_aux,
            stop_gradient_belief_to_aux=args.stop_gradient_belief,
        ))
    elif args.algo == "commnet":
        from marl_research.algorithms.jax.commnet_impl import CommNetConfig, CommNetImpl
        factory = lambda od, na, nact: CommNetImpl(CommNetConfig()._replace(obs_dim=od, n_agents=na, n_actions=nact))
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    train_unified(
        factory, layout=args.layout, n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, log_interval=args.log_interval,
        save_path=args.save, env_type=args.env_type, n_agents_simple=args.n_agents_simple,
    )
