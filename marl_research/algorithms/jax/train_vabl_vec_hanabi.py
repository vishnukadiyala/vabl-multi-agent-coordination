"""Vectorized VABL JAX training on Hanabi (turn-based, 2-player cooperative).

Adapted from train_vabl_vec.py for the Hanabi environment. Key differences
from the Overcooked trainer:

  * Turn-based: one agent acts per env step (determined by state.cur_player_idx).
    The other agent's sampled action is not applied by the env; we fill it
    with a placeholder and drop it from the log-likelihood accounting.
  * Legal-action masking: Hanabi forbids many actions at any given state
    (e.g., hinting a color your teammate doesn't have). We apply the
    legal-move mask from env.get_legal_moves(state) to both the policy
    logits (at rollout and at loss time via stored masks) and, implicitly,
    to the PPO ratio.
  * Variable episode length: Hanabi terminates when the deck is exhausted
    or all lives are lost. We roll out a fixed horizon and use the env's
    done mask to zero returns after episode end.
  * Aux target: predict the teammate's next action whenever it is their
    turn. We extract teammate-action-at-their-next-turn from the rollout
    trajectory.

Usage:
    python -m marl_research.algorithms.jax.train_vabl_vec_hanabi \\
        --episodes 25000 --horizon 80 --n-envs 64 --seed 0 \\
        --aux-lambda 0.05 --save results/vabl_hanabi_full_seed0.json
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


def _make_hanabi():
    from jaxmarl import make
    return make("hanabi")


def train_vabl_hanabi(
    config: VABLConfig = None,
    n_episodes: int = 5000,
    horizon: int = 80,
    n_envs: int = 32,
    seed: int = 0,
    log_interval: int = 10,
    save_path: str = None,
):
    if config is None:
        config = VABLConfig()

    env = _make_hanabi()

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    test_obs, test_state = env.reset(reset_rng)
    agent_names = sorted(env.agents)
    obs_dim = int(np.prod(test_obs[agent_names[0]].shape))
    n_agents = len(agent_names)
    n_actions = env.action_space(agent_names[0]).n
    n_teammates = n_agents - 1

    config = config._replace(obs_dim=obs_dim, n_agents=n_agents, n_actions=n_actions)

    print(f"VABL JAX Hanabi Training")
    print(f"  obs={obs_dim}, agents={n_agents}, actions={n_actions}")
    print(f"  N envs={n_envs}, horizon={horizon}, episodes={n_episodes}, seed={seed}")
    print(f"  use_attention={config.use_attention} use_aux_loss={config.use_aux_loss} "
          f"aux_lambda={config.aux_lambda} stop_gradient_belief_to_aux={config.stop_gradient_belief_to_aux}")

    # Init networks
    agent_net = VABLAgent(config)
    critic_net = Critic(config.critic_hidden_dim)

    rng, rng_a, rng_c, rng_v = jax.random.split(rng, 4)
    dummy_t_idx = jnp.arange(1, n_agents, dtype=jnp.int32)
    init_rngs = {"params": rng_a, "vae": rng_v}
    agent_params = agent_net.init(
        init_rngs, jnp.zeros(obs_dim), jnp.zeros(config.hidden_dim),
        jnp.zeros((n_teammates, n_actions)), dummy_t_idx, jnp.ones(n_teammates))
    critic_params = critic_net.init(rng_c, jnp.zeros(obs_dim * n_agents))

    agent_state = TrainState.create(
        apply_fn=agent_net.apply, params=agent_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip),
                       optax.adam(config.actor_lr, eps=1e-5)))
    critic_state = TrainState.create(
        apply_fn=critic_net.apply, params=critic_params,
        tx=optax.chain(optax.clip_by_global_norm(config.grad_clip),
                       optax.adam(config.critic_lr, eps=1e-5)))

    teammate_idx = jnp.array([[j for j in range(n_agents) if j != i] for i in range(n_agents)])

    vmap_reset = jax.jit(jax.vmap(env.reset))
    vmap_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))
    vmap_legal = jax.jit(jax.vmap(env.get_legal_moves))

    # ---- JIT functions ----

    @jax.jit
    def step_agents_vec(params, obs_batch, beliefs_batch, prev_actions_batch,
                        legal_masks_batch, rng):
        """Compute masked logits, sample legal actions per agent per env.

        obs_batch: [N, n_agents, obs_dim]
        beliefs_batch: [N, n_agents, hidden_dim]
        prev_actions_batch: [N, n_agents]
        legal_masks_batch: [N, n_agents, n_actions] (1 = legal)
        Returns: actions [N, n_agents], beliefs [N, n_agents, hidden_dim],
                 log_probs [N, n_agents], logits [N, n_agents, n_actions]
        """
        NEG_INF = -1e9

        def per_env(env_idx, env_obs, env_beliefs, env_prev_acts, env_masks):
            rng_env = jax.random.fold_in(rng, env_idx)

            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                rng_i_act, rng_i_vae = jax.random.split(rng_i)
                t_idx = teammate_idx[i]
                t_oh = jax.nn.one_hot(env_prev_acts[t_idx], n_actions)
                raw_logits, new_b, _, _ = agent_net.apply(
                    params, env_obs[i], env_beliefs[i], t_oh, t_idx,
                    jnp.ones(n_teammates), rngs={"vae": rng_i_vae})
                # Mask illegal actions. For self-player legal mask; if
                # env_masks[i] is all-zero (rare, e.g. defensive state), fall
                # back to action 0 to avoid a degenerate categorical.
                mask_i = env_masks[i].astype(jnp.float32)
                any_legal = mask_i.sum() > 0
                safe_mask = jnp.where(any_legal, mask_i, mask_i.at[0].set(1.0))
                masked_logits = jnp.where(safe_mask > 0, raw_logits, NEG_INF)
                action = jax.random.categorical(rng_i_act, masked_logits)
                lp = jax.nn.log_softmax(masked_logits)[action]
                return action, new_b, lp, masked_logits

            return jax.vmap(per_agent)(jnp.arange(n_agents))

        env_indices = jnp.arange(n_envs)
        return jax.vmap(per_env)(env_indices, obs_batch, beliefs_batch,
                                 prev_actions_batch, legal_masks_batch)

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
    def compute_logits_and_aux(params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                               flat_legal_masks, vae_rng):
        """Returns logits, aux_logits, kl_total. Applies legal-move mask to logits
        so PPO ratio and entropy are over the masked distribution."""
        NEG_INF = -1e9
        def forward_one(idx, obs_i, belief_i, t_oh_i, t_idx_i, mask_i):
            rng_i = jax.random.fold_in(vae_rng, idx)
            raw, _, aux, kl = agent_net.apply(
                params, obs_i, belief_i, t_oh_i, t_idx_i, jnp.ones(n_teammates),
                rngs={"vae": rng_i})
            mask_f = mask_i.astype(jnp.float32)
            any_legal = mask_f.sum() > 0
            safe_mask = jnp.where(any_legal, mask_f, mask_f.at[0].set(1.0))
            masked = jnp.where(safe_mask > 0, raw, NEG_INF)
            return masked, aux, kl
        idx = jnp.arange(flat_obs.shape[0])
        logits, aux, kls = jax.vmap(forward_one)(
            idx, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx, flat_legal_masks)
        return logits, aux, kls.sum()

    _use_aux_loss = bool(config.use_aux_loss)
    _use_vae_belief = bool(config.use_vae_belief)
    _vae_kl_weight = float(config.vae_kl_weight)

    @jax.jit
    def actor_update(agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                     flat_legal_masks, flat_actions, flat_next_t_actions,
                     flat_acting_mask, old_lp_sum, advantages_flat, aux_lambda_eff, vae_rng):
        """
        flat_acting_mask [N*H, n_agents]: 1.0 for agents who actually acted this
          step (i.e. were cur_player). Only these contribute to PPO loss; the
          other agent's sampled action was not applied by the env, so it is
          excluded from the policy-gradient term.
        """
        def loss_fn(params):
            flat_logits, flat_aux, kl_total = compute_logits_and_aux(
                params, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_legal_masks, vae_rng)
            B = flat_actions.shape[0]  # N*H
            logits = flat_logits.reshape(B, n_agents, n_actions)
            lp = jax.nn.log_softmax(logits)
            # Per-agent nlp; sum only over acting agents.
            nlp_per = jnp.take_along_axis(lp, flat_actions[..., None], axis=-1).squeeze(-1)  # [B, n_agents]
            nlp = (nlp_per * flat_acting_mask).sum(axis=-1)  # [B]
            ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
            s1 = ratio * advantages_flat
            s2 = jnp.clip(ratio, 1 - config.clip_param, 1 + config.clip_param) * advantages_flat
            p_loss = -jnp.minimum(s1, s2).mean()

            pr = jax.nn.softmax(logits)
            ent = -(pr * lp).sum(axis=-1)  # [B, n_agents]
            # Entropy: average only over acting agents (masked mean).
            ent_masked = (ent * flat_acting_mask).sum(axis=-1)
            acting_counts = flat_acting_mask.sum(axis=-1) + 1e-8
            e_loss = -(ent_masked / acting_counts).mean()

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

    @jax.jit
    def critic_update(critic_state, flat_states, flat_returns):
        def loss_fn(params):
            vals = jax.vmap(lambda s: critic_net.apply(params, s))(flat_states)
            return ((vals - flat_returns) ** 2).mean()
        loss, grads = jax.value_and_grad(loss_fn)(critic_state.params)
        return critic_state.apply_gradients(grads=grads), loss

    # ---- Training loop ----
    rewards_history = []  # per-episode final scores
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

        obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
        states_batch = obs_batch.reshape(n_envs, -1)

        # Rollout buffers
        buf_obs = jnp.zeros((horizon, n_envs, n_agents, obs_dim))
        buf_actions = jnp.zeros((horizon, n_envs, n_agents), dtype=jnp.int32)
        buf_log_probs = jnp.zeros((horizon, n_envs, n_agents))
        buf_values = jnp.zeros((horizon, n_envs))
        buf_beliefs = jnp.zeros((horizon, n_envs, n_agents, config.hidden_dim))
        buf_states = jnp.zeros((horizon, n_envs, obs_dim * n_agents))
        buf_rewards = jnp.zeros((horizon, n_envs))
        buf_dones = jnp.zeros((horizon, n_envs))
        buf_legal_masks = jnp.zeros((horizon, n_envs, n_agents, n_actions))
        buf_acting = jnp.zeros((horizon, n_envs, n_agents))  # [H, N, n_agents]

        ep_scores = jnp.zeros(n_envs)
        ep_dones = jnp.zeros(n_envs)  # once done, freeze the per-env score

        for step in range(horizon):
            rng, act_rng, step_rng = jax.random.split(rng, 3)

            legal_dict = vmap_legal(env_state_batch)  # {agent: [N, n_actions]}
            legal_masks = jnp.stack([legal_dict[a] for a in agent_names], axis=1)  # [N, n_agents, n_actions]

            values_batch = get_value_vec(critic_state.params, states_batch)
            actions_batch, new_beliefs, log_probs_batch, _ = step_agents_vec(
                agent_state.params, obs_batch, beliefs, prev_actions, legal_masks, act_rng)

            # cur_player_idx is a 2-vec in each env; argmax gives active agent.
            cur_player_idx_N = env_state_batch.cur_player_idx  # [N, n_agents] one-hot-ish
            # Convert to [N, n_agents] float mask where the active agent is 1.
            acting_mask_N = jax.nn.one_hot(
                jnp.argmax(cur_player_idx_N, axis=-1), n_agents)

            buf_obs = buf_obs.at[step].set(obs_batch)
            buf_actions = buf_actions.at[step].set(actions_batch)
            buf_log_probs = buf_log_probs.at[step].set(log_probs_batch)
            buf_values = buf_values.at[step].set(values_batch)
            buf_beliefs = buf_beliefs.at[step].set(beliefs)
            buf_states = buf_states.at[step].set(states_batch)
            buf_legal_masks = buf_legal_masks.at[step].set(legal_masks)
            buf_acting = buf_acting.at[step].set(acting_mask_N)

            # Environment step: Hanabi uses the action of whichever agent is
            # cur_player; other agents' actions are ignored. We pass them all
            # in but the env only consumes the active one.
            actions_dict_batch = {a: actions_batch[:, i] for i, a in enumerate(agent_names)}
            step_keys = jax.random.split(step_rng, n_envs)
            obs_dict_batch, env_state_batch, reward_dict_batch, done_dict_batch, _info = vmap_step(
                step_keys, env_state_batch, actions_dict_batch)

            # Shared cooperative reward: take agent_0's (they're equal).
            rewards = reward_dict_batch[agent_names[0]]
            dones = done_dict_batch["__all__"].astype(jnp.float32)

            # Mask rewards/dones for envs that had already finished.
            rewards = rewards * (1.0 - ep_dones)
            buf_rewards = buf_rewards.at[step].set(rewards)
            buf_dones = buf_dones.at[step].set(dones)

            # Accumulate per-episode score until done.
            ep_scores = ep_scores + rewards
            ep_dones = jnp.maximum(ep_dones, dones)

            obs_batch = jnp.stack([obs_dict_batch[a].reshape(n_envs, -1) for a in agent_names], axis=1)
            states_batch = obs_batch.reshape(n_envs, -1)
            beliefs = new_beliefs
            prev_actions = actions_batch

        # End of rollout. Record per-env final episode scores.
        for env_i in range(n_envs):
            s = float(ep_scores[env_i])
            rewards_history.append(s)
            best_reward = max(best_reward, s)
            total_episodes += 1

        rewards_NH = buf_rewards.transpose(1, 0)
        values_NH = buf_values.transpose(1, 0)
        dones_NH = buf_dones.transpose(1, 0)
        acting_NHA = buf_acting.transpose(1, 0, 2)  # [N, H, n_agents]

        advantages_NH, returns_NH = compute_gae_vec(rewards_NH, values_NH, dones_NH)

        flat_obs = buf_obs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, obs_dim)
        flat_beliefs = buf_beliefs.transpose(1, 0, 2, 3).reshape(n_envs * horizon * n_agents, config.hidden_dim)
        flat_actions = buf_actions.transpose(1, 0, 2).reshape(n_envs * horizon, n_agents)
        flat_acting_mask = acting_NHA.reshape(n_envs * horizon, n_agents)
        flat_legal_masks = buf_legal_masks.transpose(1, 0, 2, 3).reshape(
            n_envs * horizon * n_agents, n_actions)

        acts_NH = buf_actions.transpose(1, 0, 2)  # [N, H, n_agents]
        t_acts_NHA = acts_NH[:, :, teammate_idx]
        flat_t_oh = jax.nn.one_hot(t_acts_NHA, n_actions).reshape(
            n_envs * horizon * n_agents, n_teammates, n_actions)
        flat_t_idx = jnp.tile(teammate_idx[None, None, :, :],
                              (n_envs, horizon, 1, 1)).reshape(
            n_envs * horizon * n_agents, n_teammates).astype(jnp.int32)

        # Aux target = teammate's action at step t+1. In Hanabi only one agent
        # acts per step, so the teammate's "next action" is their next
        # decision whenever their turn comes up again. We approximate this by
        # using action at t+1 as in the Overcooked pipeline; it's the closest
        # one-step stand-in and keeps the target well-defined for every step.
        next_acts_NH = jnp.concatenate(
            [acts_NH[:, 1:], jnp.zeros((n_envs, 1, n_agents), dtype=jnp.int32)], axis=1)
        next_t_acts_NHA = next_acts_NH[:, :, teammate_idx]
        flat_next_t_actions = next_t_acts_NHA.reshape(
            n_envs * horizon * n_agents, n_teammates).astype(jnp.int32)

        # old_lp_sum: per-step, only acting agents contribute to the ratio.
        old_log_probs_NH = buf_log_probs.transpose(1, 0, 2)  # [N, H, n_agents]
        old_lp_sum_flat = (old_log_probs_NH * acting_NHA).sum(axis=-1).reshape(n_envs * horizon)

        advantages_flat = advantages_NH.reshape(n_envs * horizon)
        returns_flat = returns_NH.reshape(n_envs * horizon)
        flat_states = buf_states.transpose(1, 0, 2).reshape(n_envs * horizon, obs_dim * n_agents)

        if config.aux_anneal_fraction > 0.0:
            anneal_iters = max(1, int(config.aux_anneal_fraction * n_iterations))
            frac_remaining = max(0.0, 1.0 - iteration / anneal_iters)
            aux_lambda_eff = float(config.aux_lambda * frac_remaining)
        else:
            aux_lambda_eff = float(config.aux_lambda)

        for _ in range(config.ppo_epochs):
            rng, rng_au_vae = jax.random.split(rng)
            agent_state, a_loss = actor_update(
                agent_state, flat_obs, flat_beliefs, flat_t_oh, flat_t_idx,
                flat_legal_masks, flat_actions, flat_next_t_actions,
                flat_acting_mask, old_lp_sum_flat, advantages_flat,
                jnp.asarray(aux_lambda_eff), rng_au_vae)
            critic_state, c_loss = critic_update(critic_state, flat_states, returns_flat)

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            avg_r = np.mean(rewards_history[-n_envs * log_interval:])
            elapsed = time.time() - t0
            eps = total_episodes / elapsed
            print(f"  Iter {iteration+1:5d}/{n_iterations} | Eps {total_episodes:6d} | "
                  f"Score: {avg_r:5.2f} | Best: {best_reward:.2f} | {eps:.1f} ep/s")

    elapsed = time.time() - t0
    final = float(np.mean(rewards_history[-min(50, len(rewards_history)):]))
    print(f"\nDone in {elapsed:.0f}s ({total_episodes/elapsed:.1f} ep/s)")
    print(f"Final: {final:.2f}, Best: {best_reward:.2f}")

    cfg_record = {
        "env": "hanabi",
        "use_attention": bool(config.use_attention),
        "use_aux_loss": bool(config.use_aux_loss),
        "aux_lambda": float(config.aux_lambda),
        "stop_gradient_belief_to_aux": bool(config.stop_gradient_belief_to_aux),
        "aux_anneal_fraction": float(config.aux_anneal_fraction),
        "separate_aux_encoder": bool(config.separate_aux_encoder),
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
    parser.add_argument("--episodes", type=int, default=25000)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-aux-loss", action="store_true")
    parser.add_argument("--aux-lambda", type=float, default=0.05)
    parser.add_argument("--stop-gradient-belief", action="store_true")
    parser.add_argument("--aux-anneal-fraction", type=float, default=0.0)
    parser.add_argument("--use-vae-belief", action="store_true")
    parser.add_argument("--vae-kl-weight", type=float, default=0.005)
    args = parser.parse_args()

    cfg_kwargs = dict(
        use_attention=not args.no_attention,
        use_aux_loss=not args.no_aux_loss,
        aux_lambda=args.aux_lambda,
        stop_gradient_belief_to_aux=args.stop_gradient_belief,
        aux_anneal_fraction=args.aux_anneal_fraction,
    )
    if args.use_vae_belief:
        cfg_kwargs["use_vae_belief"] = True
        cfg_kwargs["vae_kl_weight"] = args.vae_kl_weight
    base_config = VABLConfig()._replace(**cfg_kwargs)

    train_vabl_hanabi(
        config=base_config,
        n_episodes=args.episodes, horizon=args.horizon,
        n_envs=args.n_envs, seed=args.seed, log_interval=args.log_interval,
        save_path=args.save,
    )
