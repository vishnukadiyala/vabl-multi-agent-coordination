"""VABL implementing the JaxMARLAlgo interface."""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.vabl import VABLConfig, VABLAgent, Critic
from marl_research.algorithms.jax.algo_interface import RolloutBatch


class VABLImpl:
    """VABL with the unified training interface."""

    def __init__(self, config: VABLConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.obs_dim = config.obs_dim
        self.state_dim = config.obs_dim * config.n_agents
        self.n_teammates = config.n_agents - 1
        self.carry_dim = config.hidden_dim  # belief size per agent

        self.agent_net = VABLAgent(config)
        self.critic_net = Critic(config.critic_hidden_dim)

        self.teammate_idx = jnp.array([
            [j for j in range(config.n_agents) if j != i]
            for i in range(config.n_agents)
        ])

    def init(self, rng):
        cfg = self.config
        rng_a, rng_c = jax.random.split(rng)
        agent_params = self.agent_net.init(
            rng_a,
            jnp.zeros(self.obs_dim),
            jnp.zeros(cfg.hidden_dim),
            jnp.zeros((self.n_teammates, self.n_actions)),
            jnp.ones(self.n_teammates),
        )
        critic_params = self.critic_net.init(rng_c, jnp.zeros(self.state_dim))

        agent_state = TrainState.create(
            apply_fn=self.agent_net.apply, params=agent_params,
            tx=optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.actor_lr, eps=1e-5)),
        )
        critic_state = TrainState.create(
            apply_fn=self.critic_net.apply, params=critic_params,
            tx=optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.critic_lr, eps=1e-5)),
        )
        return agent_state, critic_state

    def init_carry(self, n_envs):
        return jnp.zeros((n_envs, self.n_agents, self.config.hidden_dim))

    def step(self, agent_params, obs, carry, prev_actions, rng):
        """obs: [N, n_agents, obs_dim], carry: [N, n_agents, hidden]"""
        n_envs = obs.shape[0]

        def per_env(env_idx, env_obs, env_carry, env_prev):
            rng_env = jax.random.fold_in(rng, env_idx)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                t_oh = jax.nn.one_hot(env_prev[self.teammate_idx[i]], self.n_actions)
                logits, new_b, _ = self.agent_net.apply(
                    agent_params, env_obs[i], env_carry[i], t_oh, jnp.ones(self.n_teammates))
                action = jax.random.categorical(rng_i, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, new_b, lp
            return jax.vmap(per_agent)(jnp.arange(self.n_agents))

        return jax.vmap(per_env)(jnp.arange(n_envs), obs, carry, prev_actions)

    def get_value(self, critic_params, state):
        return jax.vmap(lambda s: self.critic_net.apply(critic_params, s))(state)

    def actor_loss(self, params, batch, clip_param, entropy_coef):
        # Flatten (B, n_agents, ...) -> (B*n_agents, ...)
        B = batch.obs.shape[0]
        flat_obs = batch.obs.reshape(B * self.n_agents, self.obs_dim)
        flat_carry = batch.carry.reshape(B * self.n_agents, self.config.hidden_dim)

        # Teammate actions for current step (for the network input)
        t_acts = batch.actions[:, self.teammate_idx]  # [B, n_agents, n_teammates]
        flat_t_oh = jax.nn.one_hot(t_acts, self.n_actions).reshape(
            B * self.n_agents, self.n_teammates, self.n_actions)

        def forward_one(o, c, t):
            logits, _, aux_logits = self.agent_net.apply(params, o, c, t, jnp.ones(self.n_teammates))
            return logits, aux_logits

        flat_logits, flat_aux = jax.vmap(forward_one)(flat_obs, flat_carry, flat_t_oh)
        logits = flat_logits.reshape(B, self.n_agents, self.n_actions)

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

        # Auxiliary loss: predict NEXT-step teammate actions from current belief
        # This matches the original PyTorch VABL implementation
        aux_logits = flat_aux.reshape(B, self.n_agents, self.n_teammates, self.n_actions)
        aux_lp = jax.nn.log_softmax(aux_logits)
        aux_targets = batch.next_actions[:, self.teammate_idx]  # [B, n_agents, n_teammates]
        aux_taken = jnp.take_along_axis(aux_lp, aux_targets[..., None], axis=-1).squeeze(-1)
        aux_loss = -aux_taken.mean()

        return p_loss + entropy_coef * e_loss + self.config.aux_lambda * aux_loss

    def critic_loss(self, params, batch):
        vals = jax.vmap(lambda s: self.critic_net.apply(params, s))(batch.states)
        return ((vals - batch.returns) ** 2).mean()
