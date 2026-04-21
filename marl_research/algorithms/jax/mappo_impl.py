"""MAPPO implementing the JaxMARLAlgo interface.

Per-agent observation only (no teammate awareness, no belief learning).
Optional GRU for temporal context (we use it as the carry).
Centralized critic for value estimation.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.algo_interface import RolloutBatch


class MAPPOConfig(NamedTuple):
    embed_dim: int = 64
    hidden_dim: int = 128
    critic_hidden_dim: int = 128
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 10.0
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 520


class MAPPOActor(nn.Module):
    """Actor: obs → MLP → GRU → policy head."""
    config: MAPPOConfig

    @nn.compact
    def __call__(self, obs, hidden):
        cfg = self.config
        x = nn.Dense(cfg.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(cfg.hidden_dim)(x)
        x = nn.relu(x)

        gru = nn.GRUCell(features=cfg.hidden_dim)
        new_hidden, _ = gru(hidden, x)

        logits = nn.Dense(cfg.n_actions)(new_hidden)
        return logits, new_hidden


class MAPPOCritic(nn.Module):
    config: MAPPOConfig

    @nn.compact
    def __call__(self, state):
        cfg = self.config
        x = nn.Dense(cfg.critic_hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(cfg.critic_hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class MAPPOImpl:
    def __init__(self, config: MAPPOConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.obs_dim = config.obs_dim
        self.state_dim = config.obs_dim * config.n_agents
        self.carry_dim = config.hidden_dim

        self.actor = MAPPOActor(config)
        self.critic = MAPPOCritic(config)

    def init(self, rng):
        cfg = self.config
        rng_a, rng_c = jax.random.split(rng)
        actor_params = self.actor.init(rng_a, jnp.zeros(self.obs_dim), jnp.zeros(cfg.hidden_dim))
        critic_params = self.critic.init(rng_c, jnp.zeros(self.state_dim))

        agent_state = TrainState.create(
            apply_fn=self.actor.apply, params=actor_params,
            tx=optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.actor_lr, eps=1e-5)),
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=critic_params,
            tx=optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.critic_lr, eps=1e-5)),
        )
        return agent_state, critic_state

    def init_carry(self, n_envs):
        return jnp.zeros((n_envs, self.n_agents, self.config.hidden_dim))

    def step(self, agent_params, obs, carry, prev_actions, rng):
        """obs: [N, n_agents, obs_dim], carry: [N, n_agents, hidden]"""
        n_envs = obs.shape[0]

        def per_env(env_idx, env_obs, env_carry):
            rng_env = jax.random.fold_in(rng, env_idx)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                logits, new_h = self.actor.apply(agent_params, env_obs[i], env_carry[i])
                action = jax.random.categorical(rng_i, logits)
                lp = jax.nn.log_softmax(logits)[action]
                return action, new_h, lp
            return jax.vmap(per_agent)(jnp.arange(self.n_agents))

        return jax.vmap(per_env)(jnp.arange(n_envs), obs, carry)

    def get_value(self, critic_params, state):
        return jax.vmap(lambda s: self.critic.apply(critic_params, s))(state)

    def actor_loss(self, params, batch, clip_param, entropy_coef):
        B = batch.obs.shape[0]
        flat_obs = batch.obs.reshape(B * self.n_agents, self.obs_dim)
        flat_carry = batch.carry.reshape(B * self.n_agents, self.config.hidden_dim)

        def forward_one(o, c):
            logits, _ = self.actor.apply(params, o, c)
            return logits

        flat_logits = jax.vmap(forward_one)(flat_obs, flat_carry)
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

        return p_loss + entropy_coef * e_loss

    def critic_loss(self, params, batch):
        vals = jax.vmap(lambda s: self.critic.apply(params, s))(batch.states)
        return ((vals - batch.returns) ** 2).mean()
