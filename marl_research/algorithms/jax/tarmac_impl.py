"""TarMAC implementing the JaxMARLAlgo interface.

Targeted Multi-Agent Communication via attention. Each agent emits a message
and a key; other agents query against keys to selectively read messages.
No persistent recurrent state across timesteps (carry is unused).
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.algo_interface import RolloutBatch


class TarMACConfig(NamedTuple):
    embed_dim: int = 64
    hidden_dim: int = 128
    message_dim: int = 64
    key_dim: int = 64
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


class TarMACAgent(nn.Module):
    """Joint forward pass for all agents with one round of communication.

    Input: obs [n_agents, obs_dim]
    Output: logits [n_agents, n_actions]
    """
    config: TarMACConfig

    @nn.compact
    def __call__(self, obs):
        cfg = self.config
        n_agents = cfg.n_agents

        # Encode each agent's obs
        x = nn.Dense(cfg.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(cfg.hidden_dim)(x)
        x = nn.relu(x)  # [n_agents, hidden_dim]

        # Generate messages, keys, queries
        messages = nn.Dense(cfg.message_dim)(x)  # [n_agents, message_dim]
        keys = nn.Dense(cfg.key_dim)(x)
        queries = nn.Dense(cfg.key_dim)(x)

        # Scaled dot-product attention with self-masking
        attn_logits = jnp.matmul(queries, keys.T) / jnp.sqrt(cfg.key_dim)  # [n_agents, n_agents]
        # Mask self
        eye = jnp.eye(n_agents) * -1e8
        attn_logits = attn_logits + eye
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attended_msgs = attn_weights @ messages  # [n_agents, message_dim]

        # Combine with own state
        combined = jnp.concatenate([x, attended_msgs], axis=-1)
        combined = nn.Dense(cfg.hidden_dim)(combined)
        combined = nn.relu(combined)

        logits = nn.Dense(cfg.n_actions)(combined)
        return logits


class TarMACCritic(nn.Module):
    config: TarMACConfig

    @nn.compact
    def __call__(self, state):
        cfg = self.config
        x = nn.Dense(cfg.critic_hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(cfg.critic_hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class TarMACImpl:
    def __init__(self, config: TarMACConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.obs_dim = config.obs_dim
        self.state_dim = config.obs_dim * config.n_agents

        self.agent_net = TarMACAgent(config)
        self.critic_net = TarMACCritic(config)

    def init(self, rng):
        cfg = self.config
        rng_a, rng_c = jax.random.split(rng)
        agent_params = self.agent_net.init(rng_a, jnp.zeros((self.n_agents, self.obs_dim)))
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
        # TarMAC has no recurrent state — use a dummy carry of size 1 per agent
        return jnp.zeros((n_envs, self.n_agents, 1))

    def step(self, agent_params, obs, carry, prev_actions, rng):
        """obs: [N, n_agents, obs_dim], carry unused"""
        n_envs = obs.shape[0]

        def per_env(env_idx, env_obs):
            rng_env = jax.random.fold_in(rng, env_idx)
            logits = self.agent_net.apply(agent_params, env_obs)  # [n_agents, n_actions]
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                action = jax.random.categorical(rng_i, logits[i])
                lp = jax.nn.log_softmax(logits[i])[action]
                return action, lp
            actions, lps = jax.vmap(per_agent)(jnp.arange(self.n_agents))
            return actions, jnp.zeros((self.n_agents, 1)), lps

        return jax.vmap(per_env)(jnp.arange(n_envs), obs)

    def get_value(self, critic_params, state):
        return jax.vmap(lambda s: self.critic_net.apply(critic_params, s))(state)

    def actor_loss(self, params, batch, clip_param, entropy_coef):
        B = batch.obs.shape[0]

        def forward_one(env_obs):
            return self.agent_net.apply(params, env_obs)

        all_logits = jax.vmap(forward_one)(batch.obs)  # [B, n_agents, n_actions]

        lp = jax.nn.log_softmax(all_logits)
        nlp = jnp.take_along_axis(lp, batch.actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
        old_lp_sum = batch.log_probs.sum(axis=-1)

        ratio = jnp.clip(jnp.exp(nlp - old_lp_sum), 0.0, 5.0)
        s1 = ratio * batch.advantages
        s2 = jnp.clip(ratio, 1 - clip_param, 1 + clip_param) * batch.advantages
        p_loss = -jnp.minimum(s1, s2).mean()

        pr = jax.nn.softmax(all_logits)
        ent = -(pr * lp).sum(axis=-1).mean(axis=-1)
        e_loss = -ent.mean()

        return p_loss + entropy_coef * e_loss

    def critic_loss(self, params, batch):
        vals = jax.vmap(lambda s: self.critic_net.apply(params, s))(batch.states)
        return ((vals - batch.returns) ** 2).mean()
