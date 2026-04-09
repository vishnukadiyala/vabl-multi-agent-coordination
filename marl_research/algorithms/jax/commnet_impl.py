"""CommNet implementing the JaxMARLAlgo interface.

Broadcast communication: each agent's hidden state is averaged with all
other agents' hidden states. Simpler than TarMAC (no targeting).
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.algo_interface import RolloutBatch


class CommNetConfig(NamedTuple):
    hidden_dim: int = 128
    comm_rounds: int = 2
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


class CommNetAgent(nn.Module):
    config: CommNetConfig

    @nn.compact
    def __call__(self, obs):
        cfg = self.config
        n_agents = cfg.n_agents

        # Encode obs
        h = nn.Dense(cfg.hidden_dim)(obs)
        h = nn.relu(h)  # [n_agents, hidden_dim]

        # Communication rounds
        for _ in range(cfg.comm_rounds):
            # For each agent, average all OTHER agents' hidden states
            sum_all = h.sum(axis=0, keepdims=True)  # [1, hidden]
            avg_others = (sum_all - h) / max(n_agents - 1, 1)  # [n_agents, hidden]
            combined = jnp.concatenate([h, avg_others], axis=-1)
            h = nn.Dense(cfg.hidden_dim)(combined)
            h = nn.relu(h)

        # Policy head
        x = nn.Dense(cfg.hidden_dim)(h)
        x = nn.relu(x)
        logits = nn.Dense(cfg.n_actions)(x)
        return logits


class CommNetCritic(nn.Module):
    config: CommNetConfig

    @nn.compact
    def __call__(self, state):
        cfg = self.config
        x = nn.Dense(cfg.critic_hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(cfg.critic_hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class CommNetImpl:
    def __init__(self, config: CommNetConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.obs_dim = config.obs_dim
        self.state_dim = config.obs_dim * config.n_agents

        self.agent_net = CommNetAgent(config)
        self.critic_net = CommNetCritic(config)

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
        return jnp.zeros((n_envs, self.n_agents, 1))

    def step(self, agent_params, obs, carry, prev_actions, rng):
        n_envs = obs.shape[0]

        def per_env(env_idx, env_obs):
            rng_env = jax.random.fold_in(rng, env_idx)
            logits = self.agent_net.apply(agent_params, env_obs)
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
        def forward_one(env_obs):
            return self.agent_net.apply(params, env_obs)

        all_logits = jax.vmap(forward_one)(batch.obs)

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
