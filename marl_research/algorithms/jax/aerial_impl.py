"""AERIAL implementing the JaxMARLAlgo interface.

Attention-based recurrence for multi-agent RL. Each agent has a recurrent
belief state. Beliefs are updated via attention over OTHER agents' belief
states (requires implicit communication of beliefs).

Key difference from VABL: attends over teammate hidden states, not actions.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

from marl_research.algorithms.jax.algo_interface import RolloutBatch


class AERIALConfig(NamedTuple):
    embed_dim: int = 64
    hidden_dim: int = 128
    attention_heads: int = 4
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


class AERIALAgent(nn.Module):
    """Joint forward for all agents with attention over teammate beliefs.

    Input: obs [n_agents, obs_dim], beliefs [n_agents, hidden_dim]
    Output: logits [n_agents, n_actions], new_beliefs [n_agents, hidden_dim]
    """
    config: AERIALConfig

    @nn.compact
    def __call__(self, obs, beliefs):
        cfg = self.config
        n_agents = cfg.n_agents
        n_teammates = n_agents - 1

        # Encode obs
        h_obs = nn.Dense(cfg.embed_dim)(obs)
        h_obs = nn.relu(h_obs)
        h_obs = nn.Dense(cfg.embed_dim)(h_obs)
        h_obs = nn.relu(h_obs)  # [n_agents, embed_dim]

        # Project beliefs for attention
        belief_proj = nn.Dense(cfg.embed_dim)(beliefs)  # [n_agents, embed_dim]

        # For each agent, attend over OTHER agents' belief projections
        # Use full attention matrix then mask out self
        attn_logits = (belief_proj @ belief_proj.T) / jnp.sqrt(cfg.embed_dim)  # [n_agents, n_agents]
        eye_mask = jnp.eye(n_agents) * -1e8  # mask self-attention
        attn_logits = attn_logits + eye_mask
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # [n_agents, n_agents]
        contexts = attn_weights @ belief_proj  # [n_agents, embed_dim]

        # GRU update: input = [h_obs || context], hidden = beliefs
        gru_input = jnp.concatenate([h_obs, contexts], axis=-1)  # [n_agents, 2*embed_dim]
        gru_input = nn.Dense(cfg.hidden_dim)(gru_input)

        gru = nn.GRUCell(features=cfg.hidden_dim)
        def per_agent_gru(belief, inp):
            new_b, _ = gru(belief, inp)
            return new_b
        new_beliefs = jax.vmap(per_agent_gru)(beliefs, gru_input)  # [n_agents, hidden_dim]

        # Policy head
        logits = nn.Dense(cfg.n_actions)(new_beliefs)
        return logits, new_beliefs


class AERIALCritic(nn.Module):
    config: AERIALConfig

    @nn.compact
    def __call__(self, state):
        cfg = self.config
        x = nn.Dense(cfg.critic_hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(cfg.critic_hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class AERIALImpl:
    def __init__(self, config: AERIALConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_actions = config.n_actions
        self.obs_dim = config.obs_dim
        self.state_dim = config.obs_dim * config.n_agents

        self.agent_net = AERIALAgent(config)
        self.critic_net = AERIALCritic(config)

    def init(self, rng):
        cfg = self.config
        rng_a, rng_c = jax.random.split(rng)
        agent_params = self.agent_net.init(
            rng_a,
            jnp.zeros((self.n_agents, self.obs_dim)),
            jnp.zeros((self.n_agents, cfg.hidden_dim)),
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

        def per_env(env_idx, env_obs, env_beliefs):
            rng_env = jax.random.fold_in(rng, env_idx)
            logits, new_beliefs = self.agent_net.apply(agent_params, env_obs, env_beliefs)
            def per_agent(i):
                rng_i = jax.random.fold_in(rng_env, i)
                action = jax.random.categorical(rng_i, logits[i])
                lp = jax.nn.log_softmax(logits[i])[action]
                return action, lp
            actions, lps = jax.vmap(per_agent)(jnp.arange(self.n_agents))
            return actions, new_beliefs, lps

        return jax.vmap(per_env)(jnp.arange(n_envs), obs, carry)

    def get_value(self, critic_params, state):
        return jax.vmap(lambda s: self.critic_net.apply(critic_params, s))(state)

    def actor_loss(self, params, batch, clip_param, entropy_coef):
        B = batch.obs.shape[0]

        def forward_one(env_obs, env_carry):
            logits, _ = self.agent_net.apply(params, env_obs, env_carry)
            return logits

        all_logits = jax.vmap(forward_one)(batch.obs, batch.carry)

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
