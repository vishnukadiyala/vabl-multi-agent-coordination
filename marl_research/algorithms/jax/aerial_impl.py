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
    # Auxiliary-loss knobs for reviewer-requested AERIAL fix-path experiment
    # (2026-04-20). Default aux_lambda=0.0 preserves the no-aux behavior used
    # for the Section 6.1 AERIAL observation runs; aux_lambda>0 with
    # use_aux_loss=True adds a VABL-style teammate-next-action prediction
    # head so the "stop-gradient on belief into aux head" intervention can
    # be tested on AERIAL.
    aux_hidden_dim: int = 64
    aux_lambda: float = 0.0
    use_aux_loss: bool = False
    stop_gradient_belief_to_aux: bool = False


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

        # Aux head: predict each teammate's next action from the agent's
        # own belief. Only instantiated when use_aux_loss=True so the
        # default baseline's random-init trajectory is identical to the
        # pre-aux-patch implementation.
        n_teammates = n_agents - 1
        if cfg.use_aux_loss:
            if cfg.stop_gradient_belief_to_aux:
                belief_for_aux = jax.lax.stop_gradient(new_beliefs)
            else:
                belief_for_aux = new_beliefs
            aux_h = nn.Dense(cfg.aux_hidden_dim, name="aux_h1")(belief_for_aux)
            aux_h = nn.relu(aux_h)
            aux_logits_flat = nn.Dense(n_teammates * cfg.n_actions, name="aux_h2")(aux_h)
            aux_logits = aux_logits_flat.reshape(n_agents, n_teammates, cfg.n_actions)
        else:
            aux_logits = jnp.zeros((n_agents, n_teammates, cfg.n_actions))

        return logits, new_beliefs, aux_logits


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
            logits, new_beliefs, _ = self.agent_net.apply(agent_params, env_obs, env_beliefs)
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
        cfg = self.config

        def forward_one(env_obs, env_carry):
            logits, _, aux_logits = self.agent_net.apply(params, env_obs, env_carry)
            return logits, aux_logits

        all_logits, all_aux = jax.vmap(forward_one)(batch.obs, batch.carry)
        # all_logits: [B, n_agents, n_actions]
        # all_aux:    [B, n_agents, n_teammates, n_actions]

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

        # Aux loss: cross-entropy predicting each teammate's next action
        # from the agent's own belief. Built with integer indexing (boolean
        # masking is not jit-friendly).
        n_agents = cfg.n_agents
        if cfg.use_aux_loss:
            aux_lp = jax.nn.log_softmax(all_aux, axis=-1)  # [B, n_agents, n_teammates, n_actions]
            teammate_idx_per_agent = jnp.array(
                [[j for j in range(n_agents) if j != i] for i in range(n_agents)],
                dtype=jnp.int32,
            )  # [n_agents, n_teammates]
            # batch.next_actions: [B, n_agents]. Gather each agent's teammate actions.
            teammate_targets = batch.next_actions[:, teammate_idx_per_agent]  # [B, n_agents, n_teammates]
            aux_taken = jnp.take_along_axis(
                aux_lp, teammate_targets[..., None], axis=-1
            ).squeeze(-1)  # [B, n_agents, n_teammates]
            aux_loss = -aux_taken.mean()
            aux_term = cfg.aux_lambda * aux_loss
        else:
            aux_term = 0.0
        return p_loss + entropy_coef * e_loss + aux_term

    def critic_loss(self, params, batch):
        vals = jax.vmap(lambda s: self.critic_net.apply(params, s))(batch.states)
        return ((vals - batch.returns) ** 2).mean()
