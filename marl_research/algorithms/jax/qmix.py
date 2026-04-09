"""QMIX (Monotonic Value Function Factorisation) in JAX/Flax.

End-to-end JAX implementation for vectorized training.
Matches the PyTorch QMIX architecture:
  - RNNAgent: Dense(obs -> hidden) -> GRUCell -> Dense -> Q-values
  - QMixer: Hypernetwork-based monotonic mixing with abs() weights
  - Target networks with periodic hard updates
  - Epsilon-greedy exploration with linear annealing
"""

from typing import NamedTuple, Tuple, Dict, Any
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import chex


class QMIXConfig(NamedTuple):
    """QMIX hyperparameters."""
    # Architecture
    hidden_dim: int = 64
    rnn_hidden_dim: int = 64
    embed_dim: int = 32
    hypernet_hidden_dim: int = 64

    # Training
    lr: float = 5e-4
    gamma: float = 0.99
    grad_clip: float = 10.0
    target_update_interval: int = 200

    # Exploration
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50000

    # Environment
    n_agents: int = 2
    n_actions: int = 6
    obs_dim: int = 30
    state_dim: int = 60


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class RNNAgent(nn.Module):
    """RNN-based agent network: obs -> Dense -> GRU -> Q-values.

    Forward:
        x = relu(Dense(obs))
        h' = GRUCell(x, h)
        q = Dense(h')
    """
    hidden_dim: int
    rnn_hidden_dim: int
    n_actions: int

    @nn.compact
    def __call__(self, obs, hidden_state):
        """
        Args:
            obs: [..., obs_dim]
            hidden_state: [..., rnn_hidden_dim]
        Returns:
            q_values: [..., n_actions]
            new_hidden: [..., rnn_hidden_dim]
        """
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)

        gru = nn.GRUCell(features=self.rnn_hidden_dim)
        new_hidden = gru(hidden_state, x)  # Flax GRU: (carry, input) -> new_carry

        q_values = nn.Dense(self.n_actions)(new_hidden)
        return q_values, new_hidden


class HyperNetwork(nn.Module):
    """Two-layer hypernetwork: state -> weight/bias vector."""
    input_dim: int
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim)(state)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class QMixer(nn.Module):
    """QMIX mixing network with hypernetworks.

    Produces Q_total from per-agent Q-values and global state:
        w1 = |hyper_w1(state)|              [n_agents, embed_dim]
        b1 = hyper_b1(state)                [1, embed_dim]
        w2 = |hyper_w2(state)|              [embed_dim, 1]
        b2 = hyper_b2(state)                [1, 1]
        Q_total = w2^T ELU(w1^T q + b1) + b2
    """
    n_agents: int
    embed_dim: int
    hypernet_hidden_dim: int

    @nn.compact
    def __call__(self, agent_qs, state):
        """
        Args:
            agent_qs: [batch, n_agents]  per-agent chosen Q-values
            state: [batch, state_dim]    global state
        Returns:
            q_total: [batch]
        """
        batch_size = agent_qs.shape[0]

        # --- First layer hypernetworks ---
        # w1: state -> [n_agents * embed_dim], then abs() for monotonicity
        w1 = HyperNetwork(
            input_dim=state.shape[-1],
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.n_agents * self.embed_dim,
        )(state)
        w1 = jnp.abs(w1)
        w1 = w1.reshape(batch_size, self.n_agents, self.embed_dim)

        # b1: state -> [embed_dim] (linear)
        b1 = nn.Dense(self.embed_dim)(state)  # [batch, embed_dim]
        b1 = b1.reshape(batch_size, 1, self.embed_dim)

        # --- Forward through first layer ---
        agent_qs_2d = agent_qs.reshape(batch_size, 1, self.n_agents)  # [batch, 1, n_agents]
        hidden = jnp.matmul(agent_qs_2d, w1)  # [batch, 1, embed_dim]
        hidden = hidden + b1
        hidden = nn.elu(hidden)

        # --- Second layer hypernetworks ---
        # w2: state -> [embed_dim], then abs()
        w2 = HyperNetwork(
            input_dim=state.shape[-1],
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embed_dim,
        )(state)
        w2 = jnp.abs(w2)
        w2 = w2.reshape(batch_size, self.embed_dim, 1)

        # b2: state -> [1] (2-layer MLP)
        b2_h = nn.Dense(self.embed_dim)(state)
        b2_h = nn.relu(b2_h)
        b2 = nn.Dense(1)(b2_h)  # [batch, 1]
        b2 = b2.reshape(batch_size, 1, 1)

        # --- Forward through second layer ---
        q_total = jnp.matmul(hidden, w2) + b2  # [batch, 1, 1]
        q_total = q_total.reshape(batch_size)

        return q_total


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class QMIX:
    """QMIX algorithm with JAX training.

    Provides:
        - init(): create agent, mixer, and target params/states
        - get_action(): epsilon-greedy action selection (jit-compiled)
        - train_step(): one gradient update on a batch (jit-compiled)
        - update_targets(): hard copy online -> target params
    """

    def __init__(self, config: QMIXConfig):
        self.config = config
        self.agent = RNNAgent(
            hidden_dim=config.hidden_dim,
            rnn_hidden_dim=config.rnn_hidden_dim,
            n_actions=config.n_actions,
        )
        self.mixer = QMixer(
            n_agents=config.n_agents,
            embed_dim=config.embed_dim,
            hypernet_hidden_dim=config.hypernet_hidden_dim,
        )

    def init(self, rng):
        """Initialize all parameters and optimizer states.

        Returns:
            agent_state: TrainState for the RNN agent
            mixer_state: TrainState for the mixing network
            target_agent_params: parameter tree (copy of agent)
            target_mixer_params: parameter tree (copy of mixer)
        """
        cfg = self.config
        rng_agent, rng_mixer = jax.random.split(rng)

        # Dummy inputs
        dummy_obs = jnp.zeros(cfg.obs_dim)
        dummy_hidden = jnp.zeros(cfg.rnn_hidden_dim)
        dummy_state = jnp.zeros(cfg.state_dim)
        dummy_agent_qs = jnp.zeros((1, cfg.n_agents))
        dummy_state_batch = jnp.zeros((1, cfg.state_dim))

        # Initialize parameters
        agent_params = self.agent.init(rng_agent, dummy_obs, dummy_hidden)
        mixer_params = self.mixer.init(rng_mixer, dummy_agent_qs, dummy_state_batch)

        # Optimizer (shared learning rate, RMSProp-like via optax)
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.lr, eps=1e-5),
        )

        agent_state = TrainState.create(
            apply_fn=self.agent.apply,
            params=agent_params,
            tx=tx,
        )

        # Separate optimizer for mixer (same config)
        mixer_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.lr, eps=1e-5),
        )
        mixer_state = TrainState.create(
            apply_fn=self.mixer.apply,
            params=mixer_params,
            tx=mixer_tx,
        )

        # Target network params (deep copy)
        target_agent_params = jax.tree.map(lambda x: x.copy(), agent_params)
        target_mixer_params = jax.tree.map(lambda x: x.copy(), mixer_params)

        return agent_state, mixer_state, target_agent_params, target_mixer_params

    def init_hidden(self, batch_size: int = 1):
        """Return zero initial hidden states.

        Returns:
            hidden: [batch_size, n_agents, rnn_hidden_dim]
        """
        return jnp.zeros((batch_size, self.config.n_agents, self.config.rnn_hidden_dim))

    @staticmethod
    def get_epsilon(training_step: int, cfg: QMIXConfig) -> float:
        """Compute epsilon for exploration schedule."""
        frac = jnp.clip(training_step / cfg.epsilon_anneal_time, 0.0, 1.0)
        epsilon = cfg.epsilon_finish + (cfg.epsilon_start - cfg.epsilon_finish) * (1.0 - frac)
        return epsilon

    @partial(jax.jit, static_argnums=(0,))
    def get_action(
        self,
        agent_state: TrainState,
        obs: jnp.ndarray,
        hidden_states: jnp.ndarray,
        rng: chex.PRNGKey,
        training_step: int,
        available_actions: jnp.ndarray = None,
    ):
        """Epsilon-greedy action selection for all agents.

        Args:
            agent_state: TrainState with agent params
            obs: [n_agents, obs_dim]
            hidden_states: [n_agents, rnn_hidden_dim]
            rng: PRNGKey
            training_step: current step (for epsilon schedule)
            available_actions: [n_agents, n_actions] binary mask or None

        Returns:
            actions: [n_agents] int32
            new_hidden: [n_agents, rnn_hidden_dim]
            q_values: [n_agents, n_actions]
        """
        cfg = self.config

        # Compute Q-values for all agents (vmapped)
        def agent_forward(obs_i, h_i):
            q_vals, new_h = agent_state.apply_fn(agent_state.params, obs_i, h_i)
            return q_vals, new_h

        q_values, new_hidden = jax.vmap(agent_forward)(obs, hidden_states)
        # q_values: [n_agents, n_actions], new_hidden: [n_agents, rnn_hidden_dim]

        # Mask unavailable actions for greedy selection
        masked_q = q_values
        if available_actions is not None:
            masked_q = jnp.where(available_actions > 0, q_values, -1e10)

        greedy_actions = jnp.argmax(masked_q, axis=-1)  # [n_agents]

        # Epsilon-greedy
        epsilon = self.get_epsilon(training_step, cfg)

        rng_explore, rng_random = jax.random.split(rng)
        explore_mask = jax.random.uniform(rng_explore, (cfg.n_agents,)) < epsilon

        # Random actions (respecting available_actions)
        if available_actions is not None:
            # Sample proportionally from available actions
            random_logits = jnp.where(available_actions > 0, 0.0, -1e10)
            random_actions = jax.random.categorical(
                rng_random, random_logits, axis=-1
            )  # [n_agents]
        else:
            random_actions = jax.random.randint(
                rng_random, (cfg.n_agents,), 0, cfg.n_actions
            )

        actions = jnp.where(explore_mask, random_actions, greedy_actions)

        return actions, new_hidden, q_values

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        agent_state: TrainState,
        mixer_state: TrainState,
        target_agent_params: Any,
        target_mixer_params: Any,
        batch: Dict[str, jnp.ndarray],
    ):
        """Perform one QMIX training step.

        Args:
            agent_state: TrainState for agent
            mixer_state: TrainState for mixer
            target_agent_params: frozen target agent params
            target_mixer_params: frozen target mixer params
            batch: dict with keys:
                obs: [batch, seq_len, n_agents, obs_dim]
                next_obs: [batch, seq_len, n_agents, obs_dim]
                state: [batch, seq_len, state_dim]
                next_state: [batch, seq_len, state_dim]
                actions: [batch, seq_len, n_agents] int32
                rewards: [batch, seq_len]
                dones: [batch, seq_len]
                mask: [batch, seq_len] (1 = valid, 0 = padding)
                available_actions: [batch, seq_len, n_agents, n_actions] (optional)
                next_available_actions: [batch, seq_len, n_agents, n_actions] (optional)

        Returns:
            agent_state: updated TrainState
            mixer_state: updated TrainState
            metrics: dict of scalar metrics
        """
        cfg = self.config

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        state = batch["state"]
        next_state = batch["next_state"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        mask = batch.get("mask", jnp.ones_like(rewards))
        available_actions = batch.get("available_actions", None)
        next_available_actions = batch.get("next_available_actions", None)

        batch_size, seq_len = obs.shape[0], obs.shape[1]

        def _unroll_agent(params, obs_seq):
            """Unroll agent RNN over a sequence for all agents.

            Args:
                params: agent network params
                obs_seq: [batch, seq_len, n_agents, obs_dim]

            Returns:
                q_all: [batch, seq_len, n_agents, n_actions]
            """
            def _step_all_agents(carry, obs_t):
                """Process one timestep for all agents.

                Args:
                    carry: hidden [batch, n_agents, rnn_hidden_dim]
                    obs_t: [batch, n_agents, obs_dim]
                Returns:
                    new_carry, q_t: [batch, n_agents, n_actions]
                """
                hidden = carry

                def _single_agent_step(obs_i, h_i):
                    # obs_i: [batch, obs_dim], h_i: [batch, rnn_hidden_dim]
                    q, new_h = jax.vmap(
                        lambda o, h: self.agent.apply(params, o, h)
                    )(obs_i, h_i)
                    return q, new_h

                # Split across agents, process each, recombine
                # obs_t: [batch, n_agents, obs_dim]
                # hidden: [batch, n_agents, rnn_hidden_dim]
                # Reshape to [batch * n_agents, ...] for efficient batched forward
                obs_flat = obs_t.reshape(batch_size * cfg.n_agents, -1)
                h_flat = hidden.reshape(batch_size * cfg.n_agents, -1)

                q_flat, new_h_flat = jax.vmap(
                    lambda o, h: self.agent.apply(params, o, h)
                )(obs_flat, h_flat)

                q_t = q_flat.reshape(batch_size, cfg.n_agents, cfg.n_actions)
                new_hidden = new_h_flat.reshape(batch_size, cfg.n_agents, cfg.rnn_hidden_dim)

                return new_hidden, q_t

            init_hidden = jnp.zeros((batch_size, cfg.n_agents, cfg.rnn_hidden_dim))
            # obs_seq: [batch, seq_len, n_agents, obs_dim] -> scan over seq_len
            obs_transposed = jnp.transpose(obs_seq, (1, 0, 2, 3))  # [seq_len, batch, n_agents, obs_dim]

            _, q_all = jax.lax.scan(_step_all_agents, init_hidden, obs_transposed)
            # q_all: [seq_len, batch, n_agents, n_actions]
            q_all = jnp.transpose(q_all, (1, 0, 2, 3))  # [batch, seq_len, n_agents, n_actions]
            return q_all

        def _loss_fn(agent_params, mixer_params):
            # Online Q-values
            q_values = _unroll_agent(agent_params, obs)
            # [batch, seq_len, n_agents, n_actions]

            # Gather chosen Q-values
            actions_oh = jax.nn.one_hot(actions, cfg.n_actions)  # [batch, seq_len, n_agents, n_actions]
            chosen_q = jnp.sum(q_values * actions_oh, axis=-1)  # [batch, seq_len, n_agents]

            # Mix to get Q_total
            # Reshape for mixer: flatten batch*seq_len
            chosen_q_flat = chosen_q.reshape(batch_size * seq_len, cfg.n_agents)
            state_flat = state.reshape(batch_size * seq_len, -1)
            q_total = self.mixer.apply(mixer_params, chosen_q_flat, state_flat)
            q_total = q_total.reshape(batch_size, seq_len)

            # Target Q-values (no grad)
            target_q_values = jax.lax.stop_gradient(
                _unroll_agent(target_agent_params, next_obs)
            )

            # Mask unavailable next actions
            if next_available_actions is not None:
                target_q_values = jnp.where(
                    next_available_actions > 0, target_q_values, -1e10
                )

            target_max_q = jnp.max(target_q_values, axis=-1)  # [batch, seq_len, n_agents]

            # Target Q_total
            target_max_q_flat = target_max_q.reshape(batch_size * seq_len, cfg.n_agents)
            next_state_flat = next_state.reshape(batch_size * seq_len, -1)
            target_q_total = jax.lax.stop_gradient(
                self.mixer.apply(target_mixer_params, target_max_q_flat, next_state_flat)
            )
            target_q_total = target_q_total.reshape(batch_size, seq_len)

            # TD target
            targets = rewards + cfg.gamma * (1.0 - dones) * target_q_total

            # Masked MSE loss
            td_error = (q_total - jax.lax.stop_gradient(targets)) ** 2
            loss = (td_error * mask).sum() / (mask.sum() + 1e-8)

            return loss, {
                "loss": loss,
                "q_total_mean": (q_total * mask).sum() / (mask.sum() + 1e-8),
                "td_error_mean": (jnp.abs(q_total - targets) * mask).sum() / (mask.sum() + 1e-8),
            }

        # Compute gradients w.r.t. both agent and mixer params
        (loss, metrics), (agent_grads, mixer_grads) = jax.value_and_grad(
            _loss_fn, argnums=(0, 1), has_aux=True
        )(agent_state.params, mixer_state.params)

        # Apply updates
        agent_state = agent_state.apply_gradients(grads=agent_grads)
        mixer_state = mixer_state.apply_gradients(grads=mixer_grads)

        return agent_state, mixer_state, metrics

    @staticmethod
    @jax.jit
    def update_targets(
        agent_state: TrainState,
        mixer_state: TrainState,
        target_agent_params: Any,
        target_mixer_params: Any,
    ):
        """Hard update: copy online params to target params.

        Returns:
            target_agent_params: updated target agent params
            target_mixer_params: updated target mixer params
        """
        target_agent_params = jax.tree.map(lambda x: x.copy(), agent_state.params)
        target_mixer_params = jax.tree.map(lambda x: x.copy(), mixer_state.params)
        return target_agent_params, target_mixer_params
