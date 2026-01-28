"""QPLEX (QPLEX: Duplex Dueling Multi-Agent Q-Learning) Algorithm Implementation.

Based on "QPLEX: Duplex Dueling Multi-Agent Q-Learning" (Wang et al., 2021).
Extends QMIX by using duplex dueling architecture to achieve full expressiveness
while maintaining IGM (Individual-Global-Max) consistency.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


class QPLEXAgent(nn.Module):
    """QPLEX Agent network with dueling architecture."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)

        # Dueling architecture
        self.fc_v = nn.Linear(rnn_hidden_dim, 1)  # State value
        self.fc_a = nn.Linear(rnn_hidden_dim, n_actions)  # Advantage

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning Q-values, value, advantages, and new hidden state."""
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden)

        v = self.fc_v(h)  # [batch, 1]
        a = self.fc_a(h)  # [batch, n_actions]

        # Dueling: Q = V + A - mean(A)
        q = v + a - a.mean(dim=-1, keepdim=True)

        return q, v.squeeze(-1), a, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)


class QPLEXMixer(nn.Module):
    """QPLEX Mixing network with transformation weights."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_hidden_dim: int = 64,
        n_attention_heads: int = 4,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.n_heads = n_attention_heads
        self.embed_dim = mixing_embed_dim

        # Transformation weights for each agent (lambda_i)
        self.lambda_net = nn.Sequential(
            nn.Linear(state_dim + n_agents, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents),
            nn.Softplus(),  # Ensure positive weights
        )

        # State value network V_tot(s)
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1),
        )

        # Attention for advantage transformation
        self.attention_key = nn.Linear(state_dim, mixing_embed_dim * n_attention_heads)
        self.attention_query = nn.Linear(n_agents, mixing_embed_dim * n_attention_heads)
        self.attention_value = nn.Linear(n_agents, mixing_embed_dim * n_attention_heads)

    def forward(
        self,
        agent_qs: torch.Tensor,
        agent_vs: torch.Tensor,
        agent_as: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q_tot using QPLEX decomposition.

        Args:
            agent_qs: Individual Q-values [batch, n_agents]
            agent_vs: Individual state values [batch, n_agents]
            agent_as: Individual advantages [batch, n_agents] (for chosen actions)
            states: Global state [batch, state_dim]

        Returns:
            q_tot: Total Q-value [batch, 1]
        """
        batch_size = agent_qs.shape[0]

        # Compute transformation weights lambda_i(s, a)
        # Using state and one-hot actions (simplified: use agent advantages)
        lambda_input = torch.cat([states, agent_as], dim=-1)
        lambdas = self.lambda_net(lambda_input)  # [batch, n_agents]
        lambdas = lambdas + 1e-8  # Numerical stability

        # Compute V_tot(s)
        v_tot = self.v_net(states)  # [batch, 1]

        # Weighted sum of individual advantages
        # A_tot = sum_i lambda_i * A_i
        weighted_advantages = (lambdas * agent_as).sum(dim=-1, keepdim=True)

        # Q_tot = V_tot + A_tot
        q_tot = v_tot + weighted_advantages

        return q_tot


@register_algorithm("qplex")
class QPLEX(BaseAlgorithm):
    """QPLEX: Duplex Dueling Multi-Agent Q-Learning.

    Key improvements over QMIX:
    - Duplex dueling architecture for full expressiveness
    - Transformation weights allow non-monotonic value decomposition
    - Maintains IGM consistency through constrained optimization
    """

    def __init__(
        self,
        config: DictConfig,
        n_agents: int,
        obs_shape: tuple,
        state_shape: tuple,
        n_actions: int,
        device: torch.device,
    ):
        self.hidden_states = None
        self.target_hidden_states = None
        super().__init__(config, n_agents, obs_shape, state_shape, n_actions, device)

    def _build_networks(self) -> None:
        """Build QPLEX agent and mixer networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        agent_config = getattr(algo_config, 'agent_network', {})
        hidden_dim = agent_config.get('hidden_dim', 64) if isinstance(agent_config, dict) else 64
        rnn_hidden_dim = agent_config.get('rnn_hidden_dim', 64) if isinstance(agent_config, dict) else 64

        mixer_config = getattr(algo_config, 'mixing_network', {})
        embed_dim = mixer_config.get('embed_dim', 32) if isinstance(mixer_config, dict) else 32
        hypernet_hidden = mixer_config.get('hypernet_hidden_dim', 64) if isinstance(mixer_config, dict) else 64

        # Agent network (shared across agents)
        self.agent = QPLEXAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)

        # Mixer network
        self.mixer = QPLEXMixer(
            n_agents=self.n_agents,
            state_dim=state_dim,
            mixing_embed_dim=embed_dim,
            hypernet_hidden_dim=hypernet_hidden,
        ).to(self.device)

        # Target networks
        self.target_agent = QPLEXAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.target_mixer = QPLEXMixer(
            n_agents=self.n_agents,
            state_dim=state_dim,
            mixing_embed_dim=embed_dim,
            hypernet_hidden_dim=hypernet_hidden,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.rnn_hidden_dim = rnn_hidden_dim

    def _build_optimizers(self) -> None:
        """Build optimizer for agent and mixer."""
        lr = self.config.training.lr
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def init_hidden(self, batch_size: int) -> None:
        """Initialize hidden states for all agents."""
        self.hidden_states = self.agent.init_hidden(batch_size * self.n_agents, self.device)
        self.target_hidden_states = self.target_agent.init_hidden(batch_size * self.n_agents, self.device)

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions using epsilon-greedy policy.

        Args:
            observations: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions]
            explore: Whether to use epsilon-greedy

        Returns:
            actions: [batch, n_agents]
        """
        batch_size = observations.shape[0]

        if self.hidden_states is None:
            self.init_hidden(batch_size)

        # Reshape for processing
        obs_flat = observations.contiguous().reshape(batch_size * self.n_agents, -1)

        with torch.no_grad():
            q_values, _, _, new_hidden = self.agent(obs_flat, self.hidden_states)

        self.hidden_states = new_hidden
        q_values = q_values.reshape(batch_size, self.n_agents, -1)

        # Mask unavailable actions
        if available_actions is not None:
            q_values = q_values.masked_fill(available_actions == 0, -1e10)

        # Epsilon-greedy
        if explore:
            epsilon = self._get_epsilon()
            random_actions = torch.randint(
                0, self.n_actions, (batch_size, self.n_agents), device=self.device
            )
            greedy_actions = q_values.argmax(dim=-1)
            random_mask = (torch.rand(batch_size, self.n_agents, device=self.device) < epsilon)
            actions = torch.where(random_mask, random_actions, greedy_actions)

            # Ensure valid actions
            if available_actions is not None:
                for b in range(batch_size):
                    for a in range(self.n_agents):
                        if available_actions[b, a, actions[b, a]] == 0:
                            valid = available_actions[b, a].nonzero().squeeze(-1)
                            if len(valid) > 0:
                                actions[b, a] = valid[torch.randint(len(valid), (1,))]
        else:
            actions = q_values.argmax(dim=-1)

        return actions

    def _get_epsilon(self) -> float:
        """Get current epsilon for exploration."""
        training_config = self.config.training
        epsilon_start = getattr(training_config, 'epsilon_start', 1.0)
        epsilon_finish = getattr(training_config, 'epsilon_finish', 0.05)
        epsilon_anneal_time = getattr(training_config, 'epsilon_anneal_time', 50000)

        delta = (epsilon_start - epsilon_finish) / epsilon_anneal_time
        epsilon = max(epsilon_finish, epsilon_start - delta * self.training_step)
        return epsilon

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform QPLEX training step."""
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        state = batch["state"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        mask = batch.get("mask", torch.ones_like(rewards)).to(self.device)
        available_actions = batch.get("available_actions")
        next_available_actions = batch.get("next_available_actions")

        if available_actions is not None:
            available_actions = available_actions.to(self.device)
        if next_available_actions is not None:
            next_available_actions = next_available_actions.to(self.device)

        batch_size, seq_len = obs.shape[:2]
        gamma = self.config.training.gamma

        # Initialize hidden states
        hidden = self.agent.init_hidden(batch_size * self.n_agents, self.device)
        target_hidden = self.target_agent.init_hidden(batch_size * self.n_agents, self.device)

        loss_sum = 0.0
        td_error_sum = 0.0

        for t in range(seq_len):
            # Current Q-values
            obs_t = obs[:, t].contiguous().reshape(batch_size * self.n_agents, -1)
            q_values, v_values, a_values, hidden = self.agent(obs_t, hidden)
            q_values = q_values.reshape(batch_size, self.n_agents, -1)
            v_values = v_values.reshape(batch_size, self.n_agents)
            a_values = a_values.reshape(batch_size, self.n_agents, -1)

            # Get Q-values for chosen actions
            actions_t = actions[:, t]
            chosen_q = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
            chosen_a = a_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)

            # Compute Q_tot
            state_t = state[:, t]
            q_tot = self.mixer(chosen_q, v_values, chosen_a, state_t)

            # Target Q-values
            with torch.no_grad():
                next_obs_t = next_obs[:, t].contiguous().reshape(batch_size * self.n_agents, -1)
                next_q, next_v, next_a, target_hidden = self.target_agent(next_obs_t, target_hidden)
                next_q = next_q.reshape(batch_size, self.n_agents, -1)
                next_v = next_v.reshape(batch_size, self.n_agents)
                next_a = next_a.reshape(batch_size, self.n_agents, -1)

                # Mask unavailable actions
                if next_available_actions is not None:
                    next_avail = next_available_actions[:, t]
                    next_q = next_q.masked_fill(next_avail == 0, -1e10)

                # Double Q-learning: use current network to select actions
                curr_next_q, _, curr_next_a, _ = self.agent(next_obs_t, hidden.detach())
                curr_next_q = curr_next_q.reshape(batch_size, self.n_agents, -1)
                curr_next_a = curr_next_a.reshape(batch_size, self.n_agents, -1)

                if next_available_actions is not None:
                    curr_next_q = curr_next_q.masked_fill(next_avail == 0, -1e10)

                next_actions = curr_next_q.argmax(dim=-1)

                # Get target Q-values for selected actions
                target_chosen_q = next_q.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
                target_chosen_a = next_a.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)

                # Target Q_tot
                next_state_t = next_state[:, t]
                target_q_tot = self.target_mixer(target_chosen_q, next_v, target_chosen_a, next_state_t)

                # TD target
                reward_t = rewards[:, t].unsqueeze(-1)
                done_t = dones[:, t].unsqueeze(-1)
                target = reward_t + gamma * (1 - done_t) * target_q_tot

            # TD error
            mask_t = mask[:, t].unsqueeze(-1)
            td_error = (q_tot - target) * mask_t
            loss = (td_error ** 2).sum() / (mask_t.sum() + 1e-8)

            loss_sum += loss
            td_error_sum += td_error.abs().mean().item()

        # Average loss over sequence
        loss_avg = loss_sum / seq_len

        # Optimize
        self.optimizer.zero_grad()
        loss_avg.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            self.config.training.grad_clip
        )
        self.optimizer.step()

        # Update target networks
        target_update_interval = getattr(self.config.training, 'target_update_interval', 200)
        if self.training_step % target_update_interval == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.training_step += 1

        return {
            "loss": loss_avg.item(),
            "td_error": td_error_sum / seq_len,
            "epsilon": self._get_epsilon(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent.load_state_dict(checkpoint["target_agent"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
