"""CommNet (Communication Neural Network) Algorithm Implementation.

Based on "Learning Multiagent Communication with Backpropagation" (Sukhbaatar et al., 2016).
Agents communicate via learned continuous messages averaged across all agents.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


class CommNetModule(nn.Module):
    """CommNet communication module with multiple communication rounds."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 64,
        comm_rounds: int = 2,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.comm_rounds = comm_rounds

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Communication layers (one per round)
        self.comm_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(comm_rounds)
        ])

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(
        self,
        obs: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with communication.

        Args:
            obs: Observations [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions]

        Returns:
            q_values: Q-values [batch, n_agents, n_actions]
        """
        batch_size = obs.shape[0]

        # Encode observations
        h = self.obs_encoder(obs)  # [batch, n_agents, hidden_dim]

        # Communication rounds
        for comm_layer in self.comm_layers:
            # Compute mean message from all other agents
            # For agent i, message = mean of h_j for j != i
            messages = []
            for i in range(self.n_agents):
                # Average over other agents
                other_h = torch.cat([h[:, :i], h[:, i+1:]], dim=1)  # [batch, n_agents-1, hidden]
                if other_h.shape[1] > 0:
                    msg = other_h.mean(dim=1)  # [batch, hidden]
                else:
                    msg = torch.zeros(batch_size, self.hidden_dim, device=h.device)
                messages.append(msg)

            messages = torch.stack(messages, dim=1)  # [batch, n_agents, hidden]

            # Combine own hidden state with received message
            combined = torch.cat([h, messages], dim=-1)  # [batch, n_agents, hidden*2]
            h = F.relu(comm_layer(combined))  # [batch, n_agents, hidden]

        # Output Q-values
        q_values = self.output(h)  # [batch, n_agents, n_actions]

        # Mask unavailable actions
        if available_actions is not None:
            q_values = q_values.masked_fill(available_actions == 0, -1e10)

        return q_values


class CommNetCritic(nn.Module):
    """Centralized critic for CommNet."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


@register_algorithm("commnet")
class CommNet(BaseAlgorithm):
    """CommNet: Learning Multiagent Communication with Backpropagation.

    Agents learn to communicate via continuous messages that are averaged
    and used to update hidden states. Trained with actor-critic (PPO).

    Key features:
    - Explicit learned communication
    - Multiple communication rounds
    - Differentiable message passing
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
        super().__init__(config, n_agents, obs_shape, state_shape, n_actions, device)

    def _build_networks(self) -> None:
        """Build CommNet and critic networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        hidden_dim = getattr(algo_config, 'hidden_dim', 64)
        comm_rounds = getattr(algo_config, 'comm_rounds', 2)

        self.commnet = CommNetModule(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            hidden_dim=hidden_dim,
            comm_rounds=comm_rounds,
        ).to(self.device)

        self.critic = CommNetCritic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def _build_optimizers(self) -> None:
        """Build optimizers for actor and critic."""
        lr = self.config.training.lr
        self.actor_optimizer = torch.optim.Adam(self.commnet.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

    def init_hidden(self, batch_size: int) -> None:
        """CommNet doesn't use persistent hidden states."""
        pass

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions for all agents with communication.

        Args:
            observations: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions]
            explore: Whether to sample or take argmax

        Returns:
            actions: [batch, n_agents]
        """
        with torch.no_grad():
            logits = self.commnet(observations, available_actions)

        if explore:
            probs = F.softmax(logits, dim=-1)
            # Sample actions for each agent
            batch_size = observations.shape[0]
            actions = torch.zeros(batch_size, self.n_agents, dtype=torch.long, device=self.device)
            for i in range(self.n_agents):
                actions[:, i] = torch.multinomial(probs[:, i], num_samples=1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)

        return actions

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages."""
        gamma = self.config.training.gamma
        gae_lambda = getattr(self.config.algorithm, 'gae_lambda', 0.95)

        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        last_gae = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=self.device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
            returns[:, t] = advantages[:, t] + values[:, t]

        return returns * mask, advantages * mask

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform CommNet training step using PPO."""
        algo_config = self.config.algorithm

        obs = batch["obs"].to(self.device)
        state = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        mask = batch.get("mask", torch.ones_like(rewards)).to(self.device)
        available_actions = batch.get("available_actions")
        if available_actions is not None:
            available_actions = available_actions.to(self.device)

        batch_size, seq_len = obs.shape[:2]

        # Compute old values and log probs
        with torch.no_grad():
            state_flat = state.view(batch_size * seq_len, -1)
            old_values = self.critic(state_flat).view(batch_size, seq_len)

            # Get old log probs
            obs_flat = obs.view(batch_size * seq_len, self.n_agents, -1)
            avail_flat = None
            if available_actions is not None:
                avail_flat = available_actions.view(batch_size * seq_len, self.n_agents, -1)

            old_logits = self.commnet(obs_flat, avail_flat)
            old_log_probs_all = F.log_softmax(old_logits, dim=-1)

            actions_flat = actions.view(batch_size * seq_len, self.n_agents)
            old_log_probs = old_log_probs_all.gather(
                2, actions_flat.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1).view(batch_size, seq_len)

            returns, advantages = self._compute_gae(rewards, old_values, dones, mask)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -10, 10)

        # PPO updates
        clip_param = getattr(algo_config, 'clip_param', 0.2)
        n_epochs = getattr(algo_config, 'ppo_epochs', 4)
        value_loss_coef = getattr(algo_config, 'value_loss_coef', 0.5)
        entropy_coef = getattr(algo_config, 'entropy_coef', 0.01)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(n_epochs):
            # Forward pass
            logits = self.commnet(obs_flat, avail_flat)
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)

            log_probs = log_probs_all.gather(
                2, actions_flat.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1).view(batch_size, seq_len)

            entropy = -(probs * log_probs_all).sum(dim=-1).mean(dim=-1).view(batch_size, seq_len)

            # Critic forward
            values = self.critic(state_flat).view(batch_size, seq_len)

            # PPO policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0, 10)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2) * mask
            policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

            # Value loss
            value_loss = (values - returns).pow(2) * mask
            value_loss = value_loss.sum() / (mask.sum() + 1e-8)

            # Entropy loss
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # Actor update
            actor_loss = policy_loss + entropy_coef * entropy_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.commnet.parameters(), self.config.training.grad_clip)
            self.actor_optimizer.step()

            # Critic update
            critic_loss = value_loss_coef * value_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.training.grad_clip)
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        self.training_step += 1

        return {
            "loss": (total_policy_loss + total_value_loss) / n_epochs,
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
        }

    def save(self, path: str) -> None:
        torch.save({
            "commnet": self.commnet.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.commnet.load_state_dict(checkpoint["commnet"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]
