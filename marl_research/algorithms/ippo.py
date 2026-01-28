"""IPPO (Independent PPO) Algorithm Implementation.

Each agent learns independently using PPO, ignoring other agents.
This serves as a simple but often surprisingly strong baseline.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


class IPPOAgent(nn.Module):
    """Independent PPO agent with actor and critic networks."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Critic network (per-agent value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.actor(obs)
        return F.softmax(logits, dim=-1)


@register_algorithm("ippo")
class IPPO(BaseAlgorithm):
    """Independent PPO - each agent learns independently.

    Simple baseline where agents treat other agents as part of the environment.
    Often surprisingly competitive with more sophisticated methods.
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
        """Build independent agent networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        hidden_dim = getattr(algo_config, 'hidden_dim', 64)

        # Shared network architecture across agents (parameter sharing)
        self.agent = IPPOAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def _build_optimizers(self) -> None:
        """Build optimizer."""
        lr = self.config.training.lr
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

    def init_hidden(self, batch_size: int) -> None:
        """IPPO doesn't use hidden states."""
        pass

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions for all agents independently.

        Args:
            observations: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions]
            explore: Whether to sample or take argmax

        Returns:
            actions: [batch, n_agents]
        """
        batch_size = observations.shape[0]
        actions_list = []

        for i in range(self.n_agents):
            obs_i = observations[:, i, :]
            logits, _ = self.agent(obs_i)

            # Mask unavailable actions
            if available_actions is not None:
                avail_i = available_actions[:, i, :]
                logits = logits.masked_fill(avail_i == 0, -1e10)

            if explore:
                probs = F.softmax(logits, dim=-1)
                actions_i = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                actions_i = logits.argmax(dim=-1)

            actions_list.append(actions_i)

        return torch.stack(actions_list, dim=1)

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
        """Perform PPO training step for all agents."""
        algo_config = self.config.algorithm

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        mask = batch.get("mask", torch.ones_like(rewards)).to(self.device)
        available_actions = batch.get("available_actions")
        if available_actions is not None:
            available_actions = available_actions.to(self.device)

        batch_size, seq_len = obs.shape[:2]

        # Compute values for all agents and average reward
        with torch.no_grad():
            all_values = []
            all_old_log_probs = []

            for i in range(self.n_agents):
                obs_i = obs[:, :, i, :].reshape(batch_size * seq_len, -1)
                logits_i, values_i = self.agent(obs_i)

                if available_actions is not None:
                    avail_i = available_actions[:, :, i, :].reshape(batch_size * seq_len, -1)
                    logits_i = logits_i.masked_fill(avail_i == 0, -1e10)

                log_probs_i = F.log_softmax(logits_i, dim=-1)
                actions_i = actions[:, :, i].reshape(batch_size * seq_len)
                old_log_prob_i = log_probs_i.gather(1, actions_i.unsqueeze(1)).squeeze(1)

                all_values.append(values_i.view(batch_size, seq_len))
                all_old_log_probs.append(old_log_prob_i.view(batch_size, seq_len))

            # Average values across agents for shared reward
            values = torch.stack(all_values, dim=-1).mean(dim=-1)
            old_log_probs = torch.stack(all_old_log_probs, dim=-1).sum(dim=-1)

            returns, advantages = self._compute_gae(rewards, values, dones, mask)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -10, 10)

        # PPO update
        clip_param = getattr(algo_config, 'clip_param', 0.2)
        n_epochs = getattr(algo_config, 'ppo_epochs', 4)
        value_loss_coef = getattr(algo_config, 'value_loss_coef', 0.5)
        entropy_coef = getattr(algo_config, 'entropy_coef', 0.01)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(n_epochs):
            all_log_probs = []
            all_values_new = []
            all_entropy = []

            for i in range(self.n_agents):
                obs_i = obs[:, :, i, :].reshape(batch_size * seq_len, -1)
                logits_i, values_i = self.agent(obs_i)

                if available_actions is not None:
                    avail_i = available_actions[:, :, i, :].reshape(batch_size * seq_len, -1)
                    logits_i = logits_i.masked_fill(avail_i == 0, -1e10)

                probs_i = F.softmax(logits_i, dim=-1)
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                actions_i = actions[:, :, i].reshape(batch_size * seq_len)
                log_prob_i = log_probs_i.gather(1, actions_i.unsqueeze(1)).squeeze(1)
                entropy_i = -(probs_i * log_probs_i).sum(dim=-1)

                all_log_probs.append(log_prob_i.view(batch_size, seq_len))
                all_values_new.append(values_i.view(batch_size, seq_len))
                all_entropy.append(entropy_i.view(batch_size, seq_len))

            log_probs = torch.stack(all_log_probs, dim=-1).sum(dim=-1)
            values_new = torch.stack(all_values_new, dim=-1).mean(dim=-1)
            entropy = torch.stack(all_entropy, dim=-1).mean(dim=-1)

            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0, 10)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2) * mask
            policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

            # Value loss
            value_loss = (values_new - returns).pow(2) * mask
            value_loss = value_loss.sum() / (mask.sum() + 1e-8)

            # Entropy loss
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # Total loss
            loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.training.grad_clip)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        self.training_step += 1

        return {
            "loss": total_policy_loss / n_epochs,
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
        }

    def save(self, path: str) -> None:
        torch.save({
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
