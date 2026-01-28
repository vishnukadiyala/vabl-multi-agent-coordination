"""MAPPO (Multi-Agent PPO) Algorithm Implementation.

Based on "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
(Yu et al., 2022). Uses centralized value function with decentralized policies.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


class MAPPOActor(nn.Module):
    """MAPPO Actor network with optional RNN."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        use_rnn: bool = True,
        rnn_hidden_dim: int = 64,
    ):
        super().__init__()
        self.use_rnn = use_rnn
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if use_rnn:
            self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(rnn_hidden_dim, n_actions)
        else:
            self.fc_out = nn.Linear(hidden_dim, n_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        if self.use_rnn:
            if hidden is None:
                hidden = torch.zeros(1, x.shape[0], self.rnn_hidden_dim, device=x.device)
            x, hidden = self.rnn(x.unsqueeze(1), hidden)
            x = x.squeeze(1)

        logits = self.fc_out(x)
        return logits, hidden


class MAPPOCritic(nn.Module):
    """MAPPO Centralized Critic using global state."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


@register_algorithm("mappo")
class MAPPO(BaseAlgorithm):
    """Multi-Agent PPO with centralized training and decentralized execution.

    Key features:
    - Centralized critic using global state
    - Decentralized actors using local observations
    - Optional RNN for handling partial observability
    - Value normalization for training stability
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
        super().__init__(config, n_agents, obs_shape, state_shape, n_actions, device)

        # Value normalization
        self.value_normalizer = ValueNormalizer(device=device)

    def _build_networks(self) -> None:
        """Build actor and critic networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        hidden_dim = getattr(algo_config, 'hidden_dim', 64)
        use_rnn = getattr(algo_config, 'use_rnn', True)
        rnn_hidden_dim = getattr(algo_config, 'rnn_hidden_dim', 64)

        # Shared actor network (parameter sharing across agents)
        self.actor = MAPPOActor(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            hidden_dim=hidden_dim,
            use_rnn=use_rnn,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)

        # Centralized critic
        self.critic = MAPPOCritic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.rnn_hidden_dim = rnn_hidden_dim if use_rnn else None

    def _build_optimizers(self) -> None:
        """Build separate optimizers for actor and critic."""
        lr = self.config.training.lr
        algo_config = self.config.algorithm

        actor_lr = getattr(algo_config, 'actor_lr', lr)
        critic_lr = getattr(algo_config, 'critic_lr', lr)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )

    def init_hidden(self, batch_size: int) -> None:
        """Initialize RNN hidden states."""
        if self.rnn_hidden_dim is not None:
            self.hidden_states = torch.zeros(
                self.n_agents, 1, batch_size, self.rnn_hidden_dim, device=self.device
            )

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions for all agents.

        Args:
            observations: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions]
            explore: Whether to sample or take argmax

        Returns:
            actions: [batch, n_agents]
        """
        batch_size = observations.shape[0]

        if self.hidden_states is None:
            self.init_hidden(batch_size)

        actions_list = []
        new_hidden_states = []

        for i in range(self.n_agents):
            obs_i = observations[:, i, :]
            hidden_i = self.hidden_states[i] if self.hidden_states is not None else None

            logits, new_hidden = self.actor(obs_i, hidden_i)

            if available_actions is not None:
                avail_i = available_actions[:, i, :]
                logits = logits.masked_fill(avail_i == 0, -1e10)

            if explore:
                probs = F.softmax(logits, dim=-1)
                actions_i = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                actions_i = logits.argmax(dim=-1)

            actions_list.append(actions_i)
            if new_hidden is not None:
                new_hidden_states.append(new_hidden)

        if new_hidden_states:
            self.hidden_states = torch.stack(new_hidden_states, dim=0)

        return torch.stack(actions_list, dim=1)

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE with value denormalization."""
        gamma = self.config.training.gamma
        gae_lambda = getattr(self.config.algorithm, 'gae_lambda', 0.95)

        # Denormalize values for GAE computation
        values_denorm = self.value_normalizer.denormalize(values)

        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        last_gae = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=self.device)
            else:
                next_value = values_denorm[:, t + 1]

            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values_denorm[:, t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
            returns[:, t] = advantages[:, t] + values_denorm[:, t]

        return returns * mask, advantages * mask

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform MAPPO training step."""
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

        # Get old values and log probs
        with torch.no_grad():
            state_flat = state.view(batch_size * seq_len, -1)
            old_values = self.critic(state_flat).view(batch_size, seq_len)

            # Collect old log probs
            old_log_probs_list = []
            for i in range(self.n_agents):
                obs_i = obs[:, :, i, :].reshape(batch_size * seq_len, -1)
                logits_i, _ = self.actor(obs_i, None)

                if available_actions is not None:
                    avail_i = available_actions[:, :, i, :].reshape(batch_size * seq_len, -1)
                    logits_i = logits_i.masked_fill(avail_i == 0, -1e10)

                log_probs_i = F.log_softmax(logits_i, dim=-1)
                actions_i = actions[:, :, i].reshape(batch_size * seq_len)
                old_log_prob_i = log_probs_i.gather(1, actions_i.unsqueeze(1)).squeeze(1)
                old_log_probs_list.append(old_log_prob_i.view(batch_size, seq_len))

            old_log_probs = torch.stack(old_log_probs_list, dim=-1).sum(dim=-1)

            # Compute returns and advantages
            returns, advantages = self._compute_gae(rewards, old_values, dones, mask)

            # Normalize advantages
            adv_mean = (advantages * mask).sum() / (mask.sum() + 1e-8)
            adv_std = ((advantages - adv_mean).pow(2) * mask).sum() / (mask.sum() + 1e-8)
            adv_std = torch.clamp(adv_std.sqrt(), min=1e-8)
            advantages = (advantages - adv_mean) / adv_std

        # Update value normalizer
        self.value_normalizer.update(returns[mask.bool()])

        # PPO updates
        clip_param = getattr(algo_config, 'clip_param', 0.2)
        n_epochs = getattr(algo_config, 'ppo_epochs', 10)
        value_loss_coef = getattr(algo_config, 'value_loss_coef', 0.5)
        entropy_coef = getattr(algo_config, 'entropy_coef', 0.01)
        max_grad_norm = self.config.training.grad_clip

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(n_epochs):
            # Forward pass for all agents
            log_probs_list = []
            entropy_list = []

            for i in range(self.n_agents):
                obs_i = obs[:, :, i, :].reshape(batch_size * seq_len, -1)
                logits_i, _ = self.actor(obs_i, None)

                if available_actions is not None:
                    avail_i = available_actions[:, :, i, :].reshape(batch_size * seq_len, -1)
                    logits_i = logits_i.masked_fill(avail_i == 0, -1e10)

                probs_i = F.softmax(logits_i, dim=-1)
                log_probs_i = F.log_softmax(logits_i, dim=-1)
                actions_i = actions[:, :, i].reshape(batch_size * seq_len)
                log_prob_i = log_probs_i.gather(1, actions_i.unsqueeze(1)).squeeze(1)
                entropy_i = -(probs_i * log_probs_i).sum(dim=-1)

                log_probs_list.append(log_prob_i.view(batch_size, seq_len))
                entropy_list.append(entropy_i.view(batch_size, seq_len))

            log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
            entropy = torch.stack(entropy_list, dim=-1).mean(dim=-1)

            # Critic forward
            state_flat = state.view(batch_size * seq_len, -1)
            values = self.critic(state_flat).view(batch_size, seq_len)

            # Policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2) * mask
            policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

            # Value loss (with normalization)
            returns_norm = self.value_normalizer.normalize(returns)
            value_loss = (values - returns_norm).pow(2) * mask
            value_loss = value_loss.sum() / (mask.sum() + 1e-8)

            # Entropy loss
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # Actor update
            actor_loss = policy_loss + entropy_coef * entropy_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            self.actor_optimizer.step()

            # Critic update
            critic_loss = value_loss_coef * value_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
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
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
            "value_normalizer": self.value_normalizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]
        if "value_normalizer" in checkpoint:
            self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])


class ValueNormalizer:
    """Running mean and std for value normalization."""

    def __init__(self, device: torch.device, clip: float = 10.0):
        self.device = device
        self.clip = clip
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = 1e-4

    def update(self, values: torch.Tensor) -> None:
        """Update running statistics."""
        if values.numel() == 0:
            return

        batch_mean = values.mean()
        batch_var = values.var()
        batch_count = values.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values."""
        return torch.clamp(
            (values - self.mean) / torch.sqrt(self.var + 1e-8),
            -self.clip, self.clip
        )

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values."""
        return values * torch.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self) -> Dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]
