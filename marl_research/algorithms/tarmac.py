"""TarMAC (Targeted Multi-Agent Communication) Algorithm Implementation.

Based on "TarMAC: Targeted Multi-Agent Communication" (Das et al., 2019, ICML).
Agents communicate via attention-based targeted messages: each agent produces a
message vector and a signature key, and receivers attend to messages using their
own query against senders' keys via scaled dot-product attention.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


class TarMACModule(nn.Module):
    """TarMAC communication module with targeted soft-attention messaging.

    Architecture per agent per communication round:
        1. Encode observation via MLP -> hidden state h_i
        2. Produce message m_i and signature key k_i from h_i
        3. Produce query q_i from h_i
        4. Attention: w_ij = softmax(q_i . k_j / sqrt(d_key)) for j != i
        5. Attended message c_i = sum_j w_ij * m_j
        6. GRU update: h_i = GRU(c_i, h_i)
    After all communication rounds, the policy head maps h_i -> action logits.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        hidden_dim: int = 128,
        message_dim: int = 64,
        key_dim: int = 64,
        comm_rounds: int = 1,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.key_dim = key_dim
        self.comm_rounds = comm_rounds

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Communication heads (one set per round for multi-round communication)
        self.message_heads = nn.ModuleList([
            nn.Linear(hidden_dim, message_dim) for _ in range(comm_rounds)
        ])
        self.key_heads = nn.ModuleList([
            nn.Linear(hidden_dim, key_dim) for _ in range(comm_rounds)
        ])
        self.query_heads = nn.ModuleList([
            nn.Linear(hidden_dim, key_dim) for _ in range(comm_rounds)
        ])

        # GRU for integrating attended messages into hidden state (one per round)
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(message_dim, hidden_dim) for _ in range(comm_rounds)
        ])

        # Policy head: maps final hidden state to action logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(
        self,
        obs: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with targeted communication.

        Args:
            obs: Observations [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions] or None

        Returns:
            logits: Action logits [batch, n_agents, n_actions]
            attention_weights: Attention weights from last round [batch, n_agents, n_agents]
        """
        batch_size = obs.shape[0]

        # Encode observations -> initial hidden states
        # obs: [batch, n_agents, obs_dim]
        h = self.obs_encoder(obs)  # [batch, n_agents, hidden_dim]

        last_attn_weights = None

        # Communication rounds
        for r in range(self.comm_rounds):
            # Produce messages, keys, queries from current hidden states
            messages = self.message_heads[r](h)  # [batch, n_agents, message_dim]
            keys = self.key_heads[r](h)          # [batch, n_agents, key_dim]
            queries = self.query_heads[r](h)     # [batch, n_agents, key_dim]

            # Scaled dot-product attention: each agent attends to all other agents
            # scores[i, j] = q_i . k_j / sqrt(d_key)
            scale = self.key_dim ** 0.5
            # [batch, n_agents, n_agents]
            attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale

            # Clamp scores to prevent numerical instability in softmax
            attn_scores = torch.clamp(attn_scores, -20.0, 20.0)

            # Mask self-attention: agent should not attend to its own message
            self_mask = torch.eye(self.n_agents, device=obs.device).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(self_mask.bool(), float("-inf"))

            # Handle edge case: single agent (all -inf row)
            if self.n_agents > 1:
                attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, n_agents, n_agents]
                # Replace any NaN from softmax (can happen if all -inf)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            else:
                attn_weights = torch.zeros_like(attn_scores)

            last_attn_weights = attn_weights

            # Attended message: c_i = sum_j w_ij * m_j
            # [batch, n_agents, message_dim]
            attended_messages = torch.bmm(attn_weights, messages)

            # GRU update: integrate attended message into hidden state
            # GRUCell expects: input [batch * n_agents, message_dim],
            #                  hidden [batch * n_agents, hidden_dim]
            h_flat = h.view(batch_size * self.n_agents, self.hidden_dim)
            c_flat = attended_messages.view(batch_size * self.n_agents, self.message_dim)
            h_flat = self.gru_cells[r](c_flat, h_flat)
            # Protect against NaN in GRU output
            h_flat = torch.nan_to_num(h_flat, nan=0.0)
            h = h_flat.view(batch_size, self.n_agents, self.hidden_dim)

        # Policy output from final hidden state
        logits = self.policy_head(h)  # [batch, n_agents, n_actions]

        # Mask unavailable actions
        if available_actions is not None:
            logits = logits.masked_fill(available_actions == 0, -1e10)

        return logits, last_attn_weights


class TarMACCritic(nn.Module):
    """Centralized critic for TarMAC using global state (CTDE)."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
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
        """Apply orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: Global state [batch, state_dim]

        Returns:
            value: State value [batch]
        """
        return self.critic(state).squeeze(-1)


@register_algorithm("tarmac")
class TarMAC(BaseAlgorithm):
    """TarMAC: Targeted Multi-Agent Communication.

    Each agent produces a message vector and a signature key. Other agents attend
    to messages using their own query and the senders' keys via soft attention,
    enabling targeted (not broadcast) communication. Trained with PPO using a
    centralized critic (CTDE).

    Key features:
    - Targeted communication via soft attention (query-key-message)
    - Multi-round communication support
    - GRU-based message integration
    - Centralized critic for PPO training
    - Orthogonal initialization for stable training
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
        """Build TarMAC actor and centralized critic networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        hidden_dim = getattr(algo_config, "hidden_dim", 128)
        message_dim = getattr(algo_config, "message_dim", 64)
        key_dim = getattr(algo_config, "key_dim", 64)
        comm_rounds = getattr(algo_config, "comm_rounds", 1)

        self.tarmac = TarMACModule(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            key_dim=key_dim,
            comm_rounds=comm_rounds,
        ).to(self.device)

        self.critic = TarMACCritic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

    def _build_optimizers(self) -> None:
        """Build separate optimizers for actor and critic."""
        lr = self.config.training.lr
        self.actor_optimizer = torch.optim.Adam(self.tarmac.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

    def init_hidden(self, batch_size: int) -> None:
        """TarMAC uses feedforward obs encoding + GRU within each step.

        No persistent hidden state across timesteps (communication is
        within-timestep only), so this is a no-op.
        """
        pass

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions for all agents with targeted communication.

        Args:
            observations: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions] or None
            explore: Whether to sample from policy or take argmax

        Returns:
            actions: [batch, n_agents]
        """
        with torch.no_grad():
            logits, _ = self.tarmac(observations, available_actions)

        if explore:
            probs = F.softmax(logits, dim=-1)
            # Protect against NaN/zero probabilities
            probs = torch.nan_to_num(probs, nan=1.0 / self.n_actions)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            batch_size = observations.shape[0]
            actions = torch.zeros(
                batch_size, self.n_agents, dtype=torch.long, device=self.device
            )
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
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: [batch, seq_len]
            values: [batch, seq_len]
            dones: [batch, seq_len]
            mask: [batch, seq_len]

        Returns:
            returns: [batch, seq_len]
            advantages: [batch, seq_len]
        """
        gamma = self.config.training.gamma
        gae_lambda = getattr(self.config.algorithm, "gae_lambda", 0.95)

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
        """Perform TarMAC training step using PPO with centralized critic.

        Args:
            batch: Dictionary with keys: obs, state, actions, rewards, dones, mask,
                   available_actions

        Returns:
            Dictionary of training metrics
        """
        algo_config = self.config.algorithm

        # Unpack batch
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

        # Flatten time dimension for network forward passes
        obs_flat = obs.view(batch_size * seq_len, self.n_agents, -1)
        state_flat = state.view(batch_size * seq_len, -1)
        actions_flat = actions.view(batch_size * seq_len, self.n_agents)
        avail_flat = None
        if available_actions is not None:
            avail_flat = available_actions.view(batch_size * seq_len, self.n_agents, -1)

        # Compute old values and log probs (detached)
        with torch.no_grad():
            old_values = self.critic(state_flat).view(batch_size, seq_len)

            old_logits, _ = self.tarmac(obs_flat, avail_flat)
            old_log_probs_all = F.log_softmax(old_logits, dim=-1)

            # Gather log probs for taken actions and sum across agents
            # old_log_probs_all: [batch*seq, n_agents, n_actions]
            # actions_flat: [batch*seq, n_agents]
            old_log_probs = old_log_probs_all.gather(
                2, actions_flat.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1).view(batch_size, seq_len)

            # Compute GAE
            returns, advantages = self._compute_gae(rewards, old_values, dones, mask)

            # Normalize advantages
            adv_mean = (advantages * mask).sum() / (mask.sum() + 1e-8)
            adv_std = ((advantages - adv_mean).pow(2) * mask).sum() / (mask.sum() + 1e-8)
            adv_std = torch.clamp(adv_std.sqrt(), min=1e-8)
            advantages = (advantages - adv_mean) / adv_std
            advantages = torch.clamp(advantages, -10, 10)

        # PPO hyperparameters
        clip_param = getattr(algo_config, "clip_param", 0.2)
        n_epochs = getattr(algo_config, "ppo_epochs", 10)
        value_loss_coef = getattr(algo_config, "value_loss_coef", 0.5)
        entropy_coef = getattr(algo_config, "entropy_coef", 0.01)
        max_grad_norm = self.config.training.grad_clip

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_attn_entropy = 0.0

        for epoch in range(n_epochs):
            # Actor forward pass
            logits, attn_weights = self.tarmac(obs_flat, avail_flat)
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)

            # Log probs for taken actions, summed across agents
            log_probs = log_probs_all.gather(
                2, actions_flat.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1).view(batch_size, seq_len)

            # Action entropy (averaged across agents)
            entropy = -(probs * log_probs_all).sum(dim=-1).mean(dim=-1).view(
                batch_size, seq_len
            )

            # Attention entropy (for logging: how targeted the communication is)
            if attn_weights is not None and self.n_agents > 1:
                # attn_weights: [batch*seq, n_agents, n_agents]
                attn_log = torch.log(attn_weights + 1e-10)
                attn_entropy = -(attn_weights * attn_log).sum(dim=-1).mean(dim=-1)
                attn_entropy = attn_entropy.view(batch_size, seq_len)
                mean_attn_entropy = (attn_entropy * mask).sum() / (mask.sum() + 1e-8)
            else:
                mean_attn_entropy = torch.tensor(0.0, device=self.device)

            # Critic forward pass
            values = self.critic(state_flat).view(batch_size, seq_len)

            # PPO clipped policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0, 10)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2) * mask
            policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

            # Value loss
            value_loss = (values - returns).pow(2) * mask
            value_loss = value_loss.sum() / (mask.sum() + 1e-8)

            # Entropy loss (encourage exploration)
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # Actor update
            actor_loss = policy_loss + entropy_coef * entropy_loss
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                continue  # Skip this epoch if loss is NaN
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tarmac.parameters(), max_grad_norm)
            self.actor_optimizer.step()

            # Critic update
            critic_loss = value_loss_coef * value_loss
            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                continue
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_attn_entropy += mean_attn_entropy.item()

        self.training_step += 1

        return {
            "loss": (total_policy_loss + total_value_loss) / n_epochs,
            "policy_loss": total_policy_loss / n_epochs,
            "value_loss": total_value_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
            "attention_entropy": total_attn_entropy / n_epochs,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "tarmac": self.tarmac.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.tarmac.load_state_dict(checkpoint["tarmac"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]
