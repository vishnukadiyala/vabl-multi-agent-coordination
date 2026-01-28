"""MAVEN (Multi-Agent Variational Exploration) Algorithm Implementation.

Based on "MAVEN: Multi-Agent Variational Exploration" (Mahajan et al., 2019).
Extends QMIX with a hierarchical latent space for coordinated exploration.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


class MAVENAgent(nn.Module):
    """MAVEN Agent conditioned on latent variable z."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim

        # Observation + latent encoder
        self.fc1 = nn.Linear(obs_dim + latent_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.fc_q = nn.Linear(rnn_hidden_dim, n_actions)

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with latent conditioning.

        Args:
            obs: Observations [batch, obs_dim]
            z: Latent variable [batch, latent_dim]
            hidden: RNN hidden state [batch, rnn_hidden_dim]

        Returns:
            q_values: Q-values [batch, n_actions]
            new_hidden: Updated hidden state
        """
        x = torch.cat([obs, z], dim=-1)
        x = F.relu(self.fc1(x))
        h = self.rnn(x, hidden)
        q = self.fc_q(h)
        return q, h

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)


class HierarchicalPolicy(nn.Module):
    """Hierarchical policy that samples latent z at episode start."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: state -> z distribution
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample latent z from state.

        Args:
            state: Global state [batch, state_dim]

        Returns:
            z: Sampled latent [batch, latent_dim]
            mu: Mean of distribution
            logvar: Log variance of distribution
        """
        h = self.encoder(state)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar


class MAVENMixer(nn.Module):
    """MAVEN Mixer (same as QMIX but with z conditioning)."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        latent_dim: int = 8,
        embed_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # State + z -> hypernetwork weights
        input_dim = state_dim + latent_dim

        # Hypernetwork for first layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(input_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(input_dim, embed_dim)

        # Hypernetwork for second layer weights
        self.hyper_w2 = nn.Sequential(
            nn.Linear(input_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        agent_qs: torch.Tensor,
        states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Mix individual Q-values into Q_tot.

        Args:
            agent_qs: Individual Q-values [batch, n_agents]
            states: Global state [batch, state_dim]
            z: Latent variable [batch, latent_dim]

        Returns:
            q_tot: Total Q-value [batch, 1]
        """
        batch_size = agent_qs.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)

        # Concatenate state and z
        sz = torch.cat([states, z], dim=-1)

        # First layer
        w1 = torch.abs(self.hyper_w1(sz)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(sz).view(batch_size, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(sz)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(sz).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze(-1)


class DiscriminatorNetwork(nn.Module):
    """Discriminator for mutual information maximization."""

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Joint action trajectory -> z prediction
        self.network = nn.Sequential(
            nn.Linear(n_agents * n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, actions_onehot: torch.Tensor) -> torch.Tensor:
        """Predict z from joint actions.

        Args:
            actions_onehot: One-hot actions [batch, n_agents, n_actions]

        Returns:
            z_pred: Predicted latent [batch, latent_dim]
        """
        x = actions_onehot.view(actions_onehot.shape[0], -1)
        return self.network(x)


@register_algorithm("maven")
class MAVEN(BaseAlgorithm):
    """MAVEN: Multi-Agent Variational Exploration.

    Key features:
    - Hierarchical latent variable z for coordinated exploration
    - Agents conditioned on shared z for consistent behavior
    - Mutual information maximization between z and trajectories
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
        self.current_z = None
        super().__init__(config, n_agents, obs_shape, state_shape, n_actions, device)

    def _build_networks(self) -> None:
        """Build MAVEN networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        hidden_dim = getattr(algo_config, 'hidden_dim', 64)
        rnn_hidden_dim = getattr(algo_config, 'rnn_hidden_dim', 64)
        latent_dim = getattr(algo_config, 'latent_dim', 8)
        embed_dim = getattr(algo_config, 'embed_dim', 32)

        # Agent network
        self.agent = MAVENAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)

        # Hierarchical policy
        self.hi_policy = HierarchicalPolicy(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Mixer
        self.mixer = MAVENMixer(
            n_agents=self.n_agents,
            state_dim=state_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            hypernet_hidden_dim=hidden_dim,
        ).to(self.device)

        # Discriminator for MI maximization
        self.discriminator = DiscriminatorNetwork(
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Target networks
        self.target_agent = MAVENAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=rnn_hidden_dim,
        ).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.target_mixer = MAVENMixer(
            n_agents=self.n_agents,
            state_dim=state_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            hypernet_hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.rnn_hidden_dim = rnn_hidden_dim
        self.latent_dim = latent_dim

    def _build_optimizers(self) -> None:
        """Build optimizers."""
        lr = self.config.training.lr

        # Main optimizer for agent and mixer
        main_params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(main_params, lr=lr)

        # Hierarchical policy optimizer
        self.hi_optimizer = torch.optim.Adam(self.hi_policy.parameters(), lr=lr)

        # Discriminator optimizer
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def init_hidden(self, batch_size: int) -> None:
        """Initialize hidden states."""
        self.hidden_states = self.agent.init_hidden(batch_size * self.n_agents, self.device)
        self.target_hidden_states = self.target_agent.init_hidden(batch_size * self.n_agents, self.device)

    def sample_z(self, state: torch.Tensor) -> torch.Tensor:
        """Sample latent z for episode."""
        z, _, _ = self.hi_policy(state)
        self.current_z = z
        return z

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        state: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions conditioned on latent z.

        Args:
            observations: [batch, n_agents, obs_dim]
            available_actions: [batch, n_agents, n_actions]
            explore: Whether to use epsilon-greedy
            state: Global state for sampling z [batch, state_dim]

        Returns:
            actions: [batch, n_agents]
        """
        batch_size = observations.shape[0]

        if self.hidden_states is None:
            self.init_hidden(batch_size)

        # Sample z if at start of episode
        if self.current_z is None and state is not None:
            self.sample_z(state)
        elif self.current_z is None:
            self.current_z = torch.zeros(batch_size, self.latent_dim, device=self.device)

        # Expand z for all agents
        z_expanded = self.current_z.unsqueeze(1).expand(-1, self.n_agents, -1)
        z_flat = z_expanded.reshape(batch_size * self.n_agents, -1)

        obs_flat = observations.view(batch_size * self.n_agents, -1)

        with torch.no_grad():
            q_values, new_hidden = self.agent(obs_flat, z_flat, self.hidden_states)

        self.hidden_states = new_hidden
        q_values = q_values.view(batch_size, self.n_agents, -1)

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

    def reset_z(self) -> None:
        """Reset latent z at end of episode."""
        self.current_z = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform MAVEN training step."""
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        state = batch["state"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        mask = batch.get("mask", torch.ones_like(rewards)).to(self.device)

        batch_size, seq_len = obs.shape[:2]
        gamma = self.config.training.gamma

        # Sample z for this batch (use first state)
        z, mu, logvar = self.hi_policy(state[:, 0])

        # KL divergence loss for VAE
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        # Initialize hidden states
        hidden = self.agent.init_hidden(batch_size * self.n_agents, self.device)
        target_hidden = self.target_agent.init_hidden(batch_size * self.n_agents, self.device)

        td_loss_sum = 0.0
        mi_loss_sum = 0.0

        z_expanded = z.unsqueeze(1).expand(-1, self.n_agents, -1)
        z_flat = z_expanded.reshape(batch_size * self.n_agents, -1)

        for t in range(seq_len):
            # Current Q-values
            obs_t = obs[:, t].view(batch_size * self.n_agents, -1)
            q_values, hidden = self.agent(obs_t, z_flat, hidden)
            q_values = q_values.view(batch_size, self.n_agents, -1)

            # Get Q-values for chosen actions
            actions_t = actions[:, t]
            chosen_q = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)

            # Compute Q_tot
            state_t = state[:, t]
            q_tot = self.mixer(chosen_q, state_t, z)

            # Target Q-values
            with torch.no_grad():
                next_obs_t = next_obs[:, t].view(batch_size * self.n_agents, -1)
                next_q, target_hidden = self.target_agent(next_obs_t, z_flat, target_hidden)
                next_q = next_q.view(batch_size, self.n_agents, -1)

                # Max Q for next state
                next_max_q = next_q.max(dim=-1)[0]
                next_state_t = next_state[:, t]
                target_q_tot = self.target_mixer(next_max_q, next_state_t, z)

                # TD target
                reward_t = rewards[:, t].unsqueeze(-1)
                done_t = dones[:, t].unsqueeze(-1)
                target = reward_t + gamma * (1 - done_t) * target_q_tot

            # TD loss
            mask_t = mask[:, t].unsqueeze(-1)
            td_error = (q_tot - target) * mask_t
            td_loss = (td_error ** 2).sum() / (mask_t.sum() + 1e-8)
            td_loss_sum += td_loss

            # Mutual information loss (discriminator)
            actions_onehot = F.one_hot(actions_t, self.n_actions).float()
            z_pred = self.discriminator(actions_onehot)
            mi_loss = F.mse_loss(z_pred, z.detach())
            mi_loss_sum += mi_loss

        # Total losses
        td_loss_avg = td_loss_sum / seq_len
        mi_loss_avg = mi_loss_sum / seq_len
        mi_weight = getattr(self.config.algorithm, 'mi_weight', 0.01)
        kl_weight = getattr(self.config.algorithm, 'kl_weight', 0.001)

        # Update main networks
        total_loss = td_loss_avg + mi_weight * mi_loss_avg + kl_weight * kl_loss
        self.optimizer.zero_grad()
        self.hi_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()) + list(self.hi_policy.parameters()),
            self.config.training.grad_clip
        )
        self.optimizer.step()
        self.hi_optimizer.step()

        # Update discriminator
        self.disc_optimizer.zero_grad()
        mi_loss_avg.backward()
        self.disc_optimizer.step()

        # Update target networks
        target_update_interval = getattr(self.config.training, 'target_update_interval', 200)
        if self.training_step % target_update_interval == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.training_step += 1

        return {
            "loss": total_loss.item(),
            "td_loss": td_loss_avg.item(),
            "mi_loss": mi_loss_avg.item(),
            "kl_loss": kl_loss.item(),
            "epsilon": self._get_epsilon(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "hi_policy": self.hi_policy.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "hi_optimizer": self.hi_optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.hi_policy.load_state_dict(checkpoint["hi_policy"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.target_agent.load_state_dict(checkpoint["target_agent"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.hi_optimizer.load_state_dict(checkpoint["hi_optimizer"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        self.training_step = checkpoint["training_step"]
