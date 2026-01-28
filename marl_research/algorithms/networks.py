"""Neural network components for MARL algorithms."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """RNN-based agent network for value-based methods."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rnn_hidden_dim: int,
        n_actions: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.rnn_hidden_dim)

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    """QMIX mixing network."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int,
        hypernet_hidden_dim: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * embed_dim),
        )

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, embed_dim),
        )

        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        seq_len = agent_qs.shape[1] if len(agent_qs.shape) > 2 else 1

        state_flat = state.reshape(-1, state.shape[-1])
        agent_qs_flat = agent_qs.reshape(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(state_flat))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)

        b1 = self.hyper_b1(state_flat)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs_flat, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state_flat))
        w2 = w2.view(-1, self.embed_dim, 1)

        b2 = self.hyper_b2(state_flat)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(batch_size, seq_len)

        return q_total


class MLPNetwork(nn.Module):
    """Simple MLP network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()

        activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        layers = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for policy gradient methods."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        share_backbone: bool = False,
    ):
        super().__init__()
        self.share_backbone = share_backbone

        if share_backbone:
            self.backbone = MLPNetwork(obs_dim, hidden_dim, hidden_dim, num_layers - 1)
            self.actor_head = nn.Linear(hidden_dim, n_actions)
            self.critic_head = nn.Linear(hidden_dim, 1)
        else:
            self.actor = MLPNetwork(obs_dim, n_actions, hidden_dim, num_layers)
            self.critic = MLPNetwork(obs_dim, 1, hidden_dim, num_layers)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.share_backbone:
            features = self.backbone(obs)
            return self.actor_head(features), self.critic_head(features)
        return self.actor(obs), self.critic(obs)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.share_backbone:
            features = self.backbone(obs)
            logits = self.actor_head(features)
        else:
            logits = self.actor(obs)
        return F.softmax(logits, dim=-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if self.share_backbone:
            features = self.backbone(obs)
            return self.critic_head(features)
        return self.critic(obs)
