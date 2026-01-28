"""QMIX Algorithm Implementation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm
from marl_research.algorithms.networks import RNNAgent, QMixer


@register_algorithm("qmix")
class QMIX(BaseAlgorithm):
    """QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL."""

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
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        self.agent = RNNAgent(
            input_dim=obs_dim,
            hidden_dim=algo_config.agent_network.hidden_dim,
            rnn_hidden_dim=algo_config.agent_network.rnn_hidden_dim,
            n_actions=self.n_actions,
        ).to(self.device)

        self.target_agent = RNNAgent(
            input_dim=obs_dim,
            hidden_dim=algo_config.agent_network.hidden_dim,
            rnn_hidden_dim=algo_config.agent_network.rnn_hidden_dim,
            n_actions=self.n_actions,
        ).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=state_dim,
            embed_dim=algo_config.mixing_network.embed_dim,
            hypernet_hidden_dim=algo_config.mixing_network.hypernet_hidden_dim,
        ).to(self.device)

        self.target_mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=state_dim,
            embed_dim=algo_config.mixing_network.embed_dim,
            hypernet_hidden_dim=algo_config.mixing_network.hypernet_hidden_dim,
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self._init_weights(self.agent)
        self._init_weights(self.mixer)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                nn.init.orthogonal_(m.weight_ih)
                nn.init.orthogonal_(m.weight_hh)
                nn.init.constant_(m.bias_ih, 0)
                nn.init.constant_(m.bias_hh, 0)

    def _build_optimizers(self) -> None:
        params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(params, lr=self.config.training.lr, alpha=0.99, eps=1e-5)

    def init_hidden(self, batch_size: int) -> None:
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        ).to(self.device)
        self.target_hidden_states = self.target_agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        ).to(self.device)

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
    ) -> torch.Tensor:
        batch_size = observations.shape[0]

        if self.hidden_states is None:
            self.init_hidden(batch_size)

        q_values, self.hidden_states = self.agent(
            observations.view(batch_size * self.n_agents, -1),
            self.hidden_states.view(batch_size * self.n_agents, -1),
        )
        q_values = q_values.view(batch_size, self.n_agents, -1)
        self.hidden_states = self.hidden_states.view(batch_size, self.n_agents, -1)

        if available_actions is not None:
            q_values[available_actions == 0] = -1e10

        if explore:
            epsilon = self._get_epsilon()
            
            # Greedy actions (already masked above)
            greedy_actions = q_values.argmax(dim=-1)
            
            # Random actions (masked)
            if available_actions is not None:
                # Create probability distribution over available actions
                # Add small constant to avoid divide by zero for empty rows (though shouldn't happen)
                avail_float = available_actions.float()
                dist = avail_float / (avail_float.sum(dim=-1, keepdim=True) + 1e-10)
                
                # Sample from distribution
                # torch.multinomial needs 2D input
                flat_dist = dist.view(-1, self.n_actions)
                random_actions = torch.multinomial(flat_dist, 1).view(batch_size, self.n_agents)
                random_actions = random_actions.to(self.device)
            else:
                random_actions = torch.randint(0, self.n_actions, (batch_size, self.n_agents)).to(self.device)
            
            random_mask = (torch.rand(batch_size, self.n_agents) < epsilon).to(self.device)
            actions = torch.where(random_mask, random_actions, greedy_actions)
        else:
            actions = q_values.argmax(dim=-1)

        return actions

    def _get_epsilon(self) -> float:
        cfg = self.config.training
        epsilon = cfg.epsilon_finish + (cfg.epsilon_start - cfg.epsilon_finish) * max(
            0, 1 - self.training_step / cfg.epsilon_anneal_time
        )
        return epsilon

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        state = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device).long()
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        next_state = batch["next_state"].to(self.device)
        dones = batch["dones"].to(self.device)
        available_actions = batch.get("available_actions")
        next_available_actions = batch.get("next_available_actions")

        batch_size, seq_len = obs.shape[:2]

        self.init_hidden(batch_size)
        target_hidden = self.target_agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        ).to(self.device)

        q_values_list = []
        target_q_values_list = []

        for t in range(seq_len):
            q_vals, self.hidden_states = self.agent(
                obs[:, t].reshape(batch_size * self.n_agents, -1),
                self.hidden_states.reshape(batch_size * self.n_agents, -1),
            )
            q_values_list.append(q_vals.view(batch_size, self.n_agents, -1))

            with torch.no_grad():
                target_q_vals, target_hidden = self.target_agent(
                    next_obs[:, t].reshape(batch_size * self.n_agents, -1),
                    target_hidden.reshape(batch_size * self.n_agents, -1),
                )
                target_q_values_list.append(target_q_vals.view(batch_size, self.n_agents, -1))
                target_hidden = target_hidden.view(batch_size, self.n_agents, -1)

        q_values = torch.stack(q_values_list, dim=1)
        target_q_values = torch.stack(target_q_values_list, dim=1)

        chosen_q_values = torch.gather(q_values, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        if next_available_actions is not None:
            next_available_actions = next_available_actions.to(self.device)
            target_q_values[next_available_actions == 0] = -1e10

        target_max_q = target_q_values.max(dim=-1)[0]

        q_total = self.mixer(chosen_q_values, state)
        target_q_total = self.target_mixer(target_max_q, next_state)

        targets = rewards + self.config.training.gamma * (1 - dones) * target_q_total

        loss = F.mse_loss(q_total, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            self.config.training.grad_clip,
        )
        self.optimizer.step()

        self.training_step += 1

        if self.training_step % self.config.training.target_update_interval == 0:
            self.update_targets()

        return {"loss": loss.item(), "q_value": chosen_q_values.mean().item()}

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "mixer": self.mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_agent.load_state_dict(checkpoint["agent"])
        self.target_mixer.load_state_dict(checkpoint["mixer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
