"""AERIAL (Attention-Based Recurrence for Multi-Agent RL) Algorithm Implementation.

Based on "Attention-Based Recurrence for Multi-Agent Reinforcement Learning
under Stochastic Partial Observability" (Phan et al., 2023, ICML).

Key mechanism: Each agent maintains a GRU hidden state (belief) that is updated
using multi-head attention over OTHER agents' hidden states. Unlike VABL which
attends only to teammate *actions*, AERIAL attends to teammate *hidden states*
(or observations when available), requiring explicit inter-agent communication
of hidden representations at each timestep.

Architecture:
1. Each agent encodes its observation via MLP -> h_obs
2. Each agent maintains a GRU hidden state (belief)
3. At each step, agents share their hidden states (communication bandwidth)
4. Each agent uses MHA with its own hidden state as query and other agents'
   hidden states as keys/values
5. The attention context is concatenated with h_obs and fed into the GRU
6. Policy head outputs action logits from updated belief
7. Centralized critic for PPO training (CTDE)

Critical distinction from VABL: AERIAL shares hidden states across agents
(requires communication bandwidth), while VABL only uses observable teammate
actions (no communication needed). This is the key comparison point -- AERIAL
should perform well but requires more information exchange.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm


def orthogonal_init_(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    """Apply orthogonal initialization to a module's weights.

    Standard initialization used by MAPPO and PPO implementations
    for improved training stability.

    Args:
        module: Neural network module to initialize
        gain: Scaling factor for weights (sqrt(2) for ReLU, 0.01 for policy heads)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.constant_(param, 0)


class AERIALAgent(nn.Module):
    """AERIAL agent network with attention-based recurrence over hidden states.

    Each agent maintains a recurrent belief (GRU hidden state) that is updated
    at every timestep using:
    - Its own observation encoding
    - An attention-weighted summary of other agents' hidden states

    This requires sharing hidden state vectors across agents at each step
    (explicit communication), unlike VABL which only requires observing actions.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        use_orthogonal_init: bool = True,
    ):
        """Initialize AERIAL agent.

        Args:
            obs_dim: Dimension of agent observations
            n_actions: Number of available actions
            n_agents: Total number of agents in the environment
            embed_dim: Dimension for observation embeddings (d_e)
            hidden_dim: Dimension for GRU hidden state / belief (d_h)
            attention_heads: Number of attention heads for MHA
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.n_teammates = n_agents - 1

        # ---- Observation encoder: obs -> h_obs (embed_dim) ----
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # ---- Hidden state projection for attention ----
        # Project hidden_dim -> embed_dim so that query/key/value dimensions
        # are compatible with MHA (which operates in embed_dim space).
        self.hidden_proj = nn.Linear(hidden_dim, embed_dim)

        # ---- Multi-Head Attention over teammate hidden states ----
        # Query: agent's own projected hidden state
        # Key/Value: teammates' projected hidden states
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            batch_first=True,
        )

        # ---- GRU for belief update ----
        # Input: concatenation of h_obs (embed_dim) and attention context (embed_dim)
        # Hidden: belief state (hidden_dim)
        self.gru = nn.GRUCell(embed_dim + embed_dim, hidden_dim)

        # ---- Policy head: belief -> action logits ----
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Apply orthogonal initialization
        if use_orthogonal_init:
            self._apply_orthogonal_init()

    def _apply_orthogonal_init(self) -> None:
        """Apply orthogonal initialization to all network layers."""
        gain = np.sqrt(2)

        # Observation encoder
        for module in self.obs_encoder:
            orthogonal_init_(module, gain=gain)

        # Hidden state projection
        orthogonal_init_(self.hidden_proj, gain=gain)

        # GRU (gain=1.0 is standard for recurrent layers)
        orthogonal_init_(self.gru, gain=1.0)

        # Policy head -- small gain for near-uniform initial policy
        orthogonal_init_(self.policy_head, gain=0.01)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize belief state to zeros.

        Args:
            batch_size: Batch size for initialization

        Returns:
            Zero-initialized belief tensor [batch_size, hidden_dim]
        """
        return torch.zeros(
            batch_size, self.hidden_dim, device=next(self.parameters()).device
        )

    def forward(
        self,
        obs: torch.Tensor,
        prev_belief: torch.Tensor,
        teammate_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass implementing AERIAL's attention-based recurrence.

        For each agent i:
        1. h_obs <- obs_encoder(o^i_t)                       -- Encode observation
        2. q^i <- hidden_proj(b^i_{t-1})                     -- Project own hidden as query
        3. K, V <- hidden_proj(b^j_{t-1}) for j != i         -- Project teammates' hiddens
        4. c^i_t <- MHA(q^i, K, V)                           -- Attend over teammates
        5. b^i_t <- GRU([h_obs || c^i_t], b^i_{t-1})         -- Update belief
        6. a^i_t ~ pi(. | b^i_t)                             -- Sample action

        Args:
            obs: Agent observation [batch, obs_dim]
            prev_belief: Previous belief state [batch, hidden_dim]
            teammate_hidden_states: Teammates' hidden states [batch, n_teammates, hidden_dim]
                If None (first timestep with no teammate info), uses zero context.

        Returns:
            action_logits: Action logits [batch, n_actions]
            belief: Updated belief state [batch, hidden_dim]
            attention_weights: Attention weights over teammates [batch, n_teammates]
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Step 1: Encode observation
        h_obs = self.obs_encoder(obs)  # [batch, embed_dim]

        # Steps 2-4: Compute attention context from teammate hidden states
        if teammate_hidden_states is not None and teammate_hidden_states.shape[1] > 0:
            # Project own hidden state as query: [batch, 1, embed_dim]
            query = self.hidden_proj(prev_belief).unsqueeze(1)

            # Project teammate hidden states as key/value: [batch, n_teammates, embed_dim]
            tm_batch, tm_n, tm_dim = teammate_hidden_states.shape
            tm_flat = teammate_hidden_states.reshape(tm_batch * tm_n, tm_dim)
            kv_flat = self.hidden_proj(tm_flat)
            kv = kv_flat.reshape(tm_batch, tm_n, self.embed_dim)

            # Multi-head attention
            attn_output, attn_weights = self.attention_layer(
                query, kv, kv
            )
            # attn_output: [batch, 1, embed_dim] -> [batch, embed_dim]
            context = attn_output.squeeze(1)
            # attn_weights: [batch, 1, n_teammates] -> [batch, n_teammates]
            attention_weights = attn_weights.squeeze(1)
        else:
            # No teammate information available -- use zero context
            context = torch.zeros(batch_size, self.embed_dim, device=device)
            attention_weights = torch.zeros(
                batch_size, max(self.n_teammates, 1), device=device
            )

        # Step 5: Update belief via GRU
        gru_input = torch.cat([h_obs, context], dim=-1)  # [batch, 2 * embed_dim]
        belief = self.gru(gru_input, prev_belief)  # [batch, hidden_dim]

        # Step 6: Compute action logits from updated belief
        action_logits = self.policy_head(belief)  # [batch, n_actions]

        return action_logits, belief, attention_weights


class AERIALCritic(nn.Module):
    """Centralized critic for AERIAL using global state.

    Used during training (CTDE) to compute value estimates with full
    state information. Mirrors the CentralizedCritic used by VABL.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        use_orthogonal_init: bool = True,
    ):
        """Initialize centralized critic.

        Args:
            state_dim: Dimension of global state
            hidden_dim: Hidden dimension for critic network
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        if use_orthogonal_init:
            self._apply_orthogonal_init()

    def _apply_orthogonal_init(self) -> None:
        """Apply orthogonal initialization to critic network."""
        gain = np.sqrt(2)
        for module in self.network:
            if isinstance(module, nn.Linear):
                if module.out_features == 1:
                    orthogonal_init_(module, gain=1.0)
                else:
                    orthogonal_init_(module, gain=gain)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from global state.

        Args:
            state: Global state [batch, state_dim]

        Returns:
            Value estimate [batch] (scalar per batch element)
        """
        return self.network(state).squeeze(-1)


class ValueNormalizer:
    """Running mean and std for value normalization (PopArt-style).

    Stabilizes training by normalizing value targets to have zero mean
    and unit variance.
    """

    def __init__(self, device: torch.device, clip: float = 10.0, epsilon: float = 1e-8):
        self.device = device
        self.clip = clip
        self.epsilon = epsilon
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = 1e-4

    def update(self, values: torch.Tensor) -> None:
        """Update running statistics with new batch of values."""
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
        """Normalize values to zero mean and unit variance."""
        std = torch.sqrt(self.var + self.epsilon)
        return torch.clamp((values - self.mean) / std, -self.clip, self.clip)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original scale."""
        std = torch.sqrt(self.var + self.epsilon)
        return values * std + self.mean

    def state_dict(self) -> Dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


@register_algorithm("aerial")
class AERIAL(BaseAlgorithm):
    """AERIAL: Attention-Based Recurrence for Multi-Agent RL.

    Uses attention over teammate hidden states to update each agent's belief,
    enabling coordination in partially observable multi-agent environments.

    Key difference from VABL:
    - AERIAL shares hidden state vectors across agents (communication required)
    - VABL only observes teammate actions (no communication needed)
    - AERIAL should generally perform better (richer information) but at the
      cost of requiring inter-agent communication bandwidth.

    Trained with PPO using a centralized critic (CTDE paradigm).
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
        self.hidden_states = None  # Belief states for each agent [batch, n_agents, hidden_dim]
        super().__init__(config, n_agents, obs_shape, state_shape, n_actions, device)

        # Value normalization for training stability
        self.value_normalizer = ValueNormalizer(device=device)

    def _build_networks(self) -> None:
        """Build AERIAL agent and centralized critic networks."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = (
            self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]
        )

        hidden_dim = getattr(algo_config, "hidden_dim", 128)
        embed_dim = getattr(algo_config, "embed_dim", 64)
        attention_heads = getattr(algo_config, "attention_heads", 4)
        use_orthogonal_init = getattr(algo_config, "use_orthogonal_init", True)

        # Shared AERIAL agent (parameter sharing across all agents)
        self.agent = AERIALAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            use_orthogonal_init=use_orthogonal_init,
        ).to(self.device)

        # Centralized critic for PPO training
        critic_hidden_dim = getattr(algo_config, "critic_hidden_dim", hidden_dim)
        self.critic = AERIALCritic(
            state_dim=state_dim,
            hidden_dim=critic_hidden_dim,
            use_orthogonal_init=use_orthogonal_init,
        ).to(self.device)

    def _build_optimizers(self) -> None:
        """Build Adam optimizers for actor and critic."""
        base_lr = self.config.training.lr
        algo_config = self.config.algorithm

        actor_lr = getattr(algo_config, "actor_lr", base_lr)
        critic_lr = getattr(algo_config, "critic_lr", base_lr)

        self.actor_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )

    def init_hidden(self, batch_size: int) -> None:
        """Initialize belief states to zeros for all agents.

        Args:
            batch_size: Batch size (typically 1 for episode collection)
        """
        hidden_dim = getattr(self.config.algorithm, "hidden_dim", 128)
        # [batch_size, n_agents, hidden_dim]
        self.hidden_states = torch.zeros(
            batch_size, self.n_agents, hidden_dim, device=self.device
        )

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        """Get current belief states."""
        return self.hidden_states

    def _get_teammate_hiddens(
        self,
        all_hidden_states: torch.Tensor,
        agent_idx: int,
    ) -> torch.Tensor:
        """Extract teammate hidden states for a given agent.

        Args:
            all_hidden_states: All agents' hidden states [batch, n_agents, hidden_dim]
            agent_idx: Index of the querying agent

        Returns:
            Teammate hidden states [batch, n_teammates, hidden_dim]
        """
        # Gather all agents except agent_idx
        indices = [j for j in range(self.n_agents) if j != agent_idx]
        if len(indices) == 0:
            # Single-agent edge case
            return all_hidden_states[:, :0, :]
        idx_tensor = torch.tensor(indices, device=all_hidden_states.device)
        return all_hidden_states[:, idx_tensor, :]

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Select actions for all agents using AERIAL forward pass.

        At each timestep, all agents share their current hidden states. Each
        agent then attends to its teammates' hidden states to form a context,
        which is combined with the observation encoding to update its belief
        via a GRU. Actions are sampled from the policy head.

        Args:
            observations: Agent observations [batch, n_agents, obs_dim]
            available_actions: Mask of available actions [batch, n_agents, n_actions]
            explore: Whether to sample actions (True) or take argmax (False)

        Returns:
            Selected actions [batch, n_agents]
        """
        batch_size = observations.shape[0]

        if self.hidden_states is None:
            self.init_hidden(batch_size)

        # Ensure hidden states match batch size
        if self.hidden_states.shape[0] != batch_size:
            self.init_hidden(batch_size)

        with torch.no_grad():
            actions_list = []
            new_hidden_states = []

            for i in range(self.n_agents):
                obs_i = observations[:, i, :]  # [batch, obs_dim]
                belief_i = self.hidden_states[:, i, :]  # [batch, hidden_dim]

                # Get teammate hidden states (the key AERIAL mechanism)
                teammate_hiddens = self._get_teammate_hiddens(
                    self.hidden_states, i
                )  # [batch, n_teammates, hidden_dim]

                # Forward pass through AERIAL agent
                action_logits, new_belief, _ = self.agent(
                    obs_i, belief_i, teammate_hiddens
                )

                # Mask unavailable actions
                if available_actions is not None:
                    avail_i = available_actions[:, i, :]
                    action_logits = action_logits.masked_fill(avail_i == 0, -1e10)

                # Sample or argmax
                if explore:
                    action_probs = F.softmax(action_logits, dim=-1)
                    actions_i = torch.multinomial(action_probs, num_samples=1).squeeze(
                        -1
                    )
                else:
                    actions_i = action_logits.argmax(dim=-1)

                actions_list.append(actions_i)
                new_hidden_states.append(new_belief)

            # Update hidden states for next timestep
            self.hidden_states = torch.stack(new_hidden_states, dim=1)

        return torch.stack(actions_list, dim=1)  # [batch, n_agents]

    def _forward_pass_sequential(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        available_actions: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential forward pass through full episode for training.

        Reconstructs the hidden state trajectory by processing each timestep
        sequentially, sharing hidden states across agents at every step.

        Args:
            obs: Observations [batch, seq_len, n_agents, obs_dim]
            actions: Actions taken [batch, seq_len, n_agents]
            available_actions: Available actions [batch, seq_len, n_agents, n_actions] or None

        Returns:
            action_log_probs: Sum of per-agent log probs [batch, seq_len]
            entropy: Mean of per-agent entropy [batch, seq_len]
            all_hidden_states: Hidden states [batch, seq_len, n_agents, hidden_dim]
        """
        batch_size, seq_len = obs.shape[:2]
        hidden_dim = getattr(self.config.algorithm, "hidden_dim", 128)

        # Initialize hidden states for all agents
        hidden_states = torch.zeros(
            batch_size, self.n_agents, hidden_dim, device=self.device
        )

        all_log_probs = []
        all_entropy = []
        all_hidden = []

        for t in range(seq_len):
            obs_t = obs[:, t]  # [batch, n_agents, obs_dim]
            actions_t = actions[:, t]  # [batch, n_agents]

            agent_log_probs = []
            agent_entropy = []
            new_hidden_states = []

            for i in range(self.n_agents):
                obs_i = obs_t[:, i, :]  # [batch, obs_dim]
                belief_i = hidden_states[:, i, :]  # [batch, hidden_dim]

                # Get teammate hidden states (shared communication)
                teammate_hiddens = self._get_teammate_hiddens(hidden_states, i)

                # Forward pass
                action_logits, new_belief, _ = self.agent(
                    obs_i, belief_i, teammate_hiddens
                )

                # Mask unavailable actions
                if available_actions is not None:
                    avail_i = available_actions[:, t, i, :]
                    action_logits = action_logits.masked_fill(avail_i == 0, -1e10)

                # Compute log prob of taken action and entropy
                action_probs = F.softmax(action_logits, dim=-1)
                action_log_probs_i = F.log_softmax(action_logits, dim=-1)

                taken_action = actions_t[:, i].long()
                log_prob = action_log_probs_i.gather(
                    1, taken_action.unsqueeze(1)
                ).squeeze(1)
                ent = -(action_probs * action_log_probs_i).sum(dim=-1)

                agent_log_probs.append(log_prob)
                agent_entropy.append(ent)
                new_hidden_states.append(new_belief)

            # Update hidden states for next timestep
            hidden_states = torch.stack(new_hidden_states, dim=1)

            # Aggregate per-agent values
            # Sum log probs across agents (joint log prob)
            all_log_probs.append(torch.stack(agent_log_probs, dim=1).sum(dim=1))
            # Mean entropy across agents
            all_entropy.append(torch.stack(agent_entropy, dim=1).mean(dim=1))
            all_hidden.append(hidden_states)

        action_log_probs = torch.stack(all_log_probs, dim=1)  # [batch, seq_len]
        entropy = torch.stack(all_entropy, dim=1)  # [batch, seq_len]
        all_hidden_states = torch.stack(all_hidden, dim=1)  # [batch, seq_len, n_agents, hidden_dim]

        return action_log_probs, entropy, all_hidden_states

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages with value denormalization.

        Args:
            rewards: Episode rewards [batch, seq_len]
            values: Value estimates [batch, seq_len] (may be normalized)
            dones: Done flags [batch, seq_len]
            mask: Valid timestep mask [batch, seq_len]

        Returns:
            returns: Computed returns [batch, seq_len]
            advantages: GAE advantages [batch, seq_len]
        """
        gamma = self.config.training.gamma
        gae_lambda = getattr(self.config.algorithm, "gae_lambda", 0.95)

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

            delta = (
                rewards[:, t]
                + gamma * next_value * (1 - dones[:, t])
                - values_denorm[:, t]
            )
            last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
            returns[:, t] = advantages[:, t] + values_denorm[:, t]

        return returns * mask, advantages * mask

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform AERIAL training step using PPO with centralized critic.

        Uses multiple epochs of PPO updates with clipped objective,
        value clipping, and entropy bonus. Follows the same training
        pattern as CommNet and MAPPO in this codebase.

        Args:
            batch: Dictionary containing episode data with keys:
                obs: [batch, seq_len, n_agents, obs_dim]
                state: [batch, seq_len, state_dim]
                actions: [batch, seq_len, n_agents]
                rewards: [batch, seq_len]
                dones: [batch, seq_len]
                mask: [batch, seq_len]
                available_actions: [batch, seq_len, n_agents, n_actions] (optional)

        Returns:
            Dictionary of training metrics
        """
        algo_config = self.config.algorithm

        # Extract batch data
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
        state_flat = state.view(batch_size * seq_len, -1)

        # ---- Compute old values and log probs (no gradient) ----
        with torch.no_grad():
            old_values = self.critic(state_flat).view(batch_size, seq_len)

            old_log_probs, _, _ = self._forward_pass_sequential(
                obs, actions, available_actions
            )
            old_log_probs = old_log_probs.detach()
            old_values = old_values.detach()

            # Compute returns and advantages
            returns, advantages = self._compute_gae(rewards, old_values, dones, mask)

            # Update value normalizer with new returns
            self.value_normalizer.update(returns[mask.bool()])

            # Normalize advantages
            adv_mean = (advantages * mask).sum() / (mask.sum() + 1e-8)
            adv_std = ((advantages - adv_mean).pow(2) * mask).sum() / (
                mask.sum() + 1e-8
            )
            adv_std = torch.clamp(adv_std.sqrt(), min=1e-8)
            advantages = (advantages - adv_mean) / adv_std
            advantages = torch.clamp(advantages, -5.0, 5.0)

        # ---- PPO training with multiple epochs ----
        clip_param = getattr(algo_config, "clip_param", 0.2)
        n_epochs = getattr(algo_config, "ppo_epochs", 10)
        value_loss_coef = getattr(algo_config, "value_loss_coef", 0.5)
        entropy_coef = getattr(algo_config, "entropy_coef", 0.01)
        max_grad_norm = self.config.training.grad_clip

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        approx_kl = 0.0

        for epoch in range(n_epochs):
            # Forward pass through full episode
            action_log_probs, entropy, _ = self._forward_pass_sequential(
                obs, actions, available_actions
            )

            # Approximate KL for early stopping
            with torch.no_grad():
                log_ratio = action_log_probs - old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

            # Early stopping if KL divergence is too large
            target_kl = getattr(algo_config, "target_kl", 0.015)
            if approx_kl > 1.5 * target_kl and epoch > 0:
                break

            # ---- Policy loss (PPO clipped objective) ----
            ratio = torch.exp(action_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.0, 5.0)  # Numerical stability

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            )
            policy_loss = -torch.min(surr1, surr2) * mask
            policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

            # ---- Value loss (with normalization) ----
            values = self.critic(state_flat).view(batch_size, seq_len)
            returns_norm = self.value_normalizer.normalize(returns)
            value_loss = (values - returns_norm).pow(2) * mask
            value_loss = value_loss.sum() / (mask.sum() + 1e-8)

            # ---- Entropy loss ----
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # ---- Actor update ----
            actor_loss = policy_loss + entropy_coef * entropy_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
            self.actor_optimizer.step()

            # ---- Critic update ----
            critic_loss = value_loss_coef * value_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        n_updates = max(1, epoch + 1)
        self.training_step += 1

        return {
            "loss": (total_policy_loss + total_value_loss) / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": approx_kl,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "value_normalizer": self.value_normalizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]
        if "value_normalizer" in checkpoint:
            self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
