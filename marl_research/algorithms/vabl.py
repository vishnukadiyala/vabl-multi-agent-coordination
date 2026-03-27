"""VABL (Variational Attention-based Belief Learning) Algorithm Implementation.

STABILITY IMPROVEMENTS (v2):
- Auxiliary loss annealing (disable after warmup phase)
- Target critic network with soft updates
- Adaptive entropy decay
- KL divergence monitoring with early stopping
- More aggressive learning rate scheduling
"""

from typing import Dict, Optional, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm
from marl_research.algorithms.vabl_networks import VABLAgent, CentralizedCritic


class ValueNormalizer:
    """Running mean and std for value normalization (PopArt-style).

    This stabilizes training by normalizing value targets to have
    zero mean and unit variance, preventing value function from
    chasing moving targets during training.
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

        # Welford's online algorithm for numerical stability
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
        normalized = (values - self.mean) / std
        return torch.clamp(normalized, -self.clip, self.clip)

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


@register_algorithm("vabl")
class VABL(BaseAlgorithm):
    """VABL: Variational Attention-based Belief Learning for Multi-Agent RL.

    Uses attention over visible teammate actions to update belief states,
    enabling implicit coordination in partially observable environments.
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
        self.hidden_states = None  # Belief states for each agent
        super().__init__(config, n_agents, obs_shape, state_shape, n_actions, device)

        # Stability tracking
        self.best_reward = float('-inf')
        self.episodes_without_improvement = 0
        self.current_entropy_coef = self.config.algorithm.entropy_coef
        self.aux_lambda_current = self.config.algorithm.aux_lambda

        # Ablation parameters from config (with defaults for backwards compatibility)
        algo_config = self.config.algorithm
        self.warmup_steps = getattr(algo_config, 'warmup_steps', 50)
        self.aux_decay_rate = getattr(algo_config, 'aux_decay_rate', 0.995)
        self.min_aux_lambda = getattr(algo_config, 'min_aux_lambda', 0.05)
        self.stop_gradient_belief = getattr(algo_config, 'stop_gradient_belief', False)

        # Value normalization for training stability (PopArt-style)
        self.use_value_norm = getattr(algo_config, 'use_value_norm', True)
        self.value_normalizer = ValueNormalizer(device=device)

    def _build_networks(self) -> None:
        """Build VABL agent networks and centralized critic with target network."""
        algo_config = self.config.algorithm
        obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else self.obs_shape[-1]
        state_dim = self.state_shape[0] if len(self.state_shape) == 1 else self.state_shape[-1]

        # Get orthogonal init setting from config (default True, like MAPPO)
        use_orthogonal_init = getattr(algo_config, 'use_orthogonal_init', True)

        # Build VABL agent (shared parameters across all agents)
        self.agent = VABLAgent(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            n_agents=self.n_agents,
            embed_dim=algo_config.embed_dim,
            hidden_dim=algo_config.hidden_dim,
            attention_dim=algo_config.attention_dim,
            aux_hidden_dim=algo_config.aux_hidden_dim,
            attention_heads=getattr(algo_config, 'attention_heads', 4),
            use_orthogonal_init=use_orthogonal_init,
            use_identity_encoding=getattr(algo_config, 'use_identity_encoding', True),
        ).to(self.device)

        # Ablation control
        self.agent.use_attention = getattr(algo_config, 'use_attention', True)
        self.use_aux_loss = getattr(algo_config, 'use_aux_loss', True)

        # Centralized critic for PPO training
        self.critic = CentralizedCritic(
            state_dim=state_dim,
            n_agents=self.n_agents,
            hidden_dim=algo_config.critic_hidden_dim,
            use_orthogonal_init=use_orthogonal_init,
        ).to(self.device)

        # Target critic for stable value estimation
        self.target_critic = CentralizedCritic(
            state_dim=state_dim,
            n_agents=self.n_agents,
            hidden_dim=algo_config.critic_hidden_dim,
            use_orthogonal_init=use_orthogonal_init,
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Freeze target critic
        for param in self.target_critic.parameters():
            param.requires_grad = False

    def _build_optimizers(self) -> None:
        """Build Adam optimizers for actor and critic (matching MAPPO setup)."""
        algo_config = self.config.algorithm
        base_lr = self.config.training.lr

        # Use separate learning rates from config, or fall back to base_lr
        actor_lr = getattr(algo_config, 'actor_lr', base_lr)
        critic_lr = getattr(algo_config, 'critic_lr', base_lr)

        # Separate optimizers for actor (agent) and critic (matching MAPPO)
        self.actor_optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5
        )

        # Optional LR schedulers (disabled by default for stability)
        self.actor_scheduler = None
        self.critic_scheduler = None

    def _soft_update_target(self, tau: float = 0.005) -> None:
        """Soft update target critic network."""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def init_hidden(self, batch_size: int) -> None:
        """Initialize belief states to zeros for all agents.

        Args:
            batch_size: Batch size (typically 1 for episode collection)
        """
        hidden_dim = self.config.algorithm.hidden_dim
        # [batch_size, n_agents, hidden_dim]
        self.hidden_states = torch.zeros(
            batch_size, self.n_agents, hidden_dim, device=self.device
        )

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        """Get current belief states."""
        return self.hidden_states

    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
        prev_actions: Optional[torch.Tensor] = None,
        visibility_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select actions for all agents using VABL forward pass.

        Args:
            observations: Agent observations [batch, n_agents, obs_dim]
            available_actions: Mask of available actions [batch, n_agents, n_actions]
            explore: Whether to sample actions (True) or take argmax (False)
            prev_actions: Previous actions taken [batch, n_agents] (indices, not one-hot)
            visibility_masks: Which teammates each agent can see [batch, n_agents, n_agents-1]

        Returns:
            Selected actions [batch, n_agents]
        """
        batch_size = observations.shape[0]

        if self.hidden_states is None:
            self.init_hidden(batch_size)

        # Ensure hidden states match batch size
        if self.hidden_states.shape[0] != batch_size:
            self.init_hidden(batch_size)

        actions_list = []
        new_hidden_states = []

        for i in range(self.n_agents):
            # Get observation for agent i
            obs_i = observations[:, i, :]  # [batch, obs_dim]
            belief_i = self.hidden_states[:, i, :]  # [batch, hidden_dim]

            # Get previous teammate actions (excluding agent i's own action)
            prev_teammate_actions = None
            teammate_indices = [j for j in range(self.n_agents) if j != i]
            teammate_idx_tensor = torch.tensor(teammate_indices, device=self.device)
            if prev_actions is not None:
                # Build one-hot encoding of teammate actions
                prev_teammate_actions = torch.zeros(
                    batch_size, self.n_agents - 1, self.n_actions, device=self.device
                )
                for idx, j in enumerate(teammate_indices):
                    prev_teammate_actions[:, idx, :] = F.one_hot(
                        prev_actions[:, j].long(), num_classes=self.n_actions
                    ).float()

            # Get visibility mask for agent i
            vis_mask_i = None
            if visibility_masks is not None:
                vis_mask_i = visibility_masks[:, i, :]  # [batch, n_agents-1]

            # Forward pass through VABL agent
            action_logits, new_belief, _, _ = self.agent(
                obs_i, belief_i, prev_teammate_actions, vis_mask_i,
                teammate_indices=teammate_idx_tensor,
            )

            # Mask unavailable actions
            if available_actions is not None:
                avail_i = available_actions[:, i, :]  # [batch, n_actions]
                action_logits = action_logits.masked_fill(avail_i == 0, -1e10)

            # Sample or argmax
            if explore:
                action_probs = F.softmax(action_logits, dim=-1)
                actions_i = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
            else:
                actions_i = action_logits.argmax(dim=-1)

            actions_list.append(actions_i)
            new_hidden_states.append(new_belief)

        # Update hidden states
        self.hidden_states = torch.stack(new_hidden_states, dim=1)

        # Stack actions
        actions = torch.stack(actions_list, dim=1)  # [batch, n_agents]

        return actions

    def _compute_returns_and_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages using target critic for bootstrap.

        Uses denormalized values for GAE computation when value normalization
        is enabled, ensuring correct advantage estimation.

        Args:
            rewards: Episode rewards [batch, seq_len]
            values: Value estimates [batch, seq_len] (may be normalized)
            dones: Done flags [batch, seq_len]
            mask: Valid timestep mask [batch, seq_len]

        Returns:
            returns: Computed returns [batch, seq_len] (in original scale)
            advantages: GAE advantages [batch, seq_len]
        """
        gamma = self.config.training.gamma
        gae_lambda = getattr(self.config.algorithm, 'gae_lambda', 0.95)

        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        # Denormalize values for GAE computation if using value normalization
        if self.use_value_norm:
            values_denorm = self.value_normalizer.denormalize(values)
        else:
            values_denorm = values

        # Compute returns and advantages using GAE
        last_gae = torch.zeros(batch_size, device=self.device)
        last_value = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = last_value
            else:
                next_value = values_denorm[:, t + 1]

            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values_denorm[:, t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * last_gae
            advantages[:, t] = last_gae
            returns[:, t] = advantages[:, t] + values_denorm[:, t]

        # Apply mask
        advantages = advantages * mask
        returns = returns * mask

        return returns, advantages

    def _compute_auxiliary_loss(
        self,
        beliefs: torch.Tensor,
        next_actions: torch.Tensor,
        next_visibility_masks: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Compute auxiliary loss for teammate action prediction.

        Implements Eq. 10 from the paper:
        L_aux = E[sum_{j!=i} m^{i<-j}_{t+1} * CE(pi_hat(.|b_t), a^j_{t+1})]

        Args:
            beliefs: Agent belief states [batch, seq_len, n_agents, hidden_dim]
            next_actions: Next timestep actions [batch, seq_len, n_agents]
            next_visibility_masks: Visibility masks for next timestep [batch, seq_len, n_agents, n_agents-1]
            mask: Valid timestep mask [batch, seq_len]

        Returns:
            aux_loss: Auxiliary prediction loss (scalar)
            aux_accuracy: Average prediction accuracy
        """
        batch_size, seq_len = beliefs.shape[:2]
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0

        # Stop gradient ablation: detach beliefs to prevent gradients from
        # auxiliary loss affecting the belief encoder
        if self.stop_gradient_belief:
            beliefs = beliefs.detach()

        for i in range(self.n_agents):
            # Get beliefs for agent i
            belief_i = beliefs[:, :, i, :]  # [batch, seq_len, hidden_dim]

            # Get auxiliary predictions
            aux_logits = self.agent.aux_head(belief_i)  # [batch, seq_len, n_actions * n_teammates]
            aux_logits = aux_logits.view(batch_size, seq_len, self.n_agents - 1, self.n_actions)

            # Get target teammate actions (excluding agent i)
            teammate_indices = [j for j in range(self.n_agents) if j != i]

            for idx, j in enumerate(teammate_indices):
                # Target: next action of teammate j
                target_actions = next_actions[:, :, j].long()  # [batch, seq_len]

                # Predictions: from current belief
                pred_logits = aux_logits[:, :, idx, :]  # [batch, seq_len, n_actions]

                # Visibility mask for this teammate
                if next_visibility_masks is not None:
                    vis_mask = next_visibility_masks[:, :, i, idx]  # [batch, seq_len]
                else:
                    vis_mask = torch.ones(batch_size, seq_len, device=self.device)

                # Combined mask: valid timestep AND visible teammate
                combined_mask = mask * vis_mask

                if combined_mask.sum() > 0:
                    # Compute cross-entropy loss (only for visible predictions)
                    ce_loss = F.cross_entropy(
                        pred_logits.view(-1, self.n_actions),
                        target_actions.view(-1),
                        reduction='none',
                    ).view(batch_size, seq_len)

                    # Apply mask and average
                    masked_loss = (ce_loss * combined_mask).sum() / (combined_mask.sum() + 1e-8)
                    total_loss = total_loss + masked_loss

                    # Compute accuracy
                    pred_actions = pred_logits.argmax(dim=-1)
                    correct = (pred_actions == target_actions).float() * combined_mask
                    total_correct += correct.sum().item()
                    total_predictions += combined_mask.sum().item()

        # Average over all agent-teammate pairs
        aux_loss = total_loss / (self.n_agents * (self.n_agents - 1))
        aux_accuracy = total_correct / (total_predictions + 1e-8)

        return aux_loss, aux_accuracy

    def _forward_pass_collect(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        visibility_masks: Optional[torch.Tensor],
        available_actions: Optional[torch.Tensor],
        algo_config,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass to collect beliefs, log probs, and entropy.

        Returns:
            beliefs: [batch, seq_len, n_agents, hidden_dim]
            action_log_probs: [batch, seq_len]
            entropy: [batch, seq_len]
        """
        batch_size, seq_len = obs.shape[:2]

        all_beliefs = []
        all_action_log_probs = []
        all_entropy = []

        hidden_states_list = [
            torch.zeros(batch_size, algo_config.hidden_dim, device=self.device)
            for _ in range(self.n_agents)
        ]

        for t in range(seq_len):
            obs_t = obs[:, t]
            actions_t = actions[:, t]
            prev_actions_t = None if t == 0 else actions[:, t - 1]

            vis_masks_t = None
            if visibility_masks is not None:
                vis_masks_t = visibility_masks[:, t]

            agent_beliefs = []
            agent_log_probs = []
            agent_entropy = []

            for i in range(self.n_agents):
                obs_i = obs_t[:, i, :]
                belief_i = hidden_states_list[i]

                prev_teammate_actions = None
                teammate_indices = [j for j in range(self.n_agents) if j != i]
                teammate_idx_tensor = torch.tensor(teammate_indices, device=self.device)
                if prev_actions_t is not None:
                    prev_teammate_actions = torch.zeros(
                        batch_size, self.n_agents - 1, self.n_actions, device=self.device
                    )
                    for idx, j in enumerate(teammate_indices):
                        prev_teammate_actions[:, idx, :] = F.one_hot(
                            prev_actions_t[:, j].long(), num_classes=self.n_actions
                        ).float()

                vis_mask_i = None
                if vis_masks_t is not None:
                    vis_mask_i = vis_masks_t[:, i, :]

                action_logits, new_belief, _, _ = self.agent(
                    obs_i, belief_i, prev_teammate_actions, vis_mask_i,
                    teammate_indices=teammate_idx_tensor,
                )

                if available_actions is not None:
                    avail_i = available_actions[:, t, i, :]
                    action_logits = action_logits.masked_fill(avail_i == 0, -1e10)

                action_probs = F.softmax(action_logits, dim=-1)
                action_log_probs_i = F.log_softmax(action_logits, dim=-1)

                taken_action = actions_t[:, i]
                log_prob = action_log_probs_i.gather(1, taken_action.unsqueeze(1)).squeeze(1)
                ent = -(action_probs * action_log_probs_i).sum(dim=-1)

                agent_beliefs.append(new_belief)
                agent_log_probs.append(log_prob)
                agent_entropy.append(ent)

            hidden_states_list = [agent_beliefs[i] for i in range(self.n_agents)]

            all_beliefs.append(torch.stack(agent_beliefs, dim=1))
            all_action_log_probs.append(torch.stack(agent_log_probs, dim=1))
            all_entropy.append(torch.stack(agent_entropy, dim=1))

        beliefs = torch.stack(all_beliefs, dim=1)
        action_log_probs = torch.stack(all_action_log_probs, dim=1).sum(dim=-1)
        entropy = torch.stack(all_entropy, dim=1).mean(dim=-1)

        return beliefs, action_log_probs, entropy

    def update_on_episode_end(self, episode_reward: float) -> None:
        """Update stability tracking after each episode.

        Args:
            episode_reward: Total reward from the episode
        """
        # Track best performance
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1

        # Anneal auxiliary loss after warmup (only if decay is enabled)
        # aux_decay_rate < 1.0 enables decay, == 1.0 keeps constant
        if self.training_step > self.warmup_steps and self.aux_decay_rate < 1.0:
            # Exponential decay of auxiliary loss (configurable)
            # Decay towards min_aux_lambda but scaled by initial aux_lambda
            initial_aux_lambda = self.config.algorithm.aux_lambda
            min_lambda = self.min_aux_lambda * (initial_aux_lambda / 0.1) if initial_aux_lambda > 0 else 0
            self.aux_lambda_current = max(
                min_lambda,
                self.aux_lambda_current * self.aux_decay_rate
            )

        # Decay entropy coefficient over time
        min_entropy = 0.005  # Maintain exploration
        entropy_decay = 0.999
        self.current_entropy_coef = max(min_entropy, self.current_entropy_coef * entropy_decay)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a training step using PPO with auxiliary loss.

        Training objective (Eq. 11): max J(theta) - lambda * L_aux

        STABILITY IMPROVEMENTS (v2):
        1. Auxiliary loss annealing (warm start then decay)
        2. Target critic for stable value estimation
        3. KL divergence early stopping
        4. Adaptive entropy decay
        5. More conservative updates

        Args:
            batch: Dictionary containing episode data

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

        visibility_masks = batch.get("visibility_masks")
        if visibility_masks is not None:
            visibility_masks = visibility_masks.to(self.device)

        available_actions = batch.get("available_actions")
        if available_actions is not None:
            available_actions = available_actions.to(self.device)

        batch_size, seq_len = obs.shape[:2]
        state_flat = state.view(batch_size * seq_len, -1)

        # Compute old values using TARGET critic for stability
        with torch.no_grad():
            old_values = self.target_critic(state_flat).view(batch_size, seq_len)
            _, old_log_probs, _ = self._forward_pass_collect(
                obs, actions, visibility_masks, available_actions, algo_config
            )
            old_log_probs = old_log_probs.detach()
            old_values = old_values.detach()

            # Compute returns and advantages using old values
            returns, advantages = self._compute_returns_and_advantages(
                rewards, old_values, dones, mask
            )

            # Update value normalizer with new returns
            if self.use_value_norm:
                self.value_normalizer.update(returns[mask.bool()])

            # Normalize advantages (with clipping for stability)
            adv_mean = (advantages * mask).sum() / (mask.sum() + 1e-8)
            adv_std = ((advantages - adv_mean).pow(2) * mask).sum() / (mask.sum() + 1e-8)
            adv_std = torch.clamp(adv_std.sqrt(), min=1e-8)
            advantages = (advantages - adv_mean) / adv_std
            # More conservative advantage clipping
            advantages = torch.clamp(advantages, -5.0, 5.0)

        # PPO training with multiple epochs (matching MAPPO)
        n_epochs = getattr(algo_config, 'ppo_epochs', 10)

        clip_param = algo_config.clip_param
        value_clip = getattr(algo_config, 'value_clip', 0.2)

        # Use annealed auxiliary loss weight
        aux_lambda = self.aux_lambda_current

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_aux_loss = 0.0
        aux_accuracy = 0.0
        approx_kl = 0.0

        for epoch in range(n_epochs):
            # Forward pass
            beliefs, action_log_probs, entropy = self._forward_pass_collect(
                obs, actions, visibility_masks, available_actions, algo_config
            )

            # Compute approximate KL divergence for early stopping
            with torch.no_grad():
                log_ratio = action_log_probs - old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

            # Early stopping if KL divergence is too large
            target_kl = 0.015  # Conservative KL target
            if approx_kl > 1.5 * target_kl and epoch > 0:
                break

            # PPO Policy Loss with clipping
            ratio = torch.exp(action_log_probs - old_log_probs)
            # Clip ratio for numerical stability
            ratio = torch.clamp(ratio, 0.0, 5.0)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2) * mask
            policy_loss = policy_loss.sum() / (mask.sum() + 1e-8)

            # Value Loss with clipping (PPO-style) - use current critic
            value_pred = self.critic(state_flat).view(batch_size, seq_len)

            # Normalize returns for value loss when using value normalization
            if self.use_value_norm:
                returns_for_loss = self.value_normalizer.normalize(returns)
            else:
                returns_for_loss = returns

            value_pred_clipped = old_values + torch.clamp(
                value_pred - old_values, -value_clip, value_clip
            )
            value_loss1 = (value_pred - returns_for_loss).pow(2)
            value_loss2 = (value_pred_clipped - returns_for_loss).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2) * mask
            value_loss = value_loss.sum() / (mask.sum() + 1e-8)

            # Entropy Loss with annealed coefficient
            entropy_loss = -(entropy * mask).sum() / (mask.sum() + 1e-8)

            # Auxiliary Loss - compute when enabled and aux_lambda > 0
            # Only compute on first epoch to save computation
            if self.use_aux_loss and epoch == 0 and self.aux_lambda_current > 0:
                next_actions = torch.cat([actions[:, 1:], actions[:, -1:]], dim=1)
                next_vis_masks = None
                if visibility_masks is not None:
                    next_vis_masks = torch.cat([visibility_masks[:, 1:], visibility_masks[:, -1:]], dim=1)

                aux_loss_val, aux_accuracy = self._compute_auxiliary_loss(
                    beliefs, next_actions, next_vis_masks, mask
                )
            else:
                aux_loss_val = torch.tensor(0.0, device=self.device)
            
            # Force aux lambda to 0 if ablated
            current_aux_lambda = aux_lambda if self.use_aux_loss else 0.0

            # Total Loss
            total_loss = (
                policy_loss
                + algo_config.value_loss_coef * value_loss
                + self.current_entropy_coef * entropy_loss
                + current_aux_lambda * aux_loss_val
            )

            # Optimization step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping (more aggressive for stability)
            grad_clip = min(self.config.training.grad_clip, 1.0)  # Very conservative
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), grad_clip)
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)

            # Skip update if gradients are too large (indicates instability)
            if actor_grad_norm < 50 and critic_grad_norm < 50:
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            if isinstance(aux_loss_val, torch.Tensor):
                total_aux_loss += aux_loss_val.item()

        # Soft update target critic
        self._soft_update_target(tau=0.005)

        # Update learning rate schedulers (if configured)
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        self.training_step += 1

        return {
            "loss": (total_policy_loss + total_value_loss + total_entropy_loss) / max(1, epoch + 1),
            "policy_loss": total_policy_loss / max(1, epoch + 1),
            "value_loss": total_value_loss / max(1, epoch + 1),
            "entropy_loss": total_entropy_loss / max(1, epoch + 1),
            "aux_loss": total_aux_loss / max(1, epoch + 1),
            "aux_accuracy": aux_accuracy,
            "value": old_values.mean().item(),
            "approx_kl": approx_kl,
            "aux_lambda": self.aux_lambda_current,
            "entropy_coef": self.current_entropy_coef,
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
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "aux_lambda_current": self.aux_lambda_current,
                "current_entropy_coef": self.current_entropy_coef,
                "best_reward": self.best_reward,
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
        if "target_critic" in checkpoint:
            self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]
        if "aux_lambda_current" in checkpoint:
            self.aux_lambda_current = checkpoint["aux_lambda_current"]
        if "current_entropy_coef" in checkpoint:
            self.current_entropy_coef = checkpoint["current_entropy_coef"]
        if "best_reward" in checkpoint:
            self.best_reward = checkpoint["best_reward"]
        if "value_normalizer" in checkpoint:
            self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])

    def get_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return {
            "training_step": self.training_step,
            "aux_lambda": self.aux_lambda_current,
            "entropy_coef": self.current_entropy_coef,
        }
