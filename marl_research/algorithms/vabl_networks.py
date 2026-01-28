"""VABL (Variational Attention-based Belief Learning) neural network components.

Based on the ICML 2026 paper: "Implicit Coordination via Attention-Driven Latent
Belief Representations in Partially Observable Environments"

Architecture (Section 5.6):
- Feature encoder phi_theta: obs -> d_x=64
- Action encoder psi_theta: 2-layer MLP, action -> d_e=64
- Attention: Scaled Dot-Product, d_k=64
- GRU: hidden dim d_h=128
- Policy head pi_theta: Linear(128, n_actions)
- Aux predictor pi_hat_phi: MLP (128 -> 64 -> n_actions) per teammate
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VABLAgent(nn.Module):
    """VABL agent network with attention-based belief learning.

    The agent maintains a belief state updated via attention over visible
    teammate actions and uses this belief to select actions.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_agents: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        aux_hidden_dim: int = 64,
        attention_heads: int = 4,
    ):
        """Initialize VABL agent.

        Args:
            obs_dim: Dimension of agent observations
            n_actions: Number of available actions
            n_agents: Total number of agents in the environment
            embed_dim: Dimension for observation/action embeddings (d_e)
            hidden_dim: Dimension for GRU hidden state / belief (d_h)
            attention_dim: Dimension for attention (d_k)
            aux_hidden_dim: Hidden dimension for auxiliary predictor
            attention_heads: Number of attention heads for MHA
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.attention_heads = attention_heads
        self.n_teammates = n_agents - 1

        self.use_attention = True

        # Feature encoder phi_theta: obs -> d_e
        self.phi_net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Action encoder psi_theta: action (one-hot) -> d_e
        self.psi_net = nn.Sequential(
            nn.Linear(n_actions, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Multi-Head Attention (Eq. 6-7 upgraded)
        # Project belief (d_h) to query (d_e)
        self.belief_proj = nn.Linear(hidden_dim, embed_dim)
        
        # MHA: d_model = embed_dim, configurable heads
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            batch_first=True
        )

        # GRU for belief update: input = [h_obs || context], hidden = belief
        # Input dim = embed_dim (obs encoding) + embed_dim (context)
        self.gru = nn.GRUCell(embed_dim + embed_dim, hidden_dim)

        # Policy head pi_theta: belief -> action logits
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Auxiliary predictor pi_hat_phi: belief -> predicted teammate actions
        # MLP (hidden_dim -> aux_hidden_dim -> n_actions * n_teammates)
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, aux_hidden_dim),
            nn.ReLU(),
            nn.Linear(aux_hidden_dim, n_actions * self.n_teammates),
        )

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize belief state to zeros.

        Args:
            batch_size: Batch size for initialization

        Returns:
            Zero-initialized belief tensor [batch_size, hidden_dim]
        """
        return torch.zeros(batch_size, self.hidden_dim, device=next(self.parameters()).device)

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation using phi network.

        Args:
            obs: Agent observations [batch, obs_dim]

        Returns:
            Encoded observation [batch, embed_dim]
        """
        return self.phi_net(obs)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions using psi network.

        Args:
            actions: One-hot encoded actions [batch, n_teammates, n_actions]

        Returns:
            Encoded actions [batch, n_teammates, embed_dim]
        """
        batch_size = actions.shape[0]
        n_teammates = actions.shape[1]

        # Reshape for batch processing
        actions_flat = actions.view(batch_size * n_teammates, self.n_actions)
        encoded_flat = self.psi_net(actions_flat)
        encoded = encoded_flat.view(batch_size, n_teammates, self.embed_dim)

        return encoded

    def attention(
        self,
        belief: torch.Tensor,
        teammate_action_embeddings: torch.Tensor,
        visibility_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute context using MHA or Mean Pooling (Ablation).

        Args:
            belief: Previous belief state [batch, hidden_dim]
            teammate_action_embeddings: Encoded teammate actions [batch, n_teammates, embed_dim]
            visibility_mask: Which teammates are visible [batch, n_teammates], 1=visible, 0=not

        Returns:
            context: Context vector [batch, embed_dim]
            weights: Attention weights (dummy for mean pooling) [batch, n_teammates]
        """
        batch_size = belief.shape[0]

        # Apply visibility mask to embeddings (zero out invisible teammates)
        if visibility_mask is not None:
            # Expand mask: [batch, n_teammates] -> [batch, n_teammates, embed_dim]
            mask_expanded = visibility_mask.unsqueeze(-1).expand_as(teammate_action_embeddings)
            masked_embeddings = teammate_action_embeddings * mask_expanded
        else:
            masked_embeddings = teammate_action_embeddings

        if not self.use_attention:
            # Ablation: Mean Pooling
            # Compute mean over visible teammates
            if visibility_mask is not None:
                # Sum over teammates
                context = masked_embeddings.sum(dim=1)  # [batch, embed_dim]
                # Count visible teammates
                counts = visibility_mask.sum(dim=1, keepdim=True) + 1e-8
                context = context / counts
            else:
                context = masked_embeddings.mean(dim=1)
            
            # Dummy weights
            weights = torch.zeros(batch_size, self.n_teammates, device=belief.device)
            return context, weights

        # Multi-Head Attention
        # Query: Projected belief [batch, 1, embed_dim]
        query = self.belief_proj(belief).unsqueeze(1)
        
        # Key/Value: Teammate embeddings [batch, n_teammates, embed_dim]
        key = value = masked_embeddings

        # Attention mask for PyTorch MHA (True = ignored/masked)
        # visibility_mask is 1 for visible, 0 for invisible.
        # PyTorch expects True for padded/ignored elements.
        key_padding_mask = None
        if visibility_mask is not None:
            key_padding_mask = (visibility_mask == 0)  # [batch, n_teammates]
            
            # Handle case where ALL keys are masked (all teammates invisible)
            # PyTorch MHA returns NaNs if all keys are masked.
            # We unmask the first element temporarily but the output will be zeroed anyway?
            # Better approach: check for fully masked rows
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                # For fully masked rows, unmask the first element to avoid NaN.
                # The result context will be Garbage, but we can mask it out later or ignore it.
                # Or better, just set key_padding_mask[all_masked, 0] = False
                # And since the value is zeroed out by masked_embeddings, the attention will attend to a zero vector.
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        attn_output, attn_weights = self.attention_layer(
            query, key, value, 
            key_padding_mask=key_padding_mask
        )
        
        # Output: [batch, 1, embed_dim] -> [batch, embed_dim]
        context = attn_output.squeeze(1)
        
        # Weights: [batch, 1, n_teammates] -> [batch, n_teammates]
        weights = attn_weights.squeeze(1)

        return context, weights

    def forward(
        self,
        obs: torch.Tensor,
        prev_belief: torch.Tensor,
        prev_teammate_actions: Optional[torch.Tensor] = None,
        visibility_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass implementing Algorithm 1 from the paper.

        For each agent i:
        1. h_obs <- phi_theta(x^{i,pub}_t)           # Encode observation
        2. H_team <- {psi_theta(a^j_{t-1}) | m^{i<-j}_t = 1}  # Encode visible actions
        3. c^i_t <- Attention(b^i_{t-1}, H_team)    # Compute context
        4. b^i_t <- GRU([h_obs || c^i_t], b^i_{t-1}) # Update belief
        5. a^i_t ~ pi_theta(. | b^i_t)              # Sample action

        Args:
            obs: Agent observations [batch, obs_dim]
            prev_belief: Previous belief state [batch, hidden_dim]
            prev_teammate_actions: Previous teammate actions (one-hot) [batch, n_teammates, n_actions]
                                   If None, uses zero context (first timestep)
            visibility_mask: Which teammates are visible [batch, n_teammates]
                            If None, assumes all visible

        Returns:
            action_logits: Action logits [batch, n_actions]
            belief: Updated belief state [batch, hidden_dim]
            aux_logits: Auxiliary prediction logits [batch, n_teammates, n_actions]
            attention_weights: Attention weights [batch, n_teammates]
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Step 1: Encode observation
        h_obs = self.encode_observation(obs)  # [batch, embed_dim]

        # Steps 2-3: Encode teammate actions and compute attention context
        if prev_teammate_actions is not None:
            # Encode teammate actions
            H_team = self.encode_actions(prev_teammate_actions)  # [batch, n_teammates, embed_dim]

            # Compute attention context
            context, attention_weights = self.attention(
                prev_belief, H_team, visibility_mask
            )  # context: [batch, attention_dim]
        else:
            # First timestep: no previous actions, use zero context
            context = torch.zeros(batch_size, self.attention_dim, device=device)
            attention_weights = torch.zeros(batch_size, self.n_teammates, device=device)

        # Step 4: Update belief via GRU
        gru_input = torch.cat([h_obs, context], dim=-1)  # [batch, embed_dim + attention_dim]
        belief = self.gru(gru_input, prev_belief)  # [batch, hidden_dim]

        # Step 5: Compute action logits
        action_logits = self.policy_head(belief)  # [batch, n_actions]

        # Auxiliary prediction: predict next teammate actions from current belief
        aux_logits = self.aux_head(belief)  # [batch, n_actions * n_teammates]
        aux_logits = aux_logits.view(batch_size, self.n_teammates, self.n_actions)

        return action_logits, belief, aux_logits, attention_weights


class CentralizedCritic(nn.Module):
    """Centralized critic for VABL using global state.

    Used during training for computing value estimates with full state information.
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
    ):
        """Initialize centralized critic.

        Args:
            state_dim: Dimension of global state
            n_agents: Number of agents
            hidden_dim: Hidden dimension for critic network
        """
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from global state.

        Args:
            state: Global state [batch, state_dim]

        Returns:
            Value estimate [batch, 1]
        """
        return self.network(state)
