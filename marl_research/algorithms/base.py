"""Base algorithm class for MARL algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseAlgorithm(ABC):
    """Abstract base class for all MARL algorithms."""

    def __init__(
        self,
        config: DictConfig,
        n_agents: int,
        obs_shape: tuple,
        state_shape: tuple,
        n_actions: int,
        device: torch.device,
    ):
        self.config = config
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device

        self._build_networks()
        self._build_optimizers()

        self.training_step = 0

    @abstractmethod
    def _build_networks(self) -> None:
        """Build neural network components."""
        pass

    @abstractmethod
    def _build_optimizers(self) -> None:
        """Build optimizers for training."""
        pass

    @abstractmethod
    def select_actions(
        self,
        observations: torch.Tensor,
        available_actions: Optional[torch.Tensor] = None,
        explore: bool = True,
    ) -> torch.Tensor:
        """Select actions for all agents.

        Args:
            observations: Agent observations [batch, n_agents, obs_dim]
            available_actions: Mask of available actions [batch, n_agents, n_actions]
            explore: Whether to use exploration

        Returns:
            Selected actions [batch, n_agents]
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing training batch data

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        pass

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        """Get current hidden states for RNN-based agents."""
        return None

    def init_hidden(self, batch_size: int) -> None:
        """Initialize hidden states for RNN-based agents."""
        pass

    def update_targets(self) -> None:
        """Update target networks if applicable."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {}
