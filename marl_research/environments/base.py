"""Base environment class for multi-agent environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EnvInfo:
    """Information about the environment."""

    n_agents: int
    obs_shape: Tuple[int, ...]
    state_shape: Tuple[int, ...]
    n_actions: int
    episode_limit: int
    action_spaces: Optional[List[Any]] = None


class BaseMAEnv(ABC):
    """Abstract base class for multi-agent environments."""

    @abstractmethod
    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Returns:
            observations: List of observations for each agent
            state: Global state
            info: Additional information
        """
        pass

    @abstractmethod
    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            actions: List of actions for each agent

        Returns:
            observations: List of observations for each agent
            state: Global state
            reward: Team reward
            done: Whether episode is finished
            info: Additional information
        """
        pass

    @abstractmethod
    def get_env_info(self) -> EnvInfo:
        """Get environment information."""
        pass

    @abstractmethod
    def get_available_actions(self) -> List[np.ndarray]:
        """Get available actions for each agent."""
        pass

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        pass

    def close(self) -> None:
        """Close the environment."""
        pass

    def seed(self, seed: int) -> None:
        """Set random seed."""
        pass
