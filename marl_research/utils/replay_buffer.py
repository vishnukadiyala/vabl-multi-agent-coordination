"""Replay buffer implementations for MARL."""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class EpisodeBuffer:
    """Buffer for storing episode data."""

    def __init__(self, episode_limit: int, n_agents: int, obs_shape: tuple, state_shape: tuple, n_actions: int):
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.reset()

    def reset(self) -> None:
        """Reset the episode buffer."""
        self.obs = []
        self.state = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.next_state = []
        self.dones = []
        self.available_actions = []
        self.next_available_actions = []
        self.visibility_masks = []
        self._t = 0

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        available_actions: Optional[np.ndarray] = None,
        next_available_actions: Optional[np.ndarray] = None,
        visibility_masks: Optional[np.ndarray] = None,
    ) -> None:
        """Add a transition to the episode buffer.

        Args:
            obs: Agent observations
            state: Global state
            actions: Actions taken
            reward: Reward received
            next_obs: Next observations
            next_state: Next global state
            done: Whether episode is done
            available_actions: Available actions mask
            next_available_actions: Next available actions mask
            visibility_masks: Visibility masks [n_agents, n_agents-1] indicating which
                             agents can see which teammates
        """
        self.obs.append(obs)
        self.state.append(state)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.next_obs.append(next_obs)
        self.next_state.append(next_state)
        self.dones.append(done)

        if available_actions is not None:
            self.available_actions.append(available_actions)
        if next_available_actions is not None:
            self.next_available_actions.append(next_available_actions)
        if visibility_masks is not None:
            self.visibility_masks.append(visibility_masks)

        self._t += 1

    def get_episode(self) -> Dict[str, np.ndarray]:
        """Get the complete episode data."""
        episode = {
            "obs": np.array(self.obs),
            "state": np.array(self.state),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "next_obs": np.array(self.next_obs),
            "next_state": np.array(self.next_state),
            "dones": np.array(self.dones),
        }

        if self.available_actions:
            episode["available_actions"] = np.array(self.available_actions)
        if self.next_available_actions:
            episode["next_available_actions"] = np.array(self.next_available_actions)
        if self.visibility_masks:
            episode["visibility_masks"] = np.array(self.visibility_masks)

        return episode

    def __len__(self) -> int:
        return self._t


class ReplayBuffer:
    """Experience replay buffer for MARL algorithms."""

    def __init__(
        self,
        buffer_size: int,
        episode_limit: int,
        n_agents: int,
        obs_shape: tuple,
        state_shape: tuple,
        n_actions: int,
    ):
        self.buffer_size = buffer_size
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.buffer: deque = deque(maxlen=buffer_size)
        self._current_episode = EpisodeBuffer(
            episode_limit, n_agents, obs_shape, state_shape, n_actions
        )

    def add_transition(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        available_actions: Optional[np.ndarray] = None,
        next_available_actions: Optional[np.ndarray] = None,
        visibility_masks: Optional[np.ndarray] = None,
    ) -> None:
        """Add a transition to the current episode.

        Args:
            obs: Agent observations
            state: Global state
            actions: Actions taken
            reward: Reward received
            next_obs: Next observations
            next_state: Next global state
            done: Whether episode is done
            available_actions: Available actions mask
            next_available_actions: Next available actions mask
            visibility_masks: Visibility masks [n_agents, n_agents-1] indicating which
                             agents can see which teammates
        """
        self._current_episode.add(
            obs,
            state,
            actions,
            reward,
            next_obs,
            next_state,
            done,
            available_actions,
            next_available_actions,
            visibility_masks,
        )

        if done:
            self._store_episode()

    def _store_episode(self) -> None:
        """Store the current episode and reset."""
        if len(self._current_episode) > 0:
            episode = self._current_episode.get_episode()
            self.buffer.append(episode)
        self._current_episode.reset()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of episodes."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        episodes = [self.buffer[i] for i in indices]

        max_len = max(len(ep["rewards"]) for ep in episodes)

        batch = {}
        for key in episodes[0].keys():
            padded = []
            for ep in episodes:
                ep_len = len(ep["rewards"])
                if ep_len < max_len:
                    pad_shape = (max_len - ep_len,) + ep[key].shape[1:]
                    padding = np.zeros(pad_shape, dtype=ep[key].dtype)
                    padded.append(np.concatenate([ep[key], padding], axis=0))
                else:
                    padded.append(ep[key])
            batch[key] = torch.FloatTensor(np.stack(padded))

        mask = torch.zeros(batch_size, max_len)
        for i, ep in enumerate(episodes):
            mask[i, : len(ep["rewards"])] = 1.0
        batch["mask"] = mask

        return batch

    def can_sample(self, batch_size: int) -> bool:
        """Check if we can sample a batch."""
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer."""

    def __init__(
        self,
        buffer_size: int,
        episode_limit: int,
        n_agents: int,
        obs_shape: tuple,
        state_shape: tuple,
        n_actions: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        super().__init__(
            buffer_size, episode_limit, n_agents, obs_shape, state_shape, n_actions
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities: deque = deque(maxlen=buffer_size)
        self.max_priority = 1.0

    def _store_episode(self) -> None:
        """Store episode with max priority."""
        if len(self._current_episode) > 0:
            episode = self._current_episode.get_episode()
            self.buffer.append(episode)
            self.priorities.append(self.max_priority)
        self._current_episode.reset()

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample with prioritization."""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        episodes = [self.buffer[i] for i in indices]
        max_len = max(len(ep["rewards"]) for ep in episodes)

        batch = {}
        for key in episodes[0].keys():
            padded = []
            for ep in episodes:
                ep_len = len(ep["rewards"])
                if ep_len < max_len:
                    pad_shape = (max_len - ep_len,) + ep[key].shape[1:]
                    padding = np.zeros(pad_shape, dtype=ep[key].dtype)
                    padded.append(np.concatenate([ep[key], padding], axis=0))
                else:
                    padded.append(ep[key])
            batch[key] = torch.FloatTensor(np.stack(padded))

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6)
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
