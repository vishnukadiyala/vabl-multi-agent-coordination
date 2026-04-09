"""Hanabi Environment Wrapper for MARL Belief Learning Research.

Hanabi is a cooperative card game with strong partial observability: each agent
can see other players' cards but not their own. This makes it the gold standard
benchmark for testing belief learning in multi-agent settings.

The game is turn-based (one player acts at a time), but this wrapper presents
it with a simultaneous-action interface consistent with the MARL framework.
Non-active agents receive a no-op action mask.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


@register_env("hanabi")
class HanabiEnv(BaseMAEnv):
    """Wrapper for Hanabi Learning Environment (Google DeepMind).

    Hanabi is a cooperative partially observable card game where players can see
    other players' hands but not their own. Players must communicate through
    limited hint actions to coordinate card plays and maximize score.

    Key features for belief learning research:
    - Strong partial observability (cannot see own cards)
    - Communication through hint actions (information tokens)
    - Cooperative objective (maximize team score, max 25)
    - Turn-based game presented as simultaneous-action environment

    Turn-based to simultaneous-action adaptation:
    - Each step, only the current player has legal actions available
    - Non-active players have only a single no-op action available
    - The no-op action (index 0) is reserved for non-active players
    - Actions from non-active players are ignored
    """

    def __init__(self, config: DictConfig):
        self.config = config.environment

        try:
            from hanabi_learning_environment import rl_env
            self._rl_env_module = rl_env
        except ImportError:
            raise ImportError(
                "hanabi_learning_environment not installed. "
                "Install with: pip install hanabi_learning_environment"
            )

        self.n_agents = getattr(self.config, "num_players", 2)
        self.colors = getattr(self.config, "colors", 5)
        self.rank = getattr(self.config, "rank", 5)
        self.hand_size = getattr(
            self.config, "hand_size", 5 if self.n_agents <= 3 else 4
        )
        self.max_information_tokens = getattr(
            self.config, "max_information_tokens", 8
        )
        self.max_life_tokens = getattr(self.config, "max_life_tokens", 3)
        self.observation_type = getattr(self.config, "observation_type", 1)
        self.episode_limit = getattr(self.config, "episode_limit", 80)

        # Maximum possible score in Hanabi (colors * rank)
        self.max_score = self.colors * self.rank

        # Build the underlying Hanabi environment
        hanabi_config = {
            "players": self.n_agents,
            "colors": self.colors,
            "ranks": self.rank,
            "hand_size": self.hand_size,
            "max_information_tokens": self.max_information_tokens,
            "max_life_tokens": self.max_life_tokens,
            "observation_type": self.observation_type,
        }
        self.env = self._rl_env_module.HanabiEnv(config=hanabi_config)

        # Determine observation and action sizes from the environment
        self.obs_dim = self.env.vectorized_observation_shape()[0]
        self.max_moves = self.env.num_moves()

        # Action space: max_moves + 1 (index 0 is no-op for non-active players)
        self.n_actions = self.max_moves + 1

        # Internal state
        self._step_count = 0
        self._current_player = 0
        self._hanabi_obs = None  # Raw observation dict from Hanabi env
        self._done = False
        self._env_info = None
        self._score = 0

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observations."""
        self._step_count = 0
        self._done = False
        self._score = 0

        self._hanabi_obs = self.env.reset()
        self._current_player = self._hanabi_obs["current_player"]

        obs = self._get_obs()
        state = self._get_state()
        return obs, state, {"current_player": self._current_player}

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Only the action of the current player is used. Actions from other
        agents are ignored (they should be masked to no-op).

        Args:
            actions: List of action indices, one per agent. Only the current
                     player's action is executed. Actions are offset by 1
                     (index 0 = no-op), so the actual Hanabi action is
                     actions[current_player] - 1.

        Returns:
            observations, state, reward, done, info
        """
        self._step_count += 1

        if self._done:
            # Episode already finished; return terminal state
            obs = self._get_obs()
            state = self._get_state()
            return obs, state, 0.0, True, {
                "score": self._score,
                "max_score": self.max_score,
                "sparse_reward": 0.0,
                "current_player": self._current_player,
            }

        # Get the active player's action (offset by 1 for no-op reservation)
        active_action = int(actions[self._current_player])

        # Convert from our action space (0=no-op, 1..max_moves=hanabi actions)
        # to Hanabi action space (0..max_moves-1)
        if active_action <= 0:
            # No-op or invalid: pick the first legal action as fallback
            legal_moves = self._get_legal_moves_for_current_player()
            hanabi_action = legal_moves[0] if len(legal_moves) > 0 else 0
        else:
            hanabi_action = active_action - 1
            # Validate that the action is legal
            legal_moves = self._get_legal_moves_for_current_player()
            if hanabi_action not in legal_moves:
                # Fallback to first legal action if chosen action is illegal
                hanabi_action = legal_moves[0] if len(legal_moves) > 0 else 0

        # Execute the action in the Hanabi environment
        prev_score = self._get_current_score()
        self._hanabi_obs, raw_reward, self._done, _ = self.env.step(hanabi_action)
        # Use only positive rewards (successful card plays).
        # The raw env returns -score at game end, which cancels all gains
        # and produces zero total episode reward — breaking the training signal.
        reward = max(raw_reward, 0)

        if not self._done:
            self._current_player = self._hanabi_obs["current_player"]

        # Compute score change as reward
        current_score = self._get_current_score()
        self._score = current_score

        # Check episode limit
        if self._step_count >= self.episode_limit:
            self._done = True

        obs = self._get_obs()
        state = self._get_state()

        info = {
            "score": self._score,
            "max_score": self.max_score,
            "score_ratio": self._score / self.max_score,
            "sparse_reward": reward,
            "current_player": self._current_player,
            "battle_won": self._score == self.max_score,
        }

        return obs, state, reward, self._done, info

    def _get_legal_moves_for_current_player(self) -> List[int]:
        """Get legal moves for the current player from the raw observation."""
        if self._hanabi_obs is None:
            return [0]
        player_obs = self._hanabi_obs["player_observations"][self._current_player]
        return player_obs["legal_moves_as_int"]

    def _get_current_score(self) -> int:
        """Get the current game score from the observation."""
        if self._hanabi_obs is None:
            return 0
        # The score is tracked in the fireworks (played cards on the table)
        # We extract it from any player's observation since score is public
        try:
            player_obs = self._hanabi_obs["player_observations"][0]
            fireworks = player_obs.get("fireworks", {})
            return sum(fireworks.values())
        except (KeyError, IndexError, TypeError):
            return 0

    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for each agent.

        Each agent gets their own vectorized observation from the Hanabi env.
        The observation includes card knowledge, fireworks, discard pile,
        information tokens, life tokens, and other agents' observed hands,
        but crucially NOT the agent's own hand.

        Returns:
            List of observation vectors, one per agent.
        """
        obs_list = []

        if self._hanabi_obs is None or self._done:
            # Return zero observations for terminal or uninitialized state
            for _ in range(self.n_agents):
                obs_list.append(np.zeros(self.obs_dim, dtype=np.float32))
            return obs_list

        for agent_idx in range(self.n_agents):
            player_obs = self._hanabi_obs["player_observations"][agent_idx]
            vectorized = np.array(
                player_obs["vectorized"], dtype=np.float32
            )

            # Pad or truncate to consistent obs_dim
            if len(vectorized) < self.obs_dim:
                padded = np.zeros(self.obs_dim, dtype=np.float32)
                padded[: len(vectorized)] = vectorized
                obs_list.append(padded)
            elif len(vectorized) > self.obs_dim:
                obs_list.append(vectorized[: self.obs_dim])
            else:
                obs_list.append(vectorized)

        return obs_list

    def _get_state(self) -> np.ndarray:
        """Get global state by concatenating all players' observations.

        The global state includes information that no single agent has:
        by concatenating all observations, we effectively get each agent's
        view of others' hands, which collectively reveals all cards.

        Returns:
            Concatenated observation vector of shape (n_agents * obs_dim,).
        """
        obs_list = self._get_obs()
        state = np.concatenate(obs_list, axis=0)
        return state

    def get_env_info(self) -> EnvInfo:
        """Get environment information."""
        if self._env_info is None:
            self._env_info = EnvInfo(
                n_agents=self.n_agents,
                obs_shape=(self.obs_dim,),
                state_shape=(self.obs_dim * self.n_agents,),
                n_actions=self.n_actions,
                episode_limit=self.episode_limit,
            )
        return self._env_info

    def get_available_actions(self) -> List[np.ndarray]:
        """Get available actions for each agent.

        Only the current player has real actions available. All other agents
        can only take the no-op action (index 0).

        Action mapping:
        - Index 0: No-op (only action available for non-active agents)
        - Index 1 to max_moves: Hanabi actions (play, discard, hint)

        Returns:
            List of binary action masks, one per agent.
        """
        available = []

        for agent_idx in range(self.n_agents):
            mask = np.zeros(self.n_actions, dtype=np.float32)

            if self._done:
                # Terminal state: only no-op available
                mask[0] = 1.0
            elif agent_idx == self._current_player:
                # Active player: legal Hanabi moves (offset by 1)
                legal_moves = self._get_legal_moves_for_current_player()
                for move in legal_moves:
                    # Offset by 1 because index 0 is reserved for no-op
                    if move + 1 < self.n_actions:
                        mask[move + 1] = 1.0
            else:
                # Non-active player: only no-op
                mask[0] = 1.0

            available.append(mask)

        return available

    def get_visibility_masks(self) -> np.ndarray:
        """Get visibility masks for belief learning.

        In Hanabi, every agent can see every other agent's hand but not their
        own. This is a fixed visibility pattern (always fully visible except
        for self).

        Returns:
            Binary mask of shape (n_agents, n_agents - 1) where 1.0 means
            the agent can observe the other agent.
        """
        # In Hanabi, you can always see other players' cards
        masks = np.ones((self.n_agents, self.n_agents - 1), dtype=np.float32)
        return masks

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the current game state."""
        if mode == "human" and self._hanabi_obs is not None:
            try:
                player_obs = self._hanabi_obs["player_observations"][
                    self._current_player
                ]
                print(f"Step: {self._step_count}")
                print(f"Current player: {self._current_player}")
                print(f"Score: {self._score}/{self.max_score}")
                fireworks = player_obs.get("fireworks", {})
                print(f"Fireworks: {fireworks}")
                info_tokens = player_obs.get("information_tokens", "?")
                life_tokens = player_obs.get("life_tokens", "?")
                print(f"Info tokens: {info_tokens}, Life tokens: {life_tokens}")
                legal = player_obs.get("legal_moves", [])
                print(f"Legal moves: {len(legal)}")
            except (KeyError, IndexError):
                print(f"Step: {self._step_count}, Score: {self._score}")
        return None

    def close(self) -> None:
        """Close the environment."""
        pass

    def seed(self, seed: int) -> None:
        """Set random seed.

        Note: The Hanabi Learning Environment does not support explicit
        seeding through its Python API. Randomness is managed internally.
        We set numpy's seed for any wrapper-level randomness.
        """
        np.random.seed(seed)
