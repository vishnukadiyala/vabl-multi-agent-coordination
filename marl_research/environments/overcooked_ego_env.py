"""Ego-Centric Partially Observable Overcooked Environment.

Wraps the standard Overcooked environment with a limited view radius per agent,
following the OvercookedV2 approach (Rutherford et al., 2024). Each agent only
observes a local window around its position, creating genuine state-level
partial observability (unlike standard Overcooked where agents see the full grid).

This addresses Reviewer iBYE's concern that standard Overcooked has no
state-level partial observability that VABL could benefit from.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


@register_env("overcooked_ego")
class OvercookedEgoEnv(BaseMAEnv):
    """Overcooked with ego-centric partial observability (limited view radius).

    Each agent's observation is masked to only show cells within `view_radius`
    Manhattan distance of the agent's position. Cells outside the radius are
    zeroed out, creating genuine partial observability at the state level.
    """

    DEFAULT_SHAPED_REWARDS = {
        'pickup_onion': 3.0,
        'pickup_tomato': 3.0,
        'place_in_pot': 5.0,
        'pickup_dish': 2.0,
        'pickup_soup': 8.0,
        'deliver_soup': 20.0,
        'pot_cooking': 1.0,
    }

    def __init__(self, config: DictConfig):
        self.config = config.environment

        try:
            from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
            from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as OC_Env
        except ImportError:
            raise ImportError(
                "Overcooked-AI not installed. Install with: pip install overcooked-ai"
            )

        self.mdp = OvercookedGridworld.from_layout_name(self.config.layout_name)
        self.base_env = OC_Env.from_mdp(
            self.mdp,
            horizon=self.config.horizon,
        )

        self.n_agents = self.config.num_agents
        self.horizon = self.config.horizon
        self.view_radius = getattr(self.config, 'view_radius', 3)
        self._step_count = 0
        self._env_info = None

        # Reward shaping
        self.use_shaped_rewards = getattr(self.config, 'use_shaped_rewards', True)
        self.shaped_reward_scale = getattr(self.config, 'shaped_reward_scale', 1.0)
        self.shaped_rewards = self.DEFAULT_SHAPED_REWARDS.copy()
        if hasattr(self.config, 'shaped_rewards'):
            for key, value in self.config.shaped_rewards.items():
                if key in self.shaped_rewards:
                    self.shaped_rewards[key] = value

        self._prev_state = None

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        self.base_env.reset()
        self._step_count = 0
        self._prev_state = self._copy_state(self.base_env.state)

        obs = self._get_obs()
        state = self._get_state()
        return obs, state, {}

    def _copy_state(self, state):
        return state.deepcopy() if hasattr(state, 'deepcopy') else state

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        action_dict = {
            0: (0, -1),   # up
            1: (0, 1),    # down
            2: (-1, 0),   # left
            3: (1, 0),    # right
            4: (0, 0),    # stay
            5: "interact",
        }

        joint_action = tuple(action_dict.get(a, (0, 0)) for a in actions)
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)

        self._step_count += 1
        if self._step_count >= self.horizon:
            done = True

        if self.use_shaped_rewards:
            shaped_reward = self._compute_shaped_reward(self._prev_state, self.base_env.state)
            reward = sparse_reward + shaped_reward * self.shaped_reward_scale
            info['sparse_reward'] = sparse_reward
            info['shaped_reward'] = shaped_reward
        else:
            reward = sparse_reward

        self._prev_state = self._copy_state(self.base_env.state)

        obs = self._get_obs()
        state = self._get_state()
        return obs, state, reward, done, info

    def _get_agent_positions(self):
        """Get (row, col) positions of each agent."""
        positions = []
        for player in self.base_env.state.players:
            # player.position is (col, row) in Overcooked
            col, row = player.position
            positions.append((row, col))
        return positions

    def _apply_view_mask(self, grid: np.ndarray, agent_row: int, agent_col: int) -> np.ndarray:
        """Zero out cells outside the agent's view radius.

        Args:
            grid: Spatial observation [height, width, channels]
            agent_row: Agent's row position
            agent_col: Agent's column position

        Returns:
            Masked grid with cells outside view_radius set to zero
        """
        height, width = grid.shape[0], grid.shape[1]
        masked = np.zeros_like(grid)

        for r in range(height):
            for c in range(width):
                # Manhattan distance
                if abs(r - agent_row) + abs(c - agent_col) <= self.view_radius:
                    masked[r, c] = grid[r, c]

        return masked

    def _get_obs(self) -> List[np.ndarray]:
        """Get ego-centric observations with limited view radius."""
        state = self.base_env.state
        encoding = self.mdp.lossless_state_encoding(state)
        encoding = np.array(encoding, dtype=np.float32)

        positions = self._get_agent_positions()
        obs_list = []

        for agent_idx in range(self.n_agents):
            agent_grid = encoding[agent_idx]  # [height, width, channels]
            row, col = positions[agent_idx]

            # Apply view radius mask
            masked_grid = self._apply_view_mask(agent_grid, row, col)
            obs_list.append(masked_grid.flatten())

        return obs_list

    def _get_state(self) -> np.ndarray:
        """Get global state (full observability for centralized critic)."""
        state = self.base_env.state
        encoding = self.mdp.lossless_state_encoding(state)
        encoding = np.array(encoding, dtype=np.float32)
        return encoding.flatten()

    def get_env_info(self) -> EnvInfo:
        if self._env_info is None:
            dummy_obs = self._get_obs()
            dummy_state = self._get_state()
            self._env_info = EnvInfo(
                n_agents=self.n_agents,
                obs_shape=dummy_obs[0].shape,
                state_shape=dummy_state.shape,
                n_actions=6,
                episode_limit=self.horizon,
            )
        return self._env_info

    def get_available_actions(self) -> List[np.ndarray]:
        return [np.ones(6, dtype=np.float32) for _ in range(self.n_agents)]

    def get_visibility_masks(self) -> np.ndarray:
        """Visibility masks based on view radius.

        Agent i can see agent j's action if j is within view_radius.
        """
        positions = self._get_agent_positions()
        masks = np.zeros((self.n_agents, self.n_agents - 1), dtype=np.float32)

        for i in range(self.n_agents):
            teammate_idx = 0
            for j in range(self.n_agents):
                if j == i:
                    continue
                ri, ci = positions[i]
                rj, cj = positions[j]
                dist = abs(ri - rj) + abs(ci - cj)
                if dist <= self.view_radius:
                    masks[i, teammate_idx] = 1.0
                teammate_idx += 1

        return masks

    # --- Reward shaping (identical to standard Overcooked) ---

    def _compute_shaped_reward(self, prev_state, curr_state) -> float:
        shaped_reward = 0.0
        if prev_state is None:
            return shaped_reward

        prev_players = prev_state.players
        curr_players = curr_state.players

        for prev_player, curr_player in zip(prev_players, curr_players):
            prev_obj = prev_player.held_object
            curr_obj = curr_player.held_object

            if prev_obj is None and curr_obj is not None:
                obj_name = curr_obj.name
                if obj_name == 'onion':
                    shaped_reward += self.shaped_rewards['pickup_onion']
                elif obj_name == 'tomato':
                    shaped_reward += self.shaped_rewards['pickup_tomato']
                elif obj_name == 'dish':
                    shaped_reward += self.shaped_rewards['pickup_dish']
                elif obj_name == 'soup':
                    shaped_reward += self.shaped_rewards['pickup_soup']

            if prev_obj is not None and curr_obj is None:
                prev_name = prev_obj.name
                if prev_name in ('onion', 'tomato'):
                    if self._pot_ingredients_increased(prev_state, curr_state):
                        shaped_reward += self.shaped_rewards['place_in_pot']

        shaped_reward += self._get_cooking_reward(curr_state)
        return shaped_reward

    def _pot_ingredients_increased(self, prev_state, curr_state) -> bool:
        prev_pots = self._get_pot_states(prev_state)
        curr_pots = self._get_pot_states(curr_state)
        for pos in curr_pots:
            if pos in prev_pots:
                if curr_pots[pos].get('num_ingredients', 0) > prev_pots[pos].get('num_ingredients', 0):
                    return True
        return False

    def _get_pot_states(self, state) -> Dict:
        pot_states = {}
        try:
            for obj in state.all_objects_list:
                if hasattr(obj, 'name') and obj.name == 'soup':
                    pos = obj.position
                    pot_states[pos] = {
                        'num_ingredients': len(obj.ingredients) if hasattr(obj, 'ingredients') else 0,
                    }
        except (AttributeError, TypeError):
            pass
        return pot_states

    def _get_cooking_reward(self, state) -> float:
        reward = 0.0
        try:
            pot_states = self.mdp.get_pot_states(state)
            cooking_pots = pot_states.get('cooking', [])
            reward += len(cooking_pots) * self.shaped_rewards['pot_cooking']
        except (AttributeError, TypeError):
            pass
        return reward

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "human":
            print(self.base_env.state)
        return None

    def close(self) -> None:
        pass
