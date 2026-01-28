"""Overcooked Environment Wrapper with Reward Shaping."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv, EnvInfo
from marl_research.environments.registry import register_env


@register_env("overcooked")
class OvercookedEnv(BaseMAEnv):
    """Wrapper for Overcooked-AI environment with optional reward shaping."""

    # Reward shaping values (can be overridden via config)
    DEFAULT_SHAPED_REWARDS = {
        'pickup_onion': 3.0,        # Agent picks up onion from dispenser
        'pickup_tomato': 3.0,       # Agent picks up tomato from dispenser
        'place_in_pot': 5.0,        # Agent places ingredient in pot
        'pickup_dish': 2.0,         # Agent picks up a dish
        'pickup_soup': 8.0,         # Agent picks up cooked soup
        'deliver_soup': 20.0,       # Delivered soup (sparse reward from env)
        'pot_cooking': 1.0,         # Per-step reward while pot is cooking
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
        self._step_count = 0
        self._env_info = None

        # Reward shaping configuration
        self.use_shaped_rewards = getattr(self.config, 'use_shaped_rewards', True)
        self.shaped_reward_scale = getattr(self.config, 'shaped_reward_scale', 1.0)

        # Allow custom shaped reward values via config
        self.shaped_rewards = self.DEFAULT_SHAPED_REWARDS.copy()
        if hasattr(self.config, 'shaped_rewards'):
            for key, value in self.config.shaped_rewards.items():
                if key in self.shaped_rewards:
                    self.shaped_rewards[key] = value

        # State tracking for reward shaping
        self._prev_state = None

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
        self.base_env.reset()
        self._step_count = 0
        self._prev_state = self._copy_state(self.base_env.state)

        obs = self._get_obs()
        state = self._get_state()
        return obs, state, {}

    def _copy_state(self, state):
        """Create a copy of the state for comparison."""
        return state.deepcopy() if hasattr(state, 'deepcopy') else state

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray, float, bool, Dict[str, Any]]:
        action_dict = {
            0: (0, -1),  # up
            1: (0, 1),   # down
            2: (-1, 0),  # left
            3: (1, 0),   # right
            4: (0, 0),   # stay
            5: "interact",
        }

        joint_action = tuple(action_dict.get(a, (0, 0)) for a in actions)

        next_state, sparse_reward, done, info = self.base_env.step(joint_action)

        self._step_count += 1

        if self._step_count >= self.horizon:
            done = True

        # Compute shaped reward
        if self.use_shaped_rewards:
            shaped_reward = self._compute_shaped_reward(self._prev_state, self.base_env.state)
            reward = sparse_reward + shaped_reward * self.shaped_reward_scale
            info['sparse_reward'] = sparse_reward
            info['shaped_reward'] = shaped_reward
        else:
            reward = sparse_reward

        # Update state tracking
        self._prev_state = self._copy_state(self.base_env.state)

        obs = self._get_obs()
        state = self._get_state()

        return obs, state, reward, done, info

    def _compute_shaped_reward(self, prev_state, curr_state) -> float:
        """Compute shaped reward based on state changes.

        Rewards progress toward cooking and delivering soup:
        - Picking up ingredients (onion/tomato)
        - Placing ingredients in pot
        - Soup cooking in pot
        - Picking up dishes
        - Picking up cooked soup
        """
        shaped_reward = 0.0

        if prev_state is None:
            return shaped_reward

        # Get player states
        prev_players = prev_state.players
        curr_players = curr_state.players

        # Check each agent for changes in held objects
        for i, (prev_player, curr_player) in enumerate(zip(prev_players, curr_players)):
            prev_obj = prev_player.held_object
            curr_obj = curr_player.held_object

            # Agent picked up something (was holding nothing, now holding something)
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

            # Agent put down ingredient (was holding onion/tomato, now holding nothing)
            # This likely means they placed it in a pot
            if prev_obj is not None and curr_obj is None:
                prev_name = prev_obj.name
                if prev_name in ('onion', 'tomato'):
                    # Check if pot contents increased
                    if self._pot_ingredients_increased(prev_state, curr_state):
                        shaped_reward += self.shaped_rewards['place_in_pot']

        # Reward for pots that are cooking
        shaped_reward += self._get_cooking_reward(curr_state)

        return shaped_reward

    def _pot_ingredients_increased(self, prev_state, curr_state) -> bool:
        """Check if any pot has more ingredients than before."""
        prev_pots = self._get_pot_states(prev_state)
        curr_pots = self._get_pot_states(curr_state)

        for pos in curr_pots:
            if pos in prev_pots:
                prev_count = prev_pots[pos].get('num_ingredients', 0)
                curr_count = curr_pots[pos].get('num_ingredients', 0)
                if curr_count > prev_count:
                    return True
        return False

    def _get_pot_states(self, state) -> Dict:
        """Extract pot states from game state."""
        pot_states = {}
        try:
            # Get all pot locations and their contents
            for obj in state.all_objects_list:
                if hasattr(obj, 'name') and obj.name == 'soup':
                    pos = obj.position
                    pot_states[pos] = {
                        'num_ingredients': len(obj.ingredients) if hasattr(obj, 'ingredients') else 0,
                        'is_cooking': obj.is_cooking if hasattr(obj, 'is_cooking') else False,
                        'is_ready': obj.is_ready if hasattr(obj, 'is_ready') else False,
                    }
        except (AttributeError, TypeError):
            pass
        return pot_states

    def _get_cooking_reward(self, state) -> float:
        """Give small reward for each pot that is actively cooking."""
        reward = 0.0
        try:
            pot_states = self.mdp.get_pot_states(state)
            # pot_states is dict with keys like 'onion', 'tomato', 'cooking', 'ready', 'empty'
            cooking_pots = pot_states.get('cooking', [])
            reward += len(cooking_pots) * self.shaped_rewards['pot_cooking']
        except (AttributeError, TypeError):
            pass
        return reward

    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for each agent.

        Uses lossless_state_encoding which returns shape (n_agents, height, width, channels).
        We flatten each agent's observation.
        """
        state = self.base_env.state

        # Get lossless encoding: shape (n_agents, height, width, channels)
        encoding = self.mdp.lossless_state_encoding(state)
        encoding = np.array(encoding, dtype=np.float32)

        obs_list = []
        for agent_idx in range(self.n_agents):
            # Flatten the spatial encoding for this agent
            obs = encoding[agent_idx].flatten()
            obs_list.append(obs)

        return obs_list

    def _get_state(self) -> np.ndarray:
        """Get global state.

        Concatenates all agents' observations into a single state vector.
        """
        state = self.base_env.state

        # Get lossless encoding and flatten all agents' views
        encoding = self.mdp.lossless_state_encoding(state)
        encoding = np.array(encoding, dtype=np.float32)

        # Flatten everything into global state
        state_vector = encoding.flatten()

        return state_vector

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
        """All actions are available in Overcooked."""
        return [np.ones(6, dtype=np.float32) for _ in range(self.n_agents)]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "human":
            print(self.base_env.state)
        return None

    def close(self) -> None:
        pass
