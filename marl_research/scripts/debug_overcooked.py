"""Debug script for Overcooked environment.

Usage:
    python -m marl_research.scripts.debug_overcooked
"""
import numpy as np
from omegaconf import OmegaConf

from marl_research.environments.overcooked_env import OvercookedEnv


def debug_overcooked():
    print("Initializing OvercookedEnv with cramped_room...")
    config = OmegaConf.create({
        'environment': {
            'layout_name': 'cramped_room',
            'horizon': 400,
            'num_agents': 2
        }
    })

    try:
        env = OvercookedEnv(config)
    except ImportError as e:
        print(f"Error: {e}")
        return

    print("Running random episodes to find non-zero reward...")

    total_episodes = 100
    nonzero_rewards = 0
    max_reward = 0.0

    for ep in range(total_episodes):
        obs, state, _ = env.reset()
        episode_reward = 0

        for t in range(400):  # Horizon
            actions = [np.random.randint(0, 6) for _ in range(2)]
            next_obs, next_state, reward, done, info = env.step(actions)

            episode_reward += reward
            if reward > 0:
                print(f"  FOUND REWARD {reward} at episode {ep} step {t}")

            if done:
                break

        if episode_reward > 0:
            nonzero_rewards += 1
            max_reward = max(max_reward, episode_reward)

        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{total_episodes}. Non-zero reward episodes: {nonzero_rewards}")

    print(f"\nFinished {total_episodes} random episodes.")
    print(f"Total episodes with reward > 0: {nonzero_rewards}")
    print(f"Max reward found: {max_reward}")


if __name__ == "__main__":
    debug_overcooked()
