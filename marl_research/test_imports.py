"""Test script to verify all imports work correctly."""

import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    # Test basic package import
    import marl_research
    print(f"Package version: {marl_research.__version__}")

    # Test algorithm imports
    from marl_research.algorithms.base import BaseAlgorithm
    from marl_research.algorithms.registry import ALGORITHM_REGISTRY, register_algorithm
    from marl_research.algorithms.qmix import QMIX
    print(f"Algorithm registry: {list(ALGORITHM_REGISTRY.keys())}")

    # Test environment imports
    from marl_research.environments.base import BaseMAEnv, EnvInfo
    from marl_research.environments.registry import ENV_REGISTRY, make_env
    from marl_research.environments.smac_env import SMACEnv, SMACV2Env
    from marl_research.environments.overcooked_env import OvercookedEnv
    print(f"Environment registry: {list(ENV_REGISTRY.keys())}")

    # Test utils imports
    from marl_research.utils.misc import set_seed, get_device
    from marl_research.utils.replay_buffer import ReplayBuffer, EpisodeBuffer
    from marl_research.utils.logger import MetricLogger
    from marl_research.utils.visualization import plot_learning_curves
    print("Utils imported successfully")

    # Test network imports
    from marl_research.algorithms.networks import RNNAgent, QMixer, MLPNetwork
    print("Networks imported successfully")

    print("\nAll imports successful!")


def test_components():
    """Test that core components work correctly."""
    print("\nTesting components...")

    from marl_research.algorithms.networks import RNNAgent, QMixer
    from marl_research.utils.replay_buffer import ReplayBuffer
    from marl_research.utils.misc import set_seed, get_device

    # Test seed setting
    set_seed(42)
    print("Seed set successfully")

    # Test device detection
    device = get_device("auto")
    print(f"Detected device: {device}")

    # Test RNN Agent
    agent = RNNAgent(input_dim=10, hidden_dim=64, rnn_hidden_dim=64, n_actions=5)
    obs = torch.randn(4, 10)
    hidden = agent.init_hidden().expand(4, -1)
    q_values, new_hidden = agent(obs, hidden)
    print(f"RNN Agent output shape: {q_values.shape}")

    # Test QMixer
    mixer = QMixer(n_agents=3, state_dim=30, embed_dim=32, hypernet_hidden_dim=64)
    agent_qs = torch.randn(4, 1, 3)
    state = torch.randn(4, 1, 30)
    q_total = mixer(agent_qs, state)
    print(f"QMixer output shape: {q_total.shape}")

    # Test replay buffer
    buffer = ReplayBuffer(
        buffer_size=100,
        episode_limit=50,
        n_agents=3,
        obs_shape=(10,),
        state_shape=(30,),
        n_actions=5,
    )
    print(f"Replay buffer created")

    print("\nAll component tests passed!")


if __name__ == "__main__":
    test_imports()
    test_components()
