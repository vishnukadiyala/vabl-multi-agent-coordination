"""Test script to verify icml2026 environment setup.

Usage:
    python -m marl_research.scripts.test_setup
"""
import warnings
warnings.filterwarnings('ignore')

print('='*60)
print('Verifying icml2026 environment setup')
print('='*60)

# Test imports
print('\n1. Testing imports...')
import torch
import numpy as np
print(f'   PyTorch: {torch.__version__}')
print(f'   NumPy: {np.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA device: {torch.cuda.get_device_name(0)}')

# Test VABL
print('\n2. Testing VABL algorithm...')
from marl_research.algorithms.vabl import VABL
from marl_research.algorithms.vabl_networks import VABLAgent
print('   VABL imports: OK')

# Test Overcooked
print('\n3. Testing Overcooked environment...')
from marl_research.environments.overcooked_env import OvercookedEnv
from omegaconf import OmegaConf
config = OmegaConf.create({
    'environment': {
        'name': 'overcooked',
        'layout_name': 'cramped_room',
        'horizon': 100,
        'num_agents': 2,
    }
})
env = OvercookedEnv(config)
env_info = env.get_env_info()
print(f'   Overcooked: OK (obs_shape={env_info.obs_shape})')

# Test simple env
print('\n4. Testing Simple environment...')
from marl_research.environments.simple_env import SimpleCoordinationEnv
config2 = OmegaConf.create({
    'environment': {
        'name': 'simple',
        'n_agents': 3,
        'obs_dim': 16,
        'n_actions': 5,
        'episode_limit': 50,
        'visibility_prob': 0.7,
    }
})
env2 = SimpleCoordinationEnv(config2)
print('   Simple env: OK')

# Test replay buffer
print('\n5. Testing ReplayBuffer...')
from marl_research.utils.replay_buffer import ReplayBuffer
buffer = ReplayBuffer(
    buffer_size=100,
    episode_limit=50,
    n_agents=3,
    obs_shape=(16,),
    state_shape=(48,),
    n_actions=5,
)
print('   ReplayBuffer: OK')

# Test device selection
print('\n6. Testing device selection...')
from marl_research.utils.misc import get_device
device = get_device('auto')
print(f'   Auto device: {device}')

print('\n' + '='*60)
print('Environment setup: SUCCESS')
print('='*60)
print('\nReady to run experiments with:')
print('  conda activate icml2026')
print('  python -m marl_research.scripts.run_vabl_experiments --device auto')
print('  python -m marl_research.runners.train algorithm=vabl environment=overcooked')
