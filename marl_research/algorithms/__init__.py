"""Algorithm implementations for MARL.

Note: As of 2026-04-08 the PyTorch implementations have been archived to
``pre_camera_ready_2026-04-08/marl_research_pytorch_old/algorithms/``. Active
development for the camera-ready uses the JAX implementations under
``marl_research.algorithms.jax``. See ``wiki/log.md`` for the rationale.
"""

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import ALGORITHM_REGISTRY, register_algorithm

__all__ = [
    "BaseAlgorithm",
    "ALGORITHM_REGISTRY",
    "register_algorithm",
]
