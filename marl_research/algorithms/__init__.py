"""Algorithm implementations for MARL."""

from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import ALGORITHM_REGISTRY, register_algorithm

# Import all algorithms to register them
from marl_research.algorithms.vabl import VABL
from marl_research.algorithms.qmix import QMIX
from marl_research.algorithms.ippo import IPPO
from marl_research.algorithms.mappo import MAPPO
from marl_research.algorithms.qplex import QPLEX
from marl_research.algorithms.commnet import CommNet
from marl_research.algorithms.maven import MAVEN
from marl_research.algorithms.tarmac import TarMAC
from marl_research.algorithms.aerial import AERIAL

__all__ = [
    "BaseAlgorithm",
    "ALGORITHM_REGISTRY",
    "register_algorithm",
    "VABL",
    "QMIX",
    "IPPO",
    "MAPPO",
    "QPLEX",
    "CommNet",
    "MAVEN",
    "TarMAC",
    "AERIAL",
]
