"""Algorithm registry for dynamic algorithm loading."""

from typing import Callable, Dict, Type

from marl_research.algorithms.base import BaseAlgorithm

ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {}


def register_algorithm(name: str) -> Callable:
    """Decorator to register an algorithm.

    Usage:
        @register_algorithm("my_algorithm")
        class MyAlgorithm(BaseAlgorithm):
            ...
    """

    def decorator(cls: Type[BaseAlgorithm]) -> Type[BaseAlgorithm]:
        if name in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {name} already registered")
        ALGORITHM_REGISTRY[name] = cls
        return cls

    return decorator


def get_algorithm(name: str) -> Type[BaseAlgorithm]:
    """Get algorithm class by name."""
    if name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Algorithm {name} not found. Available: {available}")
    return ALGORITHM_REGISTRY[name]
