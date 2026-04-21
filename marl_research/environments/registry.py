"""Environment registry for dynamic environment loading."""

from typing import Callable, Dict, Type

from omegaconf import DictConfig

from marl_research.environments.base import BaseMAEnv

ENV_REGISTRY: Dict[str, Type[BaseMAEnv]] = {}


def register_env(name: str) -> Callable:
    """Decorator to register an environment.

    Usage:
        @register_env("my_env")
        class MyEnv(BaseMAEnv):
            ...
    """

    def decorator(cls: Type[BaseMAEnv]) -> Type[BaseMAEnv]:
        if name in ENV_REGISTRY:
            raise ValueError(f"Environment {name} already registered")
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


def make_env(config: DictConfig) -> BaseMAEnv:
    """Create an environment from config."""
    env_name = config.environment.name
    if env_name not in ENV_REGISTRY:
        available = list(ENV_REGISTRY.keys())
        raise ValueError(f"Environment {env_name} not found. Available: {available}")
    return ENV_REGISTRY[env_name](config)
