"""Environment wrappers for MARL experiments."""

from marl_research.environments.base import BaseMAEnv
from marl_research.environments.registry import ENV_REGISTRY, register_env, make_env
from marl_research.environments.simple_env import SimpleCoordinationEnv

# Optional environments - import to register them if dependencies are available
try:
    from marl_research.environments.overcooked_env import OvercookedEnv
except ImportError:
    OvercookedEnv = None  # overcooked-ai not installed

try:
    from marl_research.environments.smac_env import SMACEnv, SMACV2Env
except ImportError:
    SMACEnv = None  # pysc2/smac not installed
    SMACV2Env = None

try:
    from marl_research.environments.mpe_env import MPEEnv
except ImportError:
    MPEEnv = None  # pettingzoo not installed

try:
    from marl_research.environments.overcooked_ego_env import OvercookedEgoEnv
except ImportError:
    OvercookedEgoEnv = None  # overcooked-ai not installed

try:
    from marl_research.environments.hanabi_env import HanabiEnv
except ImportError:
    HanabiEnv = None  # hanabi_learning_environment not installed

__all__ = ["BaseMAEnv", "ENV_REGISTRY", "register_env", "make_env", "SimpleCoordinationEnv"]
