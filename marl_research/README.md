# MARL Research Template

A modular Python template for Multi-Agent Reinforcement Learning (MARL) research experiments with support for SMAC, SMAC V2, and Overcooked environments.

## Project Structure

```
marl_research/
├── algorithms/              # Algorithm implementations
│   ├── base.py             # Base algorithm class
│   ├── networks.py         # Neural network components
│   ├── qmix.py             # QMIX implementation
│   └── registry.py         # Algorithm registry
├── configs/                 # Configuration files
│   ├── algorithm/          # Algorithm-specific configs
│   ├── environment/        # Environment-specific configs
│   └── default.yaml        # Default configuration
├── environments/            # Environment wrappers
│   ├── base.py             # Base environment class
│   ├── smac_env.py         # SMAC and SMAC V2 wrappers
│   ├── overcooked_env.py   # Overcooked wrapper
│   └── registry.py         # Environment registry
├── experiments/             # Experiment configurations
│   ├── smac/               # SMAC experiments
│   ├── smac_v2/            # SMAC V2 experiments
│   └── overcooked/         # Overcooked experiments
├── runners/                 # Training and evaluation
│   ├── train.py            # Main training script
│   └── evaluate.py         # Evaluation script
├── scripts/                 # Shell scripts for running experiments
├── utils/                   # Utility modules
│   ├── logger.py           # Logging utilities
│   ├── replay_buffer.py    # Replay buffer implementations
│   ├── visualization.py    # Plotting utilities
│   └── misc.py             # Miscellaneous utilities
└── results/                 # Experiment results (gitignored)
```

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Environment Dependencies

```bash
# For SMAC
pip install -e ".[smac]"

# For SMAC V2
pip install -e ".[smacv2]"

# For Overcooked
pip install -e ".[overcooked]"

# All environments
pip install -e ".[all]"
```

### StarCraft II Setup (for SMAC/SMAC V2)

1. Download StarCraft II from Blizzard
2. Set the `SC2PATH` environment variable:
   ```bash
   export SC2PATH=/path/to/StarCraftII
   ```

## Quick Start

### Training

```bash
# Train with default config (QMIX on SMAC 3m)
python -m marl_research.runners.train

# Train with custom config
python -m marl_research.runners.train \
    algorithm=qmix \
    environment=smac \
    environment.map_name=8m \
    experiment.seed=42

# Train on Overcooked
python -m marl_research.runners.train \
    algorithm=mappo \
    environment=overcooked \
    environment.layout_name=cramped_room
```

### Evaluation

```bash
python -m marl_research.runners.evaluate \
    --checkpoint results/experiment/final_checkpoint.pt \
    --num_episodes 100
```

## Running Experiments

### SMAC Experiments

```bash
# Single map
bash scripts/run_smac.sh --map 3m --algorithm qmix --seeds 5

# All easy maps
bash scripts/run_smac.sh --difficulty easy --seeds 5

# All difficulties
bash scripts/run_smac.sh --difficulty all --seeds 5
```

### SMAC V2 Experiments (with Distribution Shift)

```bash
# Single scenario
bash scripts/run_smac_v2.sh --scenario terran_5_vs_5 --seeds 5

# All Terran scenarios
bash scripts/run_smac_v2.sh --race terran --seeds 5
```

### Overcooked Experiments

```bash
# Single layout
bash scripts/run_overcooked.sh --layout cramped_room --seeds 5

# With zero-shot coordination evaluation
bash scripts/run_overcooked.sh --layout cramped_room --seeds 5 --zero_shot

# All layouts
bash scripts/run_overcooked.sh --category all --seeds 5
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

## Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration management. Configs are composable:

```yaml
# configs/default.yaml
defaults:
  - algorithm: qmix
  - environment: smac

experiment:
  name: "my_experiment"
  seed: 42
  total_timesteps: 2000000
```

Override from command line:

```bash
python -m marl_research.runners.train \
    experiment.name=new_experiment \
    training.lr=0.001 \
    environment.map_name=8m
```

## Adding New Algorithms

1. Create a new file in `algorithms/`:

```python
from marl_research.algorithms.base import BaseAlgorithm
from marl_research.algorithms.registry import register_algorithm

@register_algorithm("my_algorithm")
class MyAlgorithm(BaseAlgorithm):
    def _build_networks(self):
        ...

    def _build_optimizers(self):
        ...

    def select_actions(self, observations, available_actions, explore):
        ...

    def train_step(self, batch):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
```

2. Create config in `configs/algorithm/my_algorithm.yaml`

3. Import in `algorithms/__init__.py`

## Adding New Environments

1. Create wrapper in `environments/`:

```python
from marl_research.environments.base import BaseMAEnv
from marl_research.environments.registry import register_env

@register_env("my_env")
class MyEnv(BaseMAEnv):
    def reset(self):
        ...

    def step(self, actions):
        ...

    def get_env_info(self):
        ...

    def get_available_actions(self):
        ...
```

2. Create config in `configs/environment/my_env.yaml`

3. Import in `environments/__init__.py`

## Logging

Supports both TensorBoard and Weights & Biases:

```yaml
logging:
  use_tensorboard: true
  use_wandb: true
  wandb_project: "marl_research"
```

View TensorBoard logs:

```bash
tensorboard --logdir results/
```

## Implemented Algorithms

- [x] QMIX
- [x] VABL (Variational Attention-based Belief Learning)
- [ ] MAPPO (scaffold provided)
- [ ] IQL (scaffold provided)
- [ ] VDN
- [ ] QPLEX

## VABL Features

VABL includes several configurable ablation parameters:

```bash
# Run with custom ablation parameters
python -m marl_research.runners.train algorithm=vabl \
    algorithm.attention_heads=2 \
    algorithm.warmup_steps=100 \
    algorithm.stop_gradient_belief=true
```

### Training Features

- **Dual Metrics**: Tracks both shaped (training) and sparse (evaluation) rewards
- **Best Checkpoint**: Automatically saves `best_checkpoint.pt` on eval improvement
- **Coordination Metrics**: Tracks `coordination_rate` for multi-agent synchronization

### Ablation Sweeps

Run systematic ablation studies:

```bash
# List available ablations
python -m marl_research.scripts.run_ablation_sweep --list

# Run lambda sweep
python -m marl_research.scripts.run_ablation_sweep --ablation lambda_sweep --seeds 3
```

### Publication Figures

Generate publication-quality plots:

```bash
python -m marl_research.scripts.plot_publication_figure \
    --input results/*.json \
    --output figures/curve.png \
    --show-table
```

## Supported Environments

- [x] SMAC (StarCraft Multi-Agent Challenge)
- [x] SMAC V2 (with distribution shift)
- [x] Overcooked-AI

## Citation

If you use this template, please cite the relevant papers:

```bibtex
@article{samvelyan2019starcraft,
  title={The StarCraft Multi-Agent Challenge},
  author={Samvelyan, Mikayel and others},
  journal={AAMAS},
  year={2019}
}

@article{ellis2022smacv2,
  title={SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning},
  author={Ellis, Benjamin and others},
  journal={NeurIPS},
  year={2022}
}

@article{carroll2019utility,
  title={On the Utility of Learning about Humans for Human-AI Coordination},
  author={Carroll, Micah and others},
  journal={NeurIPS},
  year={2019}
}
```
