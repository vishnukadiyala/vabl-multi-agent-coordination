# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Agent Reinforcement Learning (MARL) research template implementing VABL (Variational Attention-based Belief Learning) for ICML 2026. The codebase supports SMAC, SMAC V2, and Overcooked environments with PyTorch and Hydra configuration management.

## Project Structure

```
ICML/
├── marl_research/           # Main package
│   ├── algorithms/          # VABL, QMIX, and other algorithms
│   ├── environments/        # SMAC, Overcooked wrappers
│   ├── runners/             # Training and evaluation
│   ├── scripts/             # Experiment runners and utilities
│   ├── configs/             # Hydra YAML configurations
│   └── utils/               # Logging, replay buffer, misc utilities
├── results/                 # Experiment output JSON files
├── figures/                 # Generated plots and visualizations
└── docs/                    # Documentation files
```

## Common Commands

### Installation
```bash
cd marl_research
pip install -e .                    # Basic installation
pip install -e ".[overcooked]"      # With Overcooked (recommended for testing)
pip install -e ".[all]"             # All environments + dev tools
```

### Running VABL Experiments (GPU-enabled)
```bash
# Quick test (20 episodes)
python -m marl_research.scripts.run_quick_test --device auto

# Full experiments with GPU
python -m marl_research.scripts.run_vabl_experiments --env simple --episodes 100 --device cuda

# Run ablation study (VABL vs QMIX)
python -m marl_research.scripts.run_ablation_study --device cuda

# Long comparison run for paper
python -m marl_research.scripts.run_long_comparison --device cuda

# Run ablation sweeps (for paper figures)
python -m marl_research.scripts.run_ablation_sweep --ablation lambda_sweep --seeds 3 --episodes 100
python -m marl_research.scripts.run_ablation_sweep --ablation lambda_sweep --wandb  # With wandb logging
python -m marl_research.scripts.run_ablation_sweep --list  # Show all available ablations

# Generate publication figures
python -m marl_research.scripts.plot_publication_figure --input results/*.json --output figures/learning_curve.png --show-table

# Quick test with wandb logging
python -m marl_research.scripts.run_quick_test --wandb
```

### Using Entry Points (after pip install)
```bash
vabl-experiments --env simple --episodes 100 --device cuda
vabl-quick-test --device cuda
vabl-ablation --device cuda
```

### Training with Hydra
```bash
# Default training (QMIX on SMAC 3m)
python -m marl_research.runners.train

# VABL on Overcooked
python -m marl_research.runners.train algorithm=vabl environment=overcooked

# Custom config overrides
python -m marl_research.runners.train algorithm=vabl environment=simple hardware.device=cuda
```

### Development
```bash
black marl_research/                # Format code (line-length=100)
isort marl_research/                # Sort imports
flake8 marl_research/               # Lint
tensorboard --logdir results/       # View logs
```

## Architecture

### Registry Pattern
Both algorithms and environments use a registry pattern for dynamic loading:
- `@register_algorithm("name")` decorator registers algorithms
- `@register_env("name")` decorator registers environments
- Instantiation via config: `ALGORITHM_REGISTRY[name]` and `make_env(config)`

### Core Module Structure

**algorithms/** - All algorithms extend `BaseAlgorithm` from `base.py`:
- `vabl.py`: VABL algorithm with attention-based belief learning (main contribution)
- `vabl_networks.py`: VABLAgent and CentralizedCritic networks
- `qmix.py`: QMIX baseline implementation
- Must implement: `_build_networks()`, `_build_optimizers()`, `select_actions()`, `train_step()`, `save()`, `load()`

**environments/** - All environments extend `BaseMAEnv` from `base.py`:
- `simple_env.py`: Simple coordination environment for quick testing
- `overcooked_env.py`: Overcooked-AI wrapper
- `smac_env.py`: SMAC/SMAC V2 wrapper
- Returns `EnvInfo` dataclass with: n_agents, obs_shape, state_shape, n_actions, episode_limit

**scripts/** - Experiment runners:
- `run_vabl_experiments.py`: Main experiment runner with GPU support
- `run_quick_test.py`: Quick validation test
- `run_ablation_study.py`: VABL vs QMIX comparison
- `run_long_comparison.py`: Extended training runs for paper
- `run_ablation_sweep.py`: Systematic hyperparameter ablation sweeps
- `plot_publication_figure.py`: Publication-grade figure generation
- `compare_results.py`, `plot_results.py`: Result analysis and visualization

**runners/** - Training orchestration:
- `Trainer`: Manages episode collection → buffer storage → batch training → logging
- `Evaluator`: Checkpoint evaluation and cross-play testing

### Data Flow
```
Config → Trainer.__init__() → [Environment + Algorithm + ReplayBuffer]
Training Loop: collect_episode() → buffer.add_transition() → train_step() → log_metrics()
```

### Training Features

**Dual Metrics System**: Tracks both shaped rewards (for training) and sparse rewards (for evaluation):
- `train/shaped_reward`: Dense reward signal used for gradient computation
- `train/sparse_reward`: Task completion reward (e.g., only soup delivery in Overcooked)

**Best Checkpoint Selection**: Automatically saves `best_checkpoint.pt` when eval reward improves:
- Enables early-stopping style model selection without stopping training
- Logs comparison of best vs final performance at training end

**Coordination Metrics**: Tracks agent synchronization (for coordination tasks):
- `train/coordination_rate`: Fraction of steps where all agents took the same action
- `joint_action_agreement`: Binary per-step agreement indicator

## GPU Support

The codebase automatically detects and uses GPU when available:
- `--device auto`: Auto-detect (prefers CUDA, then MPS, then CPU)
- `--device cuda`: Force CUDA GPU
- `--device cpu`: Force CPU

The `get_device()` function in `utils/misc.py` handles device selection.

## Key Utilities

- `utils/logger.py`: `MetricLogger` supporting TensorBoard and W&B
- `utils/replay_buffer.py`: `EpisodeBuffer` and `ReplayBuffer` with visibility mask support
- `utils/misc.py`: `set_seed()`, `get_device()`, `soft_update()`, `hard_update()`

## VABL Ablation Parameters

Configurable via `configs/algorithm/vabl.yaml` or command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_steps` | 50 | Steps before aux loss annealing |
| `attention_heads` | 4 | Number of MHA attention heads |
| `stop_gradient_belief` | false | Detach beliefs in auxiliary loss |
| `aux_decay_rate` | 0.995 | Aux lambda exponential decay rate |
| `min_aux_lambda` | 0.05 | Minimum aux lambda after decay |
| `use_attention` | true | Use MHA vs mean pooling |
| `use_aux_loss` | true | Enable/disable auxiliary loss |

Example override:
```bash
python -m marl_research.runners.train algorithm=vabl algorithm.attention_heads=2 algorithm.warmup_steps=100
```

## Environment Setup

```bash
conda activate icml2026
```

For SMAC/SMAC V2, set the StarCraft II path:
```bash
export SC2PATH=/path/to/StarCraftII
```
