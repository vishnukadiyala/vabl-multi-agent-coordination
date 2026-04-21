"""Script to run SMAC experiments."""

import argparse
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from marl_research.runners.train import Trainer
from marl_research.utils.logger import setup_logger


SMAC_MAPS = {
    "easy": ["2s3z", "3m", "8m"],
    "hard": ["3s5z", "5m_vs_6m", "8m_vs_9m"],
    "super_hard": ["3s5z_vs_3s6z", "6h_vs_8z", "corridor", "MMM2"],
}


def run_single_experiment(config: DictConfig, map_name: str, seed: int) -> dict:
    """Run a single SMAC experiment."""
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    config.environment.map_name = map_name
    config.experiment.seed = seed
    config.experiment.name = f"smac_{map_name}_seed{seed}"

    trainer = Trainer(config)
    results = trainer.train()

    return results


def run_map_experiments(config: DictConfig, map_name: str, num_seeds: int = 5):
    """Run experiments on a single map with multiple seeds."""
    results = []
    for seed in range(num_seeds):
        print(f"\n{'='*50}")
        print(f"Running {map_name} - Seed {seed}")
        print(f"{'='*50}\n")

        result = run_single_experiment(config, map_name, seed)
        results.append(result)

    return results


def run_difficulty_experiments(
    config: DictConfig, difficulty: str, num_seeds: int = 5
):
    """Run experiments on all maps of a given difficulty."""
    if difficulty not in SMAC_MAPS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Choose from {list(SMAC_MAPS.keys())}")

    maps = SMAC_MAPS[difficulty]
    all_results = {}

    for map_name in maps:
        all_results[map_name] = run_map_experiments(config, map_name, num_seeds)

    return all_results


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    """Main entry point for SMAC experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default=None, help="Specific map to run")
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["easy", "hard", "super_hard", "all"],
        help="Difficulty level of maps to run",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")

    args, _ = parser.parse_known_args()

    setup_logger(config)

    if args.map:
        results = run_map_experiments(config, args.map, args.seeds)
    elif args.difficulty:
        if args.difficulty == "all":
            results = {}
            for diff in SMAC_MAPS:
                results[diff] = run_difficulty_experiments(config, diff, args.seeds)
        else:
            results = run_difficulty_experiments(config, args.difficulty, args.seeds)
    else:
        results = run_single_experiment(
            config, config.environment.map_name, config.experiment.seed
        )

    print("\nExperiments completed!")
    return results


if __name__ == "__main__":
    main()
