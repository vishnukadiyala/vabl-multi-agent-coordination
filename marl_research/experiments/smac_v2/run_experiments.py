"""Script to run SMAC V2 experiments with distribution shift evaluation."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from marl_research.runners.train import Trainer
from marl_research.runners.evaluate import Evaluator
from marl_research.utils.logger import setup_logger


SMACV2_SCENARIOS = {
    "terran": ["terran_5_vs_5", "terran_10_vs_10"],
    "zerg": ["zerg_5_vs_5", "zerg_10_vs_10"],
    "protoss": ["protoss_5_vs_5", "protoss_10_vs_10"],
}


def run_training(config: DictConfig, scenario: str, seed: int) -> str:
    """Run training on a SMAC V2 scenario."""
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    config.environment.map_name = scenario
    config.experiment.seed = seed
    config.experiment.name = f"smacv2_{scenario}_seed{seed}"

    config.environment.capability_config = {
        "start_positions": "fixed",
        "unit_types": "fixed",
    }

    trainer = Trainer(config)
    checkpoint_path = trainer.train()

    return checkpoint_path


def run_distribution_shift_eval(
    config: DictConfig, checkpoint_path: str, scenario: str
) -> Dict[str, float]:
    """Evaluate trained model under distribution shift."""
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    config.environment.map_name = scenario
    config.environment.capability_config = {
        "start_positions": "random",
        "unit_types": "random",
    }

    evaluator = Evaluator(config)
    evaluator.load_checkpoint(checkpoint_path)

    results = evaluator.evaluate(num_episodes=32)

    return results


def run_full_experiment(config: DictConfig, scenario: str, seed: int) -> dict:
    """Run full training + distribution shift evaluation."""
    print(f"\n{'='*60}")
    print(f"Training on {scenario} (fixed distribution) - Seed {seed}")
    print(f"{'='*60}\n")

    checkpoint_path = run_training(config, scenario, seed)

    print(f"\n{'='*60}")
    print(f"Evaluating on {scenario} (random distribution)")
    print(f"{'='*60}\n")

    eval_results = run_distribution_shift_eval(config, checkpoint_path, scenario)

    return {
        "training_checkpoint": checkpoint_path,
        "distribution_shift_results": eval_results,
    }


def run_scenario_experiments(
    config: DictConfig, scenario: str, num_seeds: int = 5
) -> List[dict]:
    """Run experiments on a single scenario with multiple seeds."""
    results = []
    for seed in range(num_seeds):
        result = run_full_experiment(config, scenario, seed)
        results.append(result)
    return results


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    """Main entry point for SMAC V2 experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario")
    parser.add_argument(
        "--race",
        type=str,
        default=None,
        choices=["terran", "zerg", "protoss", "all"],
        help="Race scenarios to run",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")

    args, _ = parser.parse_known_args()

    setup_logger(config)

    if args.scenario:
        results = run_scenario_experiments(config, args.scenario, args.seeds)
    elif args.race:
        if args.race == "all":
            results = {}
            for race in SMACV2_SCENARIOS:
                results[race] = {}
                for scenario in SMACV2_SCENARIOS[race]:
                    results[race][scenario] = run_scenario_experiments(
                        config, scenario, args.seeds
                    )
        else:
            results = {}
            for scenario in SMACV2_SCENARIOS[args.race]:
                results[scenario] = run_scenario_experiments(
                    config, scenario, args.seeds
                )
    else:
        results = run_full_experiment(
            config, config.environment.map_name, config.experiment.seed
        )

    print("\nSMAC V2 experiments completed!")
    return results


if __name__ == "__main__":
    main()
