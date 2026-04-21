"""Script to run Overcooked experiments."""

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


OVERCOOKED_LAYOUTS = {
    "coordination": ["cramped_room", "asymmetric_advantages"],
    "complex": ["coordination_ring", "forced_coordination", "counter_circuit"],
}


def run_self_play_training(config: DictConfig, layout: str, seed: int) -> str:
    """Run self-play training on an Overcooked layout."""
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    config.environment.layout_name = layout
    config.experiment.seed = seed
    config.experiment.name = f"overcooked_{layout}_selfplay_seed{seed}"

    trainer = Trainer(config)
    checkpoint_path = trainer.train()

    return checkpoint_path


def run_cross_play_evaluation(
    config: DictConfig,
    checkpoint_path_1: str,
    checkpoint_path_2: str,
    layout: str,
) -> Dict[str, float]:
    """Evaluate two agents playing together (zero-shot coordination)."""
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    config.environment.layout_name = layout

    evaluator = Evaluator(config)
    results = evaluator.cross_play_evaluate(
        checkpoint_path_1, checkpoint_path_2, num_episodes=100
    )

    return results


def run_single_experiment(config: DictConfig, layout: str, seed: int) -> dict:
    """Run a single Overcooked experiment."""
    print(f"\n{'='*50}")
    print(f"Training on {layout} - Seed {seed}")
    print(f"{'='*50}\n")

    checkpoint_path = run_self_play_training(config, layout, seed)

    return {"checkpoint_path": checkpoint_path, "layout": layout, "seed": seed}


def run_layout_experiments(
    config: DictConfig, layout: str, num_seeds: int = 5
) -> List[dict]:
    """Run experiments on a single layout with multiple seeds."""
    results = []
    for seed in range(num_seeds):
        result = run_single_experiment(config, layout, seed)
        results.append(result)
    return results


def run_zero_shot_coordination_eval(
    config: DictConfig, layout: str, checkpoints: List[str]
) -> Dict[str, float]:
    """Evaluate zero-shot coordination by cross-playing trained agents."""
    cross_play_results = {}

    for i, cp1 in enumerate(checkpoints):
        for j, cp2 in enumerate(checkpoints):
            if i < j:
                key = f"agent{i}_vs_agent{j}"
                cross_play_results[key] = run_cross_play_evaluation(
                    config, cp1, cp2, layout
                )

    return cross_play_results


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    """Main entry point for Overcooked experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=str, default=None, help="Specific layout")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["coordination", "complex", "all"],
        help="Layout category to run",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument(
        "--zero_shot_eval",
        action="store_true",
        help="Run zero-shot coordination evaluation",
    )

    args, _ = parser.parse_known_args()

    setup_logger(config)

    if args.layout:
        results = run_layout_experiments(config, args.layout, args.seeds)

        if args.zero_shot_eval:
            checkpoints = [r["checkpoint_path"] for r in results]
            zs_results = run_zero_shot_coordination_eval(
                config, args.layout, checkpoints
            )
            results = {"training": results, "zero_shot_coordination": zs_results}

    elif args.category:
        if args.category == "all":
            results = {}
            for cat in OVERCOOKED_LAYOUTS:
                results[cat] = {}
                for layout in OVERCOOKED_LAYOUTS[cat]:
                    results[cat][layout] = run_layout_experiments(
                        config, layout, args.seeds
                    )
        else:
            results = {}
            for layout in OVERCOOKED_LAYOUTS[args.category]:
                results[layout] = run_layout_experiments(config, layout, args.seeds)
    else:
        results = run_single_experiment(
            config, config.environment.layout_name, config.experiment.seed
        )

    print("\nOvercooked experiments completed!")
    return results


if __name__ == "__main__":
    main()
