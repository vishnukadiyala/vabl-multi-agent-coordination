#!/usr/bin/env python
"""
5-Agent Coordination Experiments for ICML 2026 Rebuttal
========================================================

Runs VABL vs MAPPO (with ablations) on 5-agent Simple Coordination,
directly testing selective attention over multiple teammates.

This addresses Reviewer 4's concern that 2-agent Overcooked makes
attention trivially weight=1.0, and demonstrates that VABL's selective
attention mechanism is exercised with 4 visible teammates.

Usage:
    python scripts/run_5agent_experiments.py
    python scripts/run_5agent_experiments.py --device cuda
    python scripts/run_5agent_experiments.py --episodes 200 --seeds 3  # quick test
"""

import json
import os
import sys
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ["PYTHONUNBUFFERED"] = "1"

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np

# NumPy 2.0 compatibility
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "Bool"):
    np.Bool = np.bool_

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import argparse

# Ensure marl_research is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_research.utils.misc import get_device, set_seed
from marl_research.utils import ReplayBuffer
from marl_research.algorithms import ALGORITHM_REGISTRY
from marl_research.environments import make_env


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------
METHODS = {
    "vabl_full": {
        "name": "VABL (Full)",
        "algorithm": "vabl",
        "use_attention": True,
        "use_aux_loss": True,
        "aux_lambda": 0.05,
    },
    "vabl_no_attn": {
        "name": "VABL (No Attn)",
        "algorithm": "vabl",
        "use_attention": False,
        "use_aux_loss": True,
        "aux_lambda": 0.05,
    },
    "vabl_no_aux": {
        "name": "VABL (No Aux)",
        "algorithm": "vabl",
        "use_attention": True,
        "use_aux_loss": False,
        "aux_lambda": 0.0,
    },
    "mappo": {
        "name": "MAPPO",
        "algorithm": "mappo",
        "use_attention": False,
        "use_aux_loss": False,
        "aux_lambda": 0.0,
    },
}


def create_config(method_cfg, n_agents=5):
    """Create config for 5-agent simple coordination."""
    config = {
        "training": {
            "lr": 0.0005,
            "gamma": 0.99,
            "grad_clip": 10.0,
            "batch_size": 32,
            "buffer_size": 1000,
        },
        "experiment": {
            "seed": 42,
        },
        "algorithm": {
            "name": method_cfg["algorithm"],
            "embed_dim": 64,
            "hidden_dim": 128,
            "attention_dim": 64,
            "aux_hidden_dim": 64,
            "critic_hidden_dim": 128,
            "aux_lambda": method_cfg["aux_lambda"],
            "clip_param": 0.2,
            "ppo_epochs": 10,
            "value_clip": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "gae_lambda": 0.95,
            "warmup_steps": 50,
            "attention_heads": 4,
            "stop_gradient_belief": False,
            "aux_decay_rate": 1.0,
            "min_aux_lambda": 0.0,
            "use_attention": method_cfg.get("use_attention", True),
            "use_aux_loss": method_cfg.get("use_aux_loss", True),
            "use_value_norm": True,
            "use_orthogonal_init": True,
            "use_identity_encoding": True,
            "actor_lr": 0.0005,
            "critic_lr": 0.0005,
        },
        "environment": {
            "name": "simple",
            "n_agents": n_agents,
            "obs_dim": 16,
            "n_actions": 5,
            "episode_limit": 50,
            "visibility_prob": 0.7,
        },
    }
    return config


def run_single_seed(config_dict, method_cfg, n_episodes, seed, device):
    """Run one seed. Returns results dict."""
    set_seed(seed)
    cfg = OmegaConf.create(config_dict)
    cfg.experiment.seed = seed

    env = make_env(cfg)
    env_info = env.get_env_info()

    algorithm = ALGORITHM_REGISTRY[cfg.algorithm.name](
        config=cfg,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
        device=device,
    )

    # Apply ablation flags (only for VABL)
    if hasattr(algorithm, 'agent') and hasattr(algorithm.agent, 'use_attention'):
        algorithm.agent.use_attention = method_cfg.get("use_attention", True)
    if hasattr(algorithm, 'use_aux_loss'):
        algorithm.use_aux_loss = method_cfg.get("use_aux_loss", True)

    buffer = ReplayBuffer(
        buffer_size=cfg.training.buffer_size,
        episode_limit=env_info.episode_limit,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
    )

    rewards = []
    aux_losses = []
    aux_accs = []
    attention_entropies = []

    for episode in range(n_episodes):
        reset_result = env.reset()
        if len(reset_result) == 3:
            obs, state, _ = reset_result
        else:
            obs, state = reset_result

        algorithm.init_hidden(batch_size=1)
        done = False
        episode_reward = 0
        prev_actions = None

        while not done:
            avail = env.get_available_actions()
            # Stochastic visibility for Simple Coordination
            vis_masks = np.random.binomial(
                1, 0.7, size=(env_info.n_agents, env_info.n_agents - 1)
            ).astype(np.float32)

            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
            avail_t = torch.FloatTensor(np.array(avail)).unsqueeze(0).to(device)
            vis_t = torch.FloatTensor(vis_masks).unsqueeze(0).to(device)

            prev_actions_t = None
            if prev_actions is not None:
                prev_actions_t = torch.LongTensor(prev_actions).unsqueeze(0).to(device)

            with torch.no_grad():
                actions = algorithm.select_actions(
                    obs_t, avail_t, explore=True,
                    prev_actions=prev_actions_t, visibility_masks=vis_t,
                )

            actions_np = actions.squeeze(0).cpu().numpy()
            next_obs, next_state, reward, done, info = env.step(actions_np.tolist())
            next_avail = env.get_available_actions()

            buffer.add_transition(
                obs=np.array(obs), state=np.array(state),
                actions=actions_np, reward=reward, done=done,
                next_obs=np.array(next_obs), next_state=np.array(next_state),
                available_actions=np.array(avail),
                next_available_actions=np.array(next_avail),
                visibility_masks=vis_masks,
            )

            episode_reward += reward
            prev_actions = actions_np
            obs, state = next_obs, next_state

        rewards.append(float(episode_reward))

        if buffer.can_sample(batch_size=1):
            batch = buffer.sample(batch_size=1)
            metrics = algorithm.train_step(batch)
            aux_losses.append(metrics.get("aux_loss", 0))
            aux_accs.append(metrics.get("aux_accuracy", 0))
            if hasattr(algorithm, 'update_on_episode_end'):
                algorithm.update_on_episode_end(episode_reward)

        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(rewards[-25:])
            msg = f"    Ep {episode + 1:4d}/{n_episodes} | R: {avg_reward:7.1f}"
            if aux_accs:
                msg += f" | AuxAcc: {aux_accs[-1]:.3f}"
            print(msg)

    final_reward = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))
    best_reward = float(np.max(rewards))

    return {
        "rewards": rewards,
        "final_reward": final_reward,
        "best_reward": best_reward,
        "aux_accuracy_final": float(np.mean(aux_accs[-10:])) if aux_accs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="5-Agent VABL Experiments")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--method", type=str, default=None,
                        choices=list(METHODS.keys()),
                        help="Run single method (default: all)")
    args = parser.parse_args()

    device = get_device(args.device)
    results_path = PROJECT_ROOT / "results" / f"5agent_comparison_{args.n_agents}agents.json"
    figure_path = PROJECT_ROOT / "figures" / f"comparison_{args.n_agents}agent_simple.png"

    methods_to_run = {args.method: METHODS[args.method]} if args.method else METHODS

    print("=" * 72)
    print(f"VABL {args.n_agents}-Agent Coordination Experiments")
    print("=" * 72)
    print(f"Agents      : {args.n_agents}")
    print(f"Episodes    : {args.episodes}")
    print(f"Seeds       : {args.seeds}")
    print(f"Device      : {device}")
    print(f"Methods     : {list(methods_to_run.keys())}")
    print("=" * 72)

    all_results = {}

    for method_key, method_cfg in methods_to_run.items():
        print(f"\n{'=' * 72}")
        print(f"  Method: {method_cfg['name']}")
        print("=" * 72)

        config_dict = create_config(method_cfg, n_agents=args.n_agents)
        seed_results = []

        for seed_idx in range(args.seeds):
            print(f"\n  --- Seed {seed_idx} ---")
            t0 = time.time()
            result = run_single_seed(config_dict, method_cfg, args.episodes, seed_idx, device)
            elapsed = time.time() - t0
            print(f"  Seed {seed_idx} done in {elapsed:.1f}s | Final: {result['final_reward']:.1f} | Best: {result['best_reward']:.1f}")
            seed_results.append(result)

        final_rewards = [r["final_reward"] for r in seed_results]
        best_rewards = [r["best_reward"] for r in seed_results]

        summary = {
            "final_mean": float(np.mean(final_rewards)),
            "final_std": float(np.std(final_rewards)),
            "best_mean": float(np.mean(best_rewards)),
            "best_std": float(np.std(best_rewards)),
        }

        all_results[method_key] = {
            "config_name": method_cfg["name"],
            "per_seed": seed_results,
            "summary": summary,
        }

        print(f"\n  {method_cfg['name']}:")
        print(f"    Final: {summary['final_mean']:.1f} +/- {summary['final_std']:.1f}")
        print(f"    Best:  {summary['best_mean']:.1f} +/- {summary['best_std']:.1f}")

    # Summary table
    print(f"\n{'=' * 72}")
    print(f"Summary: {args.n_agents}-Agent Simple Coordination")
    print("=" * 72)
    print(f"{'Method':<22} {'Final Reward':<22} {'Best Reward':<22}")
    print("-" * 66)
    for key, data in all_results.items():
        s = data["summary"]
        print(f"{data['config_name']:<22} {s['final_mean']:>6.1f} +/- {s['final_std']:<6.1f}   {s['best_mean']:>6.1f} +/- {s['best_std']:<6.1f}")

    # Save results
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "experiment": f"{args.n_agents}-Agent Coordination Comparison",
        "n_agents": args.n_agents,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
