#!/usr/bin/env python
"""
Run AERIAL + TarMAC + VABL + MAPPO comparison on Overcooked.
For ICML 2026 rebuttal.

Usage:
    python scripts/run_baseline_comparison.py --device cuda
    python scripts/run_baseline_comparison.py --device cuda --layout cramped_room
    python scripts/run_baseline_comparison.py --device cuda --env overcooked_ego
"""

import json
import os
import sys
import time
import warnings
import logging
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
if not hasattr(np, "Inf"): np.Inf = np.inf
if not hasattr(np, "Bool"): np.Bool = np.bool_

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_research.utils.misc import get_device, set_seed
from marl_research.utils import ReplayBuffer
from marl_research.algorithms import ALGORITHM_REGISTRY
from marl_research.environments import make_env


METHODS = {
    "vabl": {
        "name": "VABL",
        "algorithm": "vabl",
        "extra_config": {
            "aux_lambda": 0.05,
            "use_attention": True,
            "use_aux_loss": True,
            "use_identity_encoding": True,
        },
    },
    "aerial": {
        "name": "AERIAL",
        "algorithm": "aerial",
        "extra_config": {},
    },
    "tarmac": {
        "name": "TarMAC",
        "algorithm": "tarmac",
        "extra_config": {},
    },
    "mappo": {
        "name": "MAPPO",
        "algorithm": "mappo",
        "extra_config": {},
    },
}


def create_config(method_cfg, env_name, layout, horizon):
    config = {
        "training": {"lr": 0.0005, "gamma": 0.99, "grad_clip": 10.0,
                      "batch_size": 32, "buffer_size": 1000},
        "experiment": {"seed": 42},
        "algorithm": {
            "name": method_cfg["algorithm"],
            "embed_dim": 64, "hidden_dim": 128, "attention_dim": 64,
            "aux_hidden_dim": 64, "critic_hidden_dim": 128,
            "message_dim": 64, "key_dim": 64, "comm_rounds": 1,
            "aux_lambda": 0.0,
            "clip_param": 0.2, "ppo_epochs": 10, "value_clip": 0.2,
            "value_loss_coef": 0.5, "entropy_coef": 0.01, "gae_lambda": 0.95,
            "warmup_steps": 50, "attention_heads": 4,
            "stop_gradient_belief": False, "aux_decay_rate": 1.0,
            "min_aux_lambda": 0.0, "use_attention": True, "use_aux_loss": True,
            "use_value_norm": True, "use_orthogonal_init": True,
            "use_identity_encoding": True,
            "actor_lr": 0.0005, "critic_lr": 0.0005,
            **method_cfg.get("extra_config", {}),
        },
        "environment": {
            "name": env_name,
            "layout_name": layout,
            "num_agents": 2, "horizon": horizon,
            "use_shaped_rewards": True, "shaped_reward_scale": 1.0,
            "view_radius": 3,  # only used by overcooked_ego
        },
    }
    return config


def run_single_seed(config_dict, n_episodes, seed, device):
    set_seed(seed)
    cfg = OmegaConf.create(config_dict)
    cfg.experiment.seed = seed

    env = make_env(cfg)
    env_info = env.get_env_info()

    algorithm = ALGORITHM_REGISTRY[cfg.algorithm.name](
        config=cfg, n_agents=env_info.n_agents, obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape, n_actions=env_info.n_actions, device=device,
    )

    buffer = ReplayBuffer(
        buffer_size=cfg.training.buffer_size, episode_limit=env_info.episode_limit,
        n_agents=env_info.n_agents, obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape, n_actions=env_info.n_actions,
    )

    rewards = []
    for episode in range(n_episodes):
        reset_result = env.reset()
        obs, state = (reset_result[0], reset_result[1])

        algorithm.init_hidden(batch_size=1)
        done = False
        episode_reward = 0
        prev_actions = None

        while not done:
            avail = env.get_available_actions()
            vis_masks = np.ones((env_info.n_agents, env_info.n_agents - 1), dtype=np.float32)
            if hasattr(env, 'get_visibility_masks'):
                vis_masks = env.get_visibility_masks()

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
            algorithm.train_step(batch)
            if hasattr(algorithm, 'update_on_episode_end'):
                algorithm.update_on_episode_end(episode_reward)

        if (episode + 1) % 25 == 0:
            avg = np.mean(rewards[-25:])
            print(f"    Ep {episode+1:4d}/{n_episodes} | R: {avg:7.1f}")

    final = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))
    best = float(np.max(rewards))
    collapse = (best - final) / best * 100 if best > 0 else 0
    return {"rewards": rewards, "final_reward": final, "best_reward": best, "collapse": collapse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env", default="overcooked", choices=["overcooked", "overcooked_ego"])
    parser.add_argument("--layout", default="asymmetric_advantages")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--method", default=None, choices=list(METHODS.keys()))
    args = parser.parse_args()

    device = get_device(args.device)
    methods_to_run = {args.method: METHODS[args.method]} if args.method else METHODS
    tag = f"{args.env}_{args.layout}"
    results_path = PROJECT_ROOT / "results" / f"baseline_comparison_{tag}.json"

    print("=" * 72)
    print(f"Baseline Comparison: {tag}")
    print(f"Episodes: {args.episodes}, Seeds: {args.seeds}, Device: {device}")
    print(f"Methods: {list(methods_to_run.keys())}")
    print("=" * 72)

    all_results = {}
    for method_key, method_cfg in methods_to_run.items():
        print(f"\n{'='*72}\n  {method_cfg['name']}\n{'='*72}")
        config_dict = create_config(method_cfg, args.env, args.layout, args.horizon)
        seed_results = []

        for s in range(args.seeds):
            print(f"\n  --- Seed {s} ---")
            t0 = time.time()
            result = run_single_seed(config_dict, args.episodes, s, device)
            print(f"  Seed {s}: final={result['final_reward']:.1f}, best={result['best_reward']:.1f}, "
                  f"collapse={result['collapse']:.0f}% ({time.time()-t0:.0f}s)")
            seed_results.append(result)

        finals = [r["final_reward"] for r in seed_results]
        bests = [r["best_reward"] for r in seed_results]
        collapses = [r["collapse"] for r in seed_results]

        summary = {
            "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals)),
            "best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
            "collapse_mean": float(np.mean(collapses)),
        }
        all_results[method_key] = {"name": method_cfg["name"], "per_seed": seed_results, "summary": summary}
        print(f"\n  {method_cfg['name']}: Final={summary['final_mean']:.1f}±{summary['final_std']:.1f}, "
              f"Best={summary['best_mean']:.1f}±{summary['best_std']:.1f}, Collapse={summary['collapse_mean']:.0f}%")

    # Summary
    print(f"\n{'='*72}\nSummary: {tag}\n{'='*72}")
    print(f"{'Method':<12} {'Final':<18} {'Best':<18} {'Collapse':<10}")
    print("-" * 58)
    for k, d in all_results.items():
        s = d["summary"]
        print(f"{d['name']:<12} {s['final_mean']:>6.1f} ± {s['final_std']:<6.1f}  "
              f"{s['best_mean']:>6.1f} ± {s['best_std']:<6.1f}  {s['collapse_mean']:>5.0f}%")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"experiment": tag, "episodes": args.episodes, "seeds": args.seeds,
                    "timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
