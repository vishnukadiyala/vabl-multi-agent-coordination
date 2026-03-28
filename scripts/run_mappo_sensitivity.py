#!/usr/bin/env python
"""
MAPPO sensitivity sweep for ICML 2026 rebuttal.
Tests whether MAPPO collapse is a hyperparameter artifact (Reviewer 6RAp Q5).

Tests MAPPO with:
1. Default: entropy_coef=0.01, constant
2. Linear decay: 0.01 -> 0.001
3. Exponential decay: 0.01 -> 0.005
4. LR warmup: 0 -> 5e-4 over first 500 episodes
5. VABL (reference): our method for comparison

Usage:
    python scripts/run_mappo_sensitivity.py --device mps
    python scripts/run_mappo_sensitivity.py --device cuda
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


def create_config(algorithm="mappo"):
    return {
        "training": {"lr": 0.0005, "gamma": 0.99, "grad_clip": 10.0,
                      "batch_size": 32, "buffer_size": 1000},
        "experiment": {"seed": 42},
        "algorithm": {
            "name": algorithm,
            "embed_dim": 64, "hidden_dim": 128, "attention_dim": 64,
            "aux_hidden_dim": 64, "critic_hidden_dim": 128,
            "aux_lambda": 0.05 if algorithm == "vabl" else 0.0,
            "clip_param": 0.2, "ppo_epochs": 10, "value_clip": 0.2,
            "value_loss_coef": 0.5, "entropy_coef": 0.01, "gae_lambda": 0.95,
            "warmup_steps": 50, "attention_heads": 4,
            "stop_gradient_belief": False, "aux_decay_rate": 1.0,
            "min_aux_lambda": 0.0, "use_attention": True, "use_aux_loss": True,
            "use_value_norm": True, "use_orthogonal_init": True,
            "use_identity_encoding": True,
            "actor_lr": 0.0005, "critic_lr": 0.0005,
        },
        "environment": {
            "name": "overcooked",
            "layout_name": "asymmetric_advantages",
            "num_agents": 2, "horizon": 400,
            "use_shaped_rewards": True, "shaped_reward_scale": 1.0,
        },
    }


def run_single_seed(config_dict, n_episodes, seed, device, entropy_schedule="constant", lr_warmup=False):
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

    # Entropy schedule params
    initial_entropy = 0.01
    if entropy_schedule == "linear_decay":
        final_entropy = 0.001
    elif entropy_schedule == "exp_decay":
        final_entropy = 0.005

    rewards = []
    for episode in range(n_episodes):
        # Apply entropy schedule
        if hasattr(algorithm, 'current_entropy_coef'):
            if entropy_schedule == "linear_decay":
                frac = episode / max(n_episodes - 1, 1)
                algorithm.current_entropy_coef = initial_entropy + frac * (final_entropy - initial_entropy)
            elif entropy_schedule == "exp_decay":
                algorithm.current_entropy_coef = max(final_entropy, initial_entropy * (0.9995 ** episode))

        # Apply LR warmup
        if lr_warmup and episode < 500:
            warmup_frac = episode / 500
            for param_group in algorithm.actor_optimizer.param_groups:
                param_group['lr'] = 0.0005 * warmup_frac
            if hasattr(algorithm, 'critic_optimizer'):
                for param_group in algorithm.critic_optimizer.param_groups:
                    param_group['lr'] = 0.0005 * warmup_frac

        reset_result = env.reset()
        obs, state = reset_result[0], reset_result[1]
        algorithm.init_hidden(batch_size=1)
        done = False
        episode_reward = 0
        prev_actions = None

        while not done:
            avail = env.get_available_actions()
            vis_masks = np.ones((env_info.n_agents, env_info.n_agents - 1), dtype=np.float32)

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

        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards[-100:])
            print(f"    Ep {episode+1:5d}/{n_episodes} | R: {avg:7.1f}")

    final = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards))
    best = float(np.max(rewards))
    collapse = (best - final) / best * 100 if best > 0 else 0
    return {"rewards": rewards, "final_reward": final, "best_reward": best, "collapse": collapse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    device = get_device(args.device)
    results_path = PROJECT_ROOT / "results" / "mappo_sensitivity.json"

    CONFIGS = {
        "mappo_default": {"algo": "mappo", "entropy": "constant", "warmup": False,
                          "name": "MAPPO (default)"},
        "mappo_linear_decay": {"algo": "mappo", "entropy": "linear_decay", "warmup": False,
                               "name": "MAPPO (linear ent decay)"},
        "mappo_exp_decay": {"algo": "mappo", "entropy": "exp_decay", "warmup": False,
                            "name": "MAPPO (exp ent decay)"},
        "mappo_lr_warmup": {"algo": "mappo", "entropy": "constant", "warmup": True,
                            "name": "MAPPO (LR warmup)"},
        "vabl_reference": {"algo": "vabl", "entropy": "constant", "warmup": False,
                           "name": "VABL (reference)"},
    }

    print("=" * 72)
    print("MAPPO Sensitivity Sweep")
    print(f"Episodes: {args.episodes}, Seeds: {args.seeds}, Device: {device}")
    print("=" * 72)

    all_results = {}
    for key, cfg_info in CONFIGS.items():
        print(f"\n{'='*72}\n  {cfg_info['name']}\n{'='*72}")
        config_dict = create_config(algorithm=cfg_info["algo"])
        seed_results = []

        for s in range(args.seeds):
            print(f"\n  --- Seed {s} ---")
            t0 = time.time()
            result = run_single_seed(
                config_dict, args.episodes, s, device,
                entropy_schedule=cfg_info["entropy"],
                lr_warmup=cfg_info["warmup"],
            )
            elapsed = time.time() - t0
            print(f"  Seed {s}: final={result['final_reward']:.1f}, best={result['best_reward']:.1f}, "
                  f"collapse={result['collapse']:.0f}% ({elapsed:.0f}s)")
            seed_results.append(result)

        finals = [r["final_reward"] for r in seed_results]
        bests = [r["best_reward"] for r in seed_results]
        collapses = [r["collapse"] for r in seed_results]

        summary = {
            "final_mean": float(np.mean(finals)), "final_std": float(np.std(finals)),
            "best_mean": float(np.mean(bests)), "best_std": float(np.std(bests)),
            "collapse_mean": float(np.mean(collapses)), "collapse_std": float(np.std(collapses)),
        }
        all_results[key] = {"name": cfg_info["name"], "per_seed": seed_results, "summary": summary}
        print(f"\n  {cfg_info['name']}: Collapse={summary['collapse_mean']:.0f}±{summary['collapse_std']:.0f}%")

    # Summary
    print(f"\n{'='*72}\nMAPPO Sensitivity Summary\n{'='*72}")
    print(f"{'Config':<28} {'Final':<18} {'Best':<18} {'Collapse':<12}")
    print("-" * 76)
    for k, d in all_results.items():
        s = d["summary"]
        print(f"{d['name']:<28} {s['final_mean']:>6.1f} ± {s['final_std']:<6.1f}  "
              f"{s['best_mean']:>6.1f} ± {s['best_std']:<6.1f}  {s['collapse_mean']:>5.0f}±{s['collapse_std']:.0f}%")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"experiment": "MAPPO sensitivity", "episodes": args.episodes,
                    "seeds": args.seeds, "timestamp": datetime.now().isoformat(),
                    "results": all_results}, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
