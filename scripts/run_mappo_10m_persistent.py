#!/usr/bin/env python
"""
Persistent MAPPO 10M runner with crash recovery.
Saves results incrementally so progress is never lost.

Usage:
    python scripts/run_mappo_10m_persistent.py --device cuda
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
import gc

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_research.utils.misc import get_device, set_seed
from marl_research.utils import ReplayBuffer
from marl_research.algorithms import ALGORITHM_REGISTRY
from marl_research.environments import make_env


RESULTS_PATH = PROJECT_ROOT / "results" / "mappo_10m_persistent.json"
N_EPISODES = 25000  # 25000 * 400 = 10M steps
N_SEEDS = 3
SAVE_EVERY = 500  # Save progress every 500 episodes


def create_config():
    return {
        "training": {"lr": 0.0005, "gamma": 0.99, "grad_clip": 10.0,
                      "batch_size": 32, "buffer_size": 1000},
        "experiment": {"seed": 42},
        "algorithm": {
            "name": "mappo",
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
        },
        "environment": {
            "name": "overcooked",
            "layout_name": "asymmetric_advantages",
            "num_agents": 2, "horizon": 400,
            "use_shaped_rewards": True, "shaped_reward_scale": 1.0,
            "view_radius": 3,
        },
    }


def load_progress():
    """Load previously saved progress."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"seeds": {}, "completed_seeds": []}


def save_progress(data):
    """Save current progress."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def run_single_seed(config_dict, n_episodes, seed, device, progress_data):
    """Run one seed with periodic saving."""
    seed_key = str(seed)

    # Check if we can resume
    start_episode = 0
    rewards = []
    if seed_key in progress_data["seeds"]:
        existing = progress_data["seeds"][seed_key]
        rewards = existing.get("rewards", [])
        start_episode = len(rewards)
        if start_episode >= n_episodes:
            print(f"  Seed {seed} already complete ({start_episode} episodes)")
            return existing

    print(f"  Seed {seed}: starting from episode {start_episode}")

    set_seed(seed)
    cfg = OmegaConf.create(config_dict)
    cfg.experiment.seed = seed

    env = make_env(cfg)
    env_info = env.get_env_info()

    algorithm = ALGORITHM_REGISTRY["mappo"](
        config=cfg, n_agents=env_info.n_agents, obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape, n_actions=env_info.n_actions, device=device,
    )

    buffer = ReplayBuffer(
        buffer_size=min(cfg.training.buffer_size, 500),  # Limit buffer to save memory
        episode_limit=env_info.episode_limit,
        n_agents=env_info.n_agents, obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape, n_actions=env_info.n_actions,
    )

    for episode in range(start_episode, n_episodes):
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
            try:
                algorithm.train_step(batch)
            except RuntimeError as e:
                print(f"  WARNING: train_step error at ep {episode}: {e}")
                continue
            if hasattr(algorithm, 'update_on_episode_end'):
                algorithm.update_on_episode_end(episode_reward)

        # Periodic GC to prevent memory buildup
        if (episode + 1) % 100 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        if (episode + 1) % 250 == 0:
            avg = np.mean(rewards[-250:])
            best_so_far = max(rewards)
            print(f"    Ep {episode+1:6d}/{n_episodes} | R: {avg:7.1f} | Best: {best_so_far:.0f}")

        # Save progress periodically
        if (episode + 1) % SAVE_EVERY == 0:
            final = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards))
            best = float(max(rewards))
            collapse = (best - final) / best * 100 if best > 0 else 0
            progress_data["seeds"][seed_key] = {
                "rewards": rewards,
                "final_reward": final,
                "best_reward": best,
                "collapse": collapse,
                "episodes_completed": len(rewards),
            }
            save_progress(progress_data)

    # Final save
    final = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else float(np.mean(rewards))
    best = float(max(rewards))
    collapse = (best - final) / best * 100 if best > 0 else 0
    result = {
        "rewards": rewards,
        "final_reward": final,
        "best_reward": best,
        "collapse": collapse,
        "episodes_completed": len(rewards),
    }
    progress_data["seeds"][seed_key] = result
    if seed_key not in progress_data["completed_seeds"]:
        progress_data["completed_seeds"].append(seed_key)
    save_progress(progress_data)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = get_device(args.device)

    print("=" * 72)
    print("MAPPO 10M Persistent Runner")
    print(f"Episodes: {N_EPISODES}, Seeds: {N_SEEDS}, Device: {device}")
    print(f"Saves every {SAVE_EVERY} episodes to {RESULTS_PATH}")
    print("=" * 72)

    progress = load_progress()
    config_dict = create_config()

    for seed in range(N_SEEDS):
        print(f"\n{'='*72}\n  Seed {seed}\n{'='*72}")
        t0 = time.time()
        result = run_single_seed(config_dict, N_EPISODES, seed, device, progress)
        elapsed = time.time() - t0
        print(f"  Seed {seed}: final={result['final_reward']:.1f}, best={result['best_reward']:.1f}, "
              f"collapse={result['collapse']:.0f}% ({elapsed/3600:.1f}h)")

    # Final summary
    print(f"\n{'='*72}\nMAPPO 10M Summary\n{'='*72}")
    finals = [progress["seeds"][str(s)]["final_reward"] for s in range(N_SEEDS)]
    bests = [progress["seeds"][str(s)]["best_reward"] for s in range(N_SEEDS)]
    collapses = [progress["seeds"][str(s)]["collapse"] for s in range(N_SEEDS)]
    print(f"Final: {np.mean(finals):.1f} ± {np.std(finals):.1f}")
    print(f"Best:  {np.mean(bests):.1f} ± {np.std(bests):.1f}")
    print(f"Collapse: {np.mean(collapses):.0f} ± {np.std(collapses):.0f}%")


if __name__ == "__main__":
    main()
