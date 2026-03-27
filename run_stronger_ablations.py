#!/usr/bin/env python
"""
Stronger VABL Ablation Study for ICML 2026
============================================

Runs 4 key VABL mechanism ablations on Overcooked (asymmetric_advantages)
with 3 seeds and 200 episodes each, producing publication-quality results.

Ablation configurations:
1. Full VABL (attention=True, aux_loss=True, lambda=0.05)
2. No Attention (attention=False, aux_loss=True) - mean pooling
3. No Aux Loss (attention=True, aux_loss=False, lambda=0.0)
4. No Attn + No Aux (attention=False, aux_loss=False)

Usage:
    python run_stronger_ablations.py
"""

import json
import os
import sys
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime

# Force unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ["PYTHONUNBUFFERED"] = "1"

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np

# NumPy 2.0 compatibility shim for overcooked-ai (which uses np.Inf)
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "Bool"):
    np.Bool = np.bool_
if not hasattr(np, "Int"):
    np.Int = np.int_
if not hasattr(np, "Float"):
    np.Float = np.float64
if not hasattr(np, "Complex"):
    np.Complex = np.complex128
if not hasattr(np, "Object"):
    np.Object = np.object_
if not hasattr(np, "String"):
    np.String = np.str_

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Ensure marl_research is importable
sys.path.insert(0, str(Path(__file__).parent))

from marl_research.utils.misc import get_device, set_seed
from marl_research.utils import ReplayBuffer
from marl_research.algorithms import ALGORITHM_REGISTRY
from marl_research.environments import make_env


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = {
    "full_vabl": {
        "name": "Full VABL",
        "use_attention": True,
        "use_aux_loss": True,
        "aux_lambda": 0.05,
    },
    "no_attention": {
        "name": "No Attention",
        "use_attention": False,
        "use_aux_loss": True,
        "aux_lambda": 0.05,
    },
    "no_aux_loss": {
        "name": "No Aux Loss",
        "use_attention": True,
        "use_aux_loss": False,
        "aux_lambda": 0.0,
    },
    "no_attn_no_aux": {
        "name": "No Attn + No Aux",
        "use_attention": False,
        "use_aux_loss": False,
        "aux_lambda": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_SEEDS = 5
N_EPISODES = 500
ENV_NAME = "overcooked"
LAYOUT = "asymmetric_advantages"
DEVICE_SELECTION = "auto"

# Output paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_PATH = PROJECT_ROOT / "results" / "ablation_strong_overcooked.json"
FIGURE_PATH = PROJECT_ROOT / "figures" / "ablation_overcooked_strong.png"


# ---------------------------------------------------------------------------
# Config builder (mirrors run_mechanism_ablations.create_ablation_config)
# ---------------------------------------------------------------------------
def create_config(ablation_cfg):
    """Create an OmegaConf experiment configuration for the given ablation."""
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
            "name": "vabl",
            "embed_dim": 64,
            "hidden_dim": 128,
            "attention_dim": 64,
            "aux_hidden_dim": 64,
            "critic_hidden_dim": 128,
            "aux_lambda": ablation_cfg["aux_lambda"],
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
            "use_attention": ablation_cfg["use_attention"],
            "use_aux_loss": ablation_cfg["use_aux_loss"],
            "use_value_norm": True,
            "use_orthogonal_init": True,
            "actor_lr": 0.0005,
            "critic_lr": 0.0005,
        },
        "environment": {
            "name": ENV_NAME,
            "layout_name": LAYOUT,
            "num_agents": 2,
            "horizon": 400,
            "use_shaped_rewards": True,
            "shaped_reward_scale": 1.0,
        },
    }
    return config


# ---------------------------------------------------------------------------
# Single-seed experiment runner
# (mirrors run_mechanism_ablations.run_ablation_experiment exactly)
# ---------------------------------------------------------------------------
def run_single_seed(config_dict, ablation_cfg, n_episodes, seed, device):
    """Run one seed of an ablation experiment. Returns a results dict."""

    set_seed(seed)
    cfg = OmegaConf.create(config_dict)
    cfg.experiment.seed = seed

    # Create environment
    env = make_env(cfg)
    env_info = env.get_env_info()

    # Create VABL algorithm
    algorithm = ALGORITHM_REGISTRY[cfg.algorithm.name](
        config=cfg,
        n_agents=env_info.n_agents,
        obs_shape=env_info.obs_shape,
        state_shape=env_info.state_shape,
        n_actions=env_info.n_actions,
        device=device,
    )

    # Apply ablation flags
    algorithm.agent.use_attention = ablation_cfg["use_attention"]
    algorithm.use_aux_loss = ablation_cfg["use_aux_loss"]

    # Replay buffer
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
            vis_masks = np.ones(
                (env_info.n_agents, env_info.n_agents - 1), dtype=np.float32
            )

            obs_t = torch.FloatTensor(np.array(obs)).unsqueeze(0).to(device)
            avail_t = torch.FloatTensor(np.array(avail)).unsqueeze(0).to(device)
            vis_t = torch.FloatTensor(vis_masks).unsqueeze(0).to(device)

            prev_actions_t = None
            if prev_actions is not None:
                prev_actions_t = torch.LongTensor(prev_actions).unsqueeze(0).to(device)

            with torch.no_grad():
                actions = algorithm.select_actions(
                    obs_t,
                    avail_t,
                    explore=True,
                    prev_actions=prev_actions_t,
                    visibility_masks=vis_t,
                )

            actions_np = actions.squeeze(0).cpu().numpy()
            next_obs, next_state, reward, done, info = env.step(actions_np.tolist())
            next_avail = env.get_available_actions()

            buffer.add_transition(
                obs=np.array(obs),
                state=np.array(state),
                actions=actions_np,
                reward=reward,
                done=done,
                next_obs=np.array(next_obs),
                next_state=np.array(next_state),
                available_actions=np.array(avail),
                next_available_actions=np.array(next_avail),
                visibility_masks=vis_masks,
            )

            episode_reward += reward
            prev_actions = actions_np
            obs, state = next_obs, next_state

        rewards.append(float(episode_reward))

        # Train
        if buffer.can_sample(batch_size=1):
            batch = buffer.sample(batch_size=1)
            metrics = algorithm.train_step(batch)
            aux_losses.append(metrics.get("aux_loss", 0))
            aux_accs.append(metrics.get("aux_accuracy", 0))
            algorithm.update_on_episode_end(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            msg = f"    Episode {episode + 1:3d}/{n_episodes} | Reward: {avg_reward:7.1f}"
            if aux_losses:
                msg += f" | Aux Loss: {aux_losses[-1]:.4f}"
            if aux_accs:
                msg += f" | Aux Acc: {aux_accs[-1]:.3f}"
            print(msg)

    final_reward = float(np.mean(rewards[-20:])) if len(rewards) >= 20 else float(np.mean(rewards))
    best_reward = float(np.max(rewards))
    final_aux_acc = float(np.mean(aux_accs[-10:])) if aux_accs else 0.0

    return {
        "rewards": rewards,
        "final_reward": final_reward,
        "best_reward": best_reward,
        "aux_losses": [float(x) for x in aux_losses],
        "aux_accs": [float(x) for x in aux_accs],
        "aux_accuracy_final": final_aux_acc,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def generate_figure(all_results, figure_path):
    """Generate publication-quality ablation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
    })

    colors = {
        "full_vabl": "#1f77b4",
        "no_attention": "#ff7f0e",
        "no_aux_loss": "#2ca02c",
        "no_attn_no_aux": "#d62728",
    }

    def smooth(data, window=10):
        alpha = 2 / (window + 1)
        smoothed = np.zeros_like(data, dtype=float)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    fig, ax = plt.subplots(figsize=(10, 6))
    smoothing_window = 10

    table_lines = []

    for ablation_key, ablation_data in all_results.items():
        seed_rewards = [np.array(s["rewards"]) for s in ablation_data["per_seed"]]
        reward_matrix = np.array(seed_rewards)  # [n_seeds, n_episodes]
        n_seeds, n_episodes = reward_matrix.shape
        episodes = np.arange(1, n_episodes + 1)
        color = colors.get(ablation_key, "#333333")
        display_name = ablation_data["config_name"]

        # Smooth each seed
        smoothed_matrix = np.array([smooth(reward_matrix[s], smoothing_window) for s in range(n_seeds)])

        # Mean and CI (raw)
        mean_raw = np.mean(reward_matrix, axis=0)

        # Mean and CI (smoothed)
        mean_sm = np.mean(smoothed_matrix, axis=0)
        if n_seeds > 1:
            std_sm = np.std(smoothed_matrix, axis=0, ddof=1)
            se_sm = std_sm / np.sqrt(n_seeds)
            t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)
            ci_sm = t_crit * se_sm

            std_raw = np.std(reward_matrix, axis=0, ddof=1)
            se_raw = std_raw / np.sqrt(n_seeds)
            ci_raw = t_crit * se_raw
        else:
            ci_sm = np.zeros_like(mean_sm)
            ci_raw = np.zeros_like(mean_raw)

        # Plot raw (faint)
        ax.fill_between(episodes, mean_raw - ci_raw, mean_raw + ci_raw,
                        alpha=0.1, color=color, linewidth=0)
        ax.plot(episodes, mean_raw, alpha=0.25, color=color, linewidth=1)

        # Plot smoothed (bold)
        ax.fill_between(episodes, mean_sm - ci_sm, mean_sm + ci_sm,
                        alpha=0.25, color=color, linewidth=0)
        ax.plot(episodes, mean_sm, color=color, linewidth=2.5, label=display_name)

        # Table summary
        final_mean = ablation_data["summary"]["final_mean"]
        final_std = ablation_data["summary"]["final_std"]
        best_mean = ablation_data["summary"]["best_mean"]
        best_std = ablation_data["summary"]["best_std"]
        table_lines.append(
            f"{display_name:<18} {final_mean:>6.1f} +/- {final_std:<5.1f}  {best_mean:>6.1f} +/- {best_std:<5.1f}"
        )

    # Inset summary table
    header = f"{'Config':<18} {'Final (last 20)':<16} {'Best':<16}"
    separator = "-" * 56
    table_text = header + "\n" + separator + "\n" + "\n".join(table_lines)
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(
        0.02, 0.98, table_text,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top", fontfamily="monospace", bbox=props,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("VABL Mechanism Ablation -- Overcooked (asymmetric_advantages)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(figure_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {figure_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    device = get_device(DEVICE_SELECTION)

    print("=" * 72)
    print("VABL Stronger Ablation Study for ICML 2026")
    print("=" * 72)
    print(f"Environment : {ENV_NAME} / {LAYOUT}")
    print(f"Episodes    : {N_EPISODES}")
    print(f"Seeds       : {N_SEEDS}")
    print(f"Device      : {device}")
    print(f"Ablations   : {list(ABLATION_CONFIGS.keys())}")
    print(f"Results path: {RESULTS_PATH}")
    print(f"Figure path : {FIGURE_PATH}")
    print(f"Start time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    all_results = {}
    global_start = time.time()

    for ablation_key, ablation_cfg in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 72}")
        print(f"  Ablation: {ablation_cfg['name']}")
        print(f"    use_attention = {ablation_cfg['use_attention']}")
        print(f"    use_aux_loss  = {ablation_cfg['use_aux_loss']}")
        print(f"    aux_lambda    = {ablation_cfg['aux_lambda']}")
        print("=" * 72)

        config_dict = create_config(ablation_cfg)
        seed_results = []

        for seed_idx in range(N_SEEDS):
            seed = seed_idx  # seeds 0, 1, 2
            print(f"\n  --- Seed {seed} ---")
            t0 = time.time()
            result = run_single_seed(config_dict, ablation_cfg, N_EPISODES, seed, device)
            elapsed = time.time() - t0
            print(f"  Seed {seed} done in {elapsed:.1f}s | Final: {result['final_reward']:.2f} | Best: {result['best_reward']:.2f}")
            seed_results.append(result)

        # Aggregate across seeds
        final_rewards = [r["final_reward"] for r in seed_results]
        best_rewards = [r["best_reward"] for r in seed_results]
        aux_accs = [r["aux_accuracy_final"] for r in seed_results]

        summary = {
            "final_mean": float(np.mean(final_rewards)),
            "final_std": float(np.std(final_rewards)),
            "best_mean": float(np.mean(best_rewards)),
            "best_std": float(np.std(best_rewards)),
            "aux_accuracy_mean": float(np.mean(aux_accs)),
            "aux_accuracy_std": float(np.std(aux_accs)),
        }

        all_results[ablation_key] = {
            "config_name": ablation_cfg["name"],
            "config": ablation_cfg,
            "per_seed": seed_results,
            "summary": summary,
        }

        print(f"\n  {ablation_cfg['name']} aggregate:")
        print(f"    Final Reward : {summary['final_mean']:.2f} +/- {summary['final_std']:.2f}")
        print(f"    Best Reward  : {summary['best_mean']:.2f} +/- {summary['best_std']:.2f}")
        print(f"    Aux Accuracy : {summary['aux_accuracy_mean']:.3f} +/- {summary['aux_accuracy_std']:.3f}")

    total_elapsed = time.time() - global_start

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("Summary Table")
    print("=" * 72)
    print(f"{'Ablation':<22} {'Final Reward':<22} {'Best Reward':<22} {'Aux Acc':<14}")
    print("-" * 72)
    for key, data in all_results.items():
        s = data["summary"]
        print(
            f"{data['config_name']:<22} "
            f"{s['final_mean']:>6.1f} +/- {s['final_std']:<6.1f}   "
            f"{s['best_mean']:>6.1f} +/- {s['best_std']:<6.1f}   "
            f"{s['aux_accuracy_mean']:.3f}"
        )

    # -----------------------------------------------------------------------
    # Save JSON results
    # -----------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "experiment": "VABL Mechanism Ablation (Strong)",
        "env": ENV_NAME,
        "layout": LAYOUT,
        "episodes": N_EPISODES,
        "seeds": N_SEEDS,
        "device": str(device),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": round(total_elapsed, 1),
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Generate figure
    # -----------------------------------------------------------------------
    print("\nGenerating ablation figure...")
    generate_figure(all_results, FIGURE_PATH)

    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
