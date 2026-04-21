"""Analyze the hyperparameter sensitivity sweep results.

Compares heads in {1, 2, 8} (from scripts/run_hyperparam_sweep.sh) against
the canonical heads=4 Full VABL baseline from results/remote_pull/phase2/.

Reports:
- Per-seed Final50 and drop-from-peak for each heads value
- Aggregate mean/std across seeds
- Cohen's d vs canonical heads=4 baseline
- Verdict: is the pathology hyperparameter-robust?

Expected structure for sweep JSONs (from train_vabl_vec_fast.py):
    rewards: list[float]  # per-episode reward
    final_reward, best_reward, config: dict with attention_heads
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HEADS_VALUES = [1, 2, 8]
SEEDS = [0, 1, 2, 3, 4]
SWEEP_DIR = Path("results/hyperparam_sweep")
CANONICAL_FULL_DIR = Path("results/remote_pull/phase2")
SMOOTH_WINDOW = 50
FINAL_WINDOW = 50


def compute_metrics(rewards: list[float]) -> dict:
    """Peak (smoothed), Final50, drop = peak - final50, peak position."""
    arr = np.asarray(rewards, dtype=np.float64)
    smoothed = np.convolve(arr, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="valid")
    peak_idx = int(np.argmax(smoothed))
    peak = float(smoothed[peak_idx])
    final50 = float(arr[-FINAL_WINDOW:].mean())
    return {
        "peak": peak,
        "final50": final50,
        "drop": peak - final50,
        "peak_frac": peak_idx / max(1, len(smoothed)),
        "n_episodes": int(arr.size),
    }


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d for independent samples with pooled std."""
    a = np.asarray(a); b = np.asarray(b)
    va = a.var(ddof=1) if a.size > 1 else 0.0
    vb = b.var(ddof=1) if b.size > 1 else 0.0
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(1, a.size + b.size - 2))
    if pooled < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def load_sweep() -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for heads in HEADS_VALUES:
        out[heads] = []
        for seed in SEEDS:
            p = SWEEP_DIR / f"full_heads{heads}_seed{seed}.json"
            if not p.exists():
                print(f"  MISSING: {p}")
                continue
            with open(p) as f:
                d = json.load(f)
            rewards = d.get("rewards") or d.get("episode_rewards") or []
            if not rewards:
                print(f"  EMPTY: {p}")
                continue
            m = compute_metrics(rewards)
            m["seed"] = seed
            m["best_reward"] = d.get("best_reward")
            m["final_reward"] = d.get("final_reward")
            m["config_heads"] = d.get("config", {}).get("attention_heads")
            out[heads].append(m)
    return out


def load_canonical() -> list[dict]:
    """Canonical heads=4 Full VABL baseline from phase2."""
    out: list[dict] = []
    for seed in SEEDS:
        p = CANONICAL_FULL_DIR / f"phase2_A_full_seed{seed}.json"
        if not p.exists():
            print(f"  canonical MISSING: {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        rewards = d.get("rewards") or d.get("episode_rewards") or []
        m = compute_metrics(rewards)
        m["seed"] = seed
        out.append(m)
    return out


def summarize(runs: list[dict], label: str) -> dict:
    finals = [r["final50"] for r in runs]
    drops = [r["drop"] for r in runs]
    peaks = [r["peak"] for r in runs]
    return {
        "label": label,
        "n": len(runs),
        "final50_mean": float(np.mean(finals)) if finals else float("nan"),
        "final50_std": float(np.std(finals)) if finals else float("nan"),
        "drop_mean": float(np.mean(drops)) if drops else float("nan"),
        "drop_std": float(np.std(drops)) if drops else float("nan"),
        "peak_mean": float(np.mean(peaks)) if peaks else float("nan"),
        "finals_per_seed": finals,
        "drops_per_seed": drops,
    }


def print_per_seed(runs: list[dict], label: str) -> None:
    print(f"  {label}:")
    for r in sorted(runs, key=lambda x: x["seed"]):
        print(
            f"    seed={r['seed']}  peak={r['peak']:7.2f} @{r['peak_frac']*100:3.0f}%  "
            f"final50={r['final50']:7.2f}  drop={r['drop']:6.2f}  "
            f"n_eps={r['n_episodes']}"
        )


def main() -> None:
    print("=" * 78)
    print("Hyperparameter sensitivity sweep analysis: attention_heads")
    print("Overcooked AA, Full VABL (attn + aux lambda=0.05), 25k episodes, 5 seeds")
    print("=" * 78)

    sweep = load_sweep()
    canonical = load_canonical()

    if not canonical:
        print("ERROR: no canonical heads=4 baseline found at", CANONICAL_FULL_DIR)
        return

    print()
    print("--- canonical heads=4 baseline ---")
    print_per_seed(canonical, "heads=4 (canonical)")
    canon_summary = summarize(canonical, "heads=4")
    print(
        f"  AGG  final50={canon_summary['final50_mean']:.2f} ± {canon_summary['final50_std']:.2f}  "
        f"drop={canon_summary['drop_mean']:.2f} ± {canon_summary['drop_std']:.2f}"
    )

    print()
    print("--- sweep: heads in {1, 2, 8} ---")
    sweep_summaries: dict[int, dict] = {}
    for heads in HEADS_VALUES:
        print()
        runs = sweep.get(heads, [])
        if not runs:
            print(f"  heads={heads}: NO RUNS")
            continue
        print_per_seed(runs, f"heads={heads}")
        s = summarize(runs, f"heads={heads}")
        sweep_summaries[heads] = s
        print(
            f"  AGG  final50={s['final50_mean']:.2f} ± {s['final50_std']:.2f}  "
            f"drop={s['drop_mean']:.2f} ± {s['drop_std']:.2f}"
        )

    print()
    print("=" * 78)
    print("Cohen's d vs canonical heads=4 baseline")
    print("=" * 78)
    print(f"  {'heads':>6} {'d(final50)':>14} {'d(drop)':>12}  interpretation")
    for heads in HEADS_VALUES:
        s = sweep_summaries.get(heads)
        if not s:
            continue
        d_final = cohens_d(s["finals_per_seed"], canon_summary["finals_per_seed"])
        d_drop = cohens_d(s["drops_per_seed"], canon_summary["drops_per_seed"])
        interp = "robust (d<0.5)" if abs(d_final) < 0.5 and abs(d_drop) < 0.5 else "SENSITIVE"
        print(f"  {heads:>6d} {d_final:>+14.3f} {d_drop:>+12.3f}  {interp}")

    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)
    # Robust if all heads values produce similar drop magnitudes (within ±5 of baseline)
    # and similar Final50 means (within ±10 of baseline).
    robust = True
    details: list[str] = []
    for heads, s in sweep_summaries.items():
        final_gap = abs(s["final50_mean"] - canon_summary["final50_mean"])
        drop_gap = abs(s["drop_mean"] - canon_summary["drop_mean"])
        if final_gap > 10 or drop_gap > 5:
            robust = False
            details.append(
                f"heads={heads}: final gap={final_gap:.1f}, drop gap={drop_gap:.1f}"
            )
    if robust and sweep_summaries:
        print("  PATHOLOGY IS HYPERPARAMETER-ROBUST")
        print("  All heads values produce Final50 within 10 and drop within 5 of heads=4.")
        print("  Paper framing holds; we can cite this as evidence in Section 6/Discussion.")
    elif sweep_summaries:
        print("  PATHOLOGY IS HEAD-SENSITIVE (framing needs attention)")
        for d in details:
            print(" ", d)
    else:
        print("  NO SWEEP DATA LOADED")

    # Dump a machine-readable summary
    summary_path = SWEEP_DIR / "summary.json"
    dump = {
        "canonical_heads4": canon_summary,
        "sweep": {str(h): s for h, s in sweep_summaries.items()},
    }
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(dump, f, indent=2)
    print()
    print(f"Machine-readable summary written to {summary_path}")


if __name__ == "__main__":
    main()
