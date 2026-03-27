#!/usr/bin/env python
"""
Master script for all ICML 2026 rebuttal experiments.
Runs sequentially on a single GPU to avoid contention.

Usage:
    python scripts/run_all_rebuttal_experiments.py --device cuda
    python scripts/run_all_rebuttal_experiments.py --device cuda --skip-ablations  # if already done
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_cmd(description, cmd, timeout_hours=12):
    """Run a command with logging."""
    print(f"\n{'='*72}")
    print(f"  STARTING: {description}")
    print(f"  CMD: {cmd}")
    print(f"  TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        timeout=timeout_hours * 3600,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  FINISHED: {description} [{status}] in {elapsed/60:.1f} min")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all rebuttal experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-5agent", action="store_true")
    parser.add_argument("--skip-ego", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    device = args.device
    seeds = args.seeds

    print("=" * 72)
    print("ICML 2026 REBUTTAL — FULL EXPERIMENT SUITE")
    print("=" * 72)
    print(f"Device: {device}, Seeds: {seeds}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    results = {}

    # ---------------------------------------------------------------
    # 1. ABLATIONS (5 seeds, 500 eps, horizon 400)
    # ---------------------------------------------------------------
    if not args.skip_ablations:
        # Asymmetric Advantages
        results["ablation_aa"] = run_cmd(
            "Ablation: Overcooked Asymmetric Advantages (5s/500ep)",
            f"{py} run_stronger_ablations.py",
        )

        # Cramped Room (modify layout inline)
        results["ablation_cr"] = run_cmd(
            "Ablation: Overcooked Cramped Room (5s/500ep)",
            f"{py} -c \""
            f"exec(open('run_stronger_ablations.py').read()"
            f".replace('asymmetric_advantages','cramped_room')"
            f".replace('ablation_strong_overcooked','ablation_strong_cramped_room')"
            f".replace('ablation_overcooked_strong','ablation_cramped_room_strong'))\"",
        )

    # ---------------------------------------------------------------
    # 2. NEW BASELINES: AERIAL + TarMAC on Overcooked AA
    # ---------------------------------------------------------------
    if not args.skip_baselines:
        for algo in ["aerial", "tarmac"]:
            results[f"baseline_{algo}"] = run_cmd(
                f"Baseline: {algo.upper()} on Overcooked AA (5s/500ep)",
                f"{py} -c \""
                f"exec(open('run_stronger_ablations.py').read()"
                f".replace(\\\"'vabl'\\\", \\\"'{algo}'\\\")"
                f".replace('ABLATION_CONFIGS', 'ABLATION_CONFIGS_UNUSED')"
                f")\"",
            )

    # ---------------------------------------------------------------
    # 3. 5-AGENT Simple Coordination
    # ---------------------------------------------------------------
    if not args.skip_5agent:
        results["5agent"] = run_cmd(
            "5-Agent Simple Coordination (5s/500ep, VABL+MAPPO+ablations)",
            f"{py} scripts/run_5agent_experiments.py --episodes 500 --seeds {seeds} --device {device}",
        )

    # ---------------------------------------------------------------
    # 4. EGO-CENTRIC PO OVERCOOKED
    # ---------------------------------------------------------------
    if not args.skip_ego:
        for algo in ["vabl", "mappo"]:
            results[f"ego_{algo}"] = run_cmd(
                f"Ego-centric PO: {algo.upper()} on Overcooked AA (5s/500ep, r=3)",
                f"{py} scripts/run_5agent_experiments.py --episodes 500 --seeds {seeds} --device {device} "
                f"--method {algo}_full" if algo == "vabl" else
                f"{py} scripts/run_5agent_experiments.py --episodes 500 --seeds {seeds} --device {device} "
                f"--method mappo",
            )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*72}")
    print("EXPERIMENT SUITE COMPLETE")
    print(f"{'='*72}")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
