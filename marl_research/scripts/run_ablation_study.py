"""Ablation study comparing VABL vs QMIX.

Usage:
    python -m marl_research.scripts.run_ablation_study
    python -m marl_research.scripts.run_ablation_study --device cuda
"""
import subprocess
import sys
import glob
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def main():
    print("Starting Optimized Ablation Study & Comparison (Parallel)...")

    project_root = get_project_root()
    device_arg = "--device auto"

    # Parse command line for device argument
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            device_arg = f"--device {sys.argv[idx + 1]}"

    # Increase episodes to see learning in Overcooked (100 eps)
    # Compare Proposed VABL vs QMIX Baseline
    cmd_proposed = f"{sys.executable} -m marl_research.scripts.run_vabl_experiments --full --algorithm vabl --aux_lambda 0.5 --exp_name proposed --seeds 2 --episodes 100 {device_arg}"
    cmd_qmix = f"{sys.executable} -m marl_research.scripts.run_vabl_experiments --full --algorithm qmix --exp_name qmix --seeds 2 --episodes 100 {device_arg}"

    print("\n--- Launching Proposed Method (VABL) ---")
    p1 = subprocess.Popen(cmd_proposed, shell=True, cwd=str(project_root))

    print("--- Launching Baseline (QMIX) ---")
    p2 = subprocess.Popen(cmd_qmix, shell=True, cwd=str(project_root))

    print("\nWaiting for experiments to complete (this may take a few minutes)...")
    p1.wait()
    p2.wait()

    print("\nExperiments completed!")

    # Find results
    results_dir = project_root / "results"
    files_proposed = sorted(glob.glob(str(results_dir / "vabl_full_results_proposed_*.json")))
    if not files_proposed:
        print("Error: No proposed results found.")
        return
    proposed_file = files_proposed[-1]

    files_qmix = sorted(glob.glob(str(results_dir / "vabl_full_results_qmix_*.json")))
    if not files_qmix:
        print("Error: No qmix results found.")
        return
    qmix_file = files_qmix[-1]

    print(f"\nProposed file: {proposed_file}")
    print(f"QMIX file: {qmix_file}")

    # Generate Comparison Plot
    print("\n--- Generating Comparison Plot ---")
    output_path = project_root / "figures" / "method_comparison.png"
    plot_cmd = f'"{sys.executable}" -m marl_research.scripts.compare_results --files "{proposed_file}" "{qmix_file}" --labels VABL QMIX --output "{output_path}"'
    subprocess.check_call(plot_cmd, shell=True, cwd=str(project_root))

    print("\nComparison study completed!")
    print(f"Check {output_path}")


if __name__ == "__main__":
    main()
