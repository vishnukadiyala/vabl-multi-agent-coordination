"""Long-run comparison study for ICML paper.

Usage:
    python -m marl_research.scripts.run_long_comparison
    python -m marl_research.scripts.run_long_comparison --device cuda
"""
import subprocess
import sys
import glob
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def main():
    print("Starting Long-Run Comparison Study (Parallel)...")
    print("WARNING: This will take several hours to complete.")

    project_root = get_project_root()
    device_arg = "--device auto"

    # Parse command line for device argument
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            device_arg = f"--device {sys.argv[idx + 1]}"

    # Configuration for long run:
    # 5000 episodes to ensure convergence in Overcooked
    # 5 seeds for statistical significance (standard for ICML papers)
    episodes = 5000
    seeds = 5

    cmd_proposed = f"{sys.executable} -m marl_research.scripts.run_vabl_experiments --full --algorithm vabl --aux_lambda 0.5 --exp_name proposed_long --seeds {seeds} --episodes {episodes} {device_arg}"
    cmd_qmix = f"{sys.executable} -m marl_research.scripts.run_vabl_experiments --full --algorithm qmix --exp_name qmix_long --seeds {seeds} --episodes {episodes} {device_arg}"

    print(f"\n--- Launching Proposed Method (VABL) [Episodes: {episodes}, Seeds: {seeds}] ---")
    p1 = subprocess.Popen(cmd_proposed, shell=True, cwd=str(project_root))

    print(f"--- Launching Baseline (QMIX) [Episodes: {episodes}, Seeds: {seeds}] ---")
    p2 = subprocess.Popen(cmd_qmix, shell=True, cwd=str(project_root))

    print("\nExperiments likely running in background. You can monitor progress by checking CPU/GPU usage or 'results/' folder.")
    print("Waiting for completion...")

    try:
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating processes...")
        p1.terminate()
        p2.terminate()
        return

    print("\nExperiments completed!")

    # Find results
    results_dir = project_root / "results"
    files_proposed = sorted(glob.glob(str(results_dir / "vabl_full_results_proposed_long_*.json")))
    if not files_proposed:
        print("Error: No proposed results found.")
        return
    proposed_file = files_proposed[-1]

    files_qmix = sorted(glob.glob(str(results_dir / "vabl_full_results_qmix_long_*.json")))
    if not files_qmix:
        print("Error: No qmix results found.")
        return
    qmix_file = files_qmix[-1]

    print(f"\nProposed file: {proposed_file}")
    print(f"QMIX file: {qmix_file}")

    # Generate Comparison Plot
    print("\n--- Generating Comparison Plot ---")
    output_path = project_root / "figures" / "long_run_comparison.png"
    plot_cmd = f'"{sys.executable}" -m marl_research.scripts.compare_results --files "{proposed_file}" "{qmix_file}" --labels VABL QMIX --output "{output_path}"'
    subprocess.check_call(plot_cmd, shell=True, cwd=str(project_root))

    print("\nLong run comparison completed!")
    print(f"Check {output_path}")


if __name__ == "__main__":
    main()
