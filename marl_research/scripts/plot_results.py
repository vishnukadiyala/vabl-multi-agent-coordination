import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle different data structures (full vs single experiment)
    if isinstance(data, list):
        # Likely from --full or the structure I saved: [{'env':..., 'results': [...]}]
        # Actually my save logic for --full saves `all_results` which is a list, but wait.
        # In --full loop: all_results = [results_seed0, results_seed1...] NO.
        # In --full loop: `experiments` list.
        # Inside loop: `all_results` collects seeds.
        # BUT `all_results` is overwritten in each iteration of `experiments`.
        # AND I didn't collect the results of ALL experiments in `run_vabl_experiments.py` for --full.
        # I need to fix `run_vabl_experiments.py` to collect ALL experiments data for --full case properly.
        pass

    # Iterate over all experiments in the data
    for i, experiment_data in enumerate(data):
        env_name = experiment_data.get('env', 'unknown')
        layout = experiment_data.get('layout')
        if layout:
            env_name = f"{env_name}_{layout}"
            
        results_list = experiment_data.get('results', [])
        
        if not results_list:
            print(f"No results found for {env_name}.")
            continue

        # Aggregate data across seeds
        rewards_matrix = []
        aux_loss_matrix = []
        aux_acc_matrix = []

        for res in results_list:
            rewards_matrix.append(res['rewards'])
            aux_loss_matrix.append(res['aux_loss'])
            aux_acc_matrix.append(res['aux_accuracy'])

        rewards_matrix = np.array(rewards_matrix)
        aux_loss_matrix = np.array(aux_loss_matrix)
        aux_acc_matrix = np.array(aux_acc_matrix)

        mean_rewards = np.mean(rewards_matrix, axis=0)
        std_rewards = np.std(rewards_matrix, axis=0)

        mean_loss = np.mean(aux_loss_matrix, axis=0)
        mean_acc = np.mean(aux_acc_matrix, axis=0)
        
        
        episodes_rewards = np.arange(1, len(mean_rewards) + 1)
        episodes_loss = np.arange(1, len(mean_loss) + 1)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Rewards
        axes[0].plot(episodes_rewards, mean_rewards, label='VABL')
        axes[0].fill_between(episodes_rewards, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
        axes[0].set_title(f'Rewards ({env_name})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].grid(True)

        # Aux Loss
        axes[1].plot(episodes_loss, mean_loss, color='orange', label='Aux Loss')
        axes[1].set_title(f'Auxiliary Loss ({env_name})')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)

        # Aux Accuracy
        axes[2].plot(episodes_loss, mean_acc, color='green', label='Aux Accuracy')
        axes[2].set_title(f'Auxiliary Accuracy ({env_name})')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Accuracy')
        axes[2].grid(True)
        
        plt.tight_layout()
        plot_filename = f"{Path(json_path).stem}_{env_name}.png"
        plt.savefig(plot_filename)
        print(f"Plots saved to {plot_filename}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, help='Path to results JSON file')
    args = parser.parse_args()
    plot_results(args.json_file)
