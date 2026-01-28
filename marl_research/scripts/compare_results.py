import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_mean_std(results_list):
    """Aggregate results across seeds."""
    rewards_matrix = []
    
    for res in results_list:
        rewards_matrix.append(res['rewards'])
        
    rewards_matrix = np.array(rewards_matrix)
    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    
    return mean_rewards, std_rewards

def compare_experiments(file_paths, labels, output_name='ablation_comparison.png'):
    all_data = []
    for path in file_paths:
        all_data.append(load_results(path))
        
    # Assume all files have the same structure (list of experiments for --full, or list of one experiment)
    # matching by environment/layout
    
    # Structure of data[i] is [ {'env': ..., 'layout': ..., 'results': [...]}, ... ]
    
    n_envs = len(all_data[0])
    
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]
        
    for env_idx in range(n_envs):
        ax = axes[env_idx]
        env_name_title = ""
        
        for exp_idx, data in enumerate(all_data):
            exp_data = data[env_idx] # Assume same order of experiments
            
            env_name = exp_data.get('env', 'unknown')
            layout = exp_data.get('layout')
            title_name = f"{env_name} ({layout})" if layout else env_name
            env_name_title = title_name
            
            results_list = exp_data.get('results', [])
            if not results_list:
                continue
                
            mean, std = get_mean_std(results_list)
            episodes = np.arange(1, len(mean) + 1)
            
            label = labels[exp_idx]
            ax.plot(episodes, mean, label=label)
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)
            
        ax.set_title(env_name_title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_name)
    print(f"Comparison plot saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help='List of JSON result files')
    parser.add_argument('--labels', nargs='+', required=True, help='Labels for each file')
    parser.add_argument('--output', type=str, default='ablation_comparison.png', help='Output filename')
    
    args = parser.parse_args()
    
    assert len(args.files) == len(args.labels), "Number of files and labels must match"
    
    compare_experiments(args.files, args.labels, args.output)
