
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_smac_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Structure: {'vabl': {'3m': ...}, 'qmix': {'3m': ...}}
    
    # Get all maps
    maps = set()
    for algo in data:
        maps.update(data[algo].keys())
    
    print(f"Found maps: {maps}")

    for map_name in maps:
        print(f"Plotting {map_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Rewards and Win Rates
        for algo, color in [('vabl', 'blue'), ('qmix', 'orange')]:
            if map_name not in data[algo]:
                continue
                
            algo_data = data[algo][map_name]
            rewards = algo_data.get('rewards', [])
            win_rates = algo_data.get('win_rates', [])
            
            # Filter out empty runs
            rewards = [r for r in rewards if r]
            win_rates = [w for w in win_rates if w]
            
            if not rewards:
                continue

            # Convert to numpy for mean/std
            # Note: runs might have different lengths if crashed, assuming equal length for now
            # Pad with nan if needed or just truncate to min length
            min_len = min(len(r) for r in rewards)
            rewards = np.array([r[:min_len] for r in rewards])
            win_rates = np.array([w[:min_len] for w in win_rates])
            
            mean_reward = np.mean(rewards, axis=0)
            std_reward = np.std(rewards, axis=0)
            
            mean_win = np.mean(win_rates, axis=0) * 100 # Convert to percentage
            std_win = np.std(win_rates, axis=0) * 100
            
            x = np.arange(1, len(mean_reward) + 1)
            
            # Reward Plot
            axes[0].plot(x, mean_reward, label=algo.upper(), color=color)
            axes[0].fill_between(x, mean_reward - std_reward, mean_reward + std_reward, color=color, alpha=0.2)
            
            # Win Rate Plot
            axes[1].plot(x, mean_win, label=algo.upper(), color=color)
            axes[1].fill_between(x, mean_win - std_win, mean_win + std_win, color=color, alpha=0.2)
            
        axes[0].set_title(f'Mean Reward ({map_name})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title(f'Win Rate % ({map_name})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        output_path = f"smac_plot_{map_name}.png"
        plt.savefig(output_path)
        print(f"Saved {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, help='Path to results JSON file')
    args = parser.parse_args()
    plot_smac_results(args.json_file)
