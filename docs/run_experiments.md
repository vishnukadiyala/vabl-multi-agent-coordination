# Running VABL Experiments

## Quick Start

Open a terminal and run these commands:

```bash
# 1. Navigate to project directory
cd "C:\Users\Vishnu Kadiyala\VS Code\ICML"

# 2. Activate the conda environment
conda activate icml2026

# 3. Run experiments (choose one)
```

---

## Option 1: Quick Test (Simple Environment)

Fast test to verify everything works (~2 minutes):

```bash
python run_vabl_experiments.py --env simple --episodes 50
```

---

## Option 2: Overcooked Training

### Short run (~5 minutes):
```bash
python run_vabl_experiments.py --env overcooked --layout cramped_room --episodes 50
```

### Medium run (~15 minutes):
```bash
python run_vabl_experiments.py --env overcooked --layout cramped_room --episodes 200
```

### Full training with multiple seeds (~1 hour):
```bash
python run_vabl_experiments.py --env overcooked --layout cramped_room --episodes 500 --seeds 3
```

---

## Option 3: Different Overcooked Layouts

```bash
# Cramped Room (default, easiest)
python run_vabl_experiments.py --env overcooked --layout cramped_room --episodes 100

# Asymmetric Advantages
python run_vabl_experiments.py --env overcooked --layout asymmetric_advantages --episodes 100

# Coordination Ring
python run_vabl_experiments.py --env overcooked --layout coordination_ring --episodes 100
```

---

## Option 4: Full Paper Reproduction

Run all experiments from the paper:

```bash
python run_vabl_experiments.py --full --seeds 3
```

---

## Using the Hydra-based Trainer (Alternative)

For more control, use the main training script:

```bash
# Simple environment
python -m marl_research.runners.train algorithm=vabl environment=simple experiment.total_timesteps=10000

# Overcooked
python -m marl_research.runners.train algorithm=vabl environment=overcooked environment.layout_name=cramped_room experiment.total_timesteps=100000
```

---

## Expected Output

You should see output like:

```
======================================================================
VABL Experiment Runner - ICML 2026 Paper
======================================================================

Environment: simple
Episodes: 50

--- Seed 0 ---
  Episode   20 | Reward:   -3.38 | Aux Loss: 1.6101 | Aux Acc: 20.1%
  Episode   40 | Reward:   29.17 | Aux Loss: 1.5716 | Aux Acc: 29.3%

======================================================================
Final Results
======================================================================
  Reward: 75.20 +/- 11.91
  Aux Loss: 1.5253 +/- 0.0399
  Aux Acc: 35.1% +/- 5.4%
```

---

## Key Metrics to Watch

| Metric | What it means |
|--------|---------------|
| **Reward** | Higher is better (agents coordinating) |
| **Aux Loss** | Should decrease (learning to predict teammates) |
| **Aux Acc** | Should increase above random (16.7% for 6 actions) |

---

## Troubleshooting

**If conda activate fails:**
```bash
conda init
# Close and reopen terminal, then try again
```

**If imports fail:**
```bash
cd "C:\Users\Vishnu Kadiyala\VS Code\ICML\marl_research"
pip install -e .
```

**To verify setup:**
```bash
python test_setup.py
```

---

## Option 5: Ablation Sweeps

Run systematic ablation studies for the paper:

```bash
# List all available ablations
python -m marl_research.scripts.run_ablation_sweep --list

# Run lambda sweep (auxiliary loss weight)
python -m marl_research.scripts.run_ablation_sweep --ablation lambda_sweep --seeds 3 --episodes 100

# Run attention heads sweep
python -m marl_research.scripts.run_ablation_sweep --ablation attention_heads_sweep --seeds 3

# Run stop gradient ablation
python -m marl_research.scripts.run_ablation_sweep --ablation stop_gradient --seeds 5
```

Available ablations:
- `lambda_sweep`: aux_lambda values [0.0, 0.01, 0.1, 0.5, 1.0]
- `belief_dim_sweep`: hidden_dim values [16, 32, 64, 128]
- `attention_heads_sweep`: attention_heads values [1, 2, 4, 8]
- `stop_gradient`: stop_gradient_belief [False, True]
- `no_attention`: use_attention [True, False]
- `no_aux_loss`: use_aux_loss [True, False]
- `warmup_sweep`: warmup_steps values [0, 25, 50, 100, 200]
- `aux_decay_sweep`: aux_decay_rate values [0.99, 0.995, 0.999, 1.0]

---

## Option 6: Generate Publication Figures

Create publication-quality learning curve plots:

```bash
# Basic plot
python -m marl_research.scripts.plot_publication_figure \
    --input results/*.json \
    --output figures/learning_curve.png

# With summary table and baseline
python -m marl_research.scripts.plot_publication_figure \
    --input results/*.json \
    --output figures/learning_curve.png \
    --smoothing-window 10 \
    --baseline-value 0.0 \
    --baseline-label "Random Policy" \
    --show-table \
    --title "VABL vs Baselines"
```

Features:
- Smoothed curves with 95% confidence intervals
- Raw (faint) + smoothed (bold) curves
- Best-100 / Last-100 summary table inset
- Horizontal baseline reference lines
- 300 DPI publication-ready output

---

## New Training Metrics

The trainer now tracks additional metrics:

| Metric | Description |
|--------|-------------|
| `train/shaped_reward` | Dense reward signal (for training) |
| `train/sparse_reward` | Task completion reward (for evaluation) |
| `train/coordination_rate` | Fraction of coordinated actions |

Best checkpoint is automatically saved as `best_checkpoint.pt` when eval reward improves.
