# VABL Implementation and Execution Plan

## Paper: "Implicit Coordination via Attention-Driven Latent Belief Representations in Partially Observable Environments" (ICML 2026)

---

## Executive Summary

The VABL (Variational Attention-based Belief Learning) algorithm is **already fully implemented** in this codebase. This plan covers:
1. Understanding the existing implementation
2. Setting up the environments
3. Running the experiments
4. Ablation studies for paper reproduction

---

## Part 1: Implementation Status (COMPLETE)

### Core Algorithm Components

| Component | File | Status |
|-----------|------|--------|
| VABL Algorithm | `marl_research/algorithms/vabl.py` | Complete |
| VABL Networks | `marl_research/algorithms/vabl_networks.py` | Complete |
| VABL Config | `marl_research/configs/algorithm/vabl.yaml` | Complete |
| Trainer with visibility masks | `marl_research/runners/train.py` | Complete |
| Replay Buffer with visibility masks | `marl_research/utils/replay_buffer.py` | Complete |

### Architecture Implementation (Section 5.6 of paper)

```
VABLAgent Network:
├── phi_net (Observation Encoder): obs_dim -> 64 -> ReLU -> 64 -> ReLU
├── psi_net (Action Encoder): n_actions -> 64 -> ReLU -> 64 -> ReLU
├── Attention Mechanism (Scaled Dot-Product):
│   ├── W_q: hidden_dim (128) -> attention_dim (64)
│   ├── W_k: embed_dim (64) -> attention_dim (64)
│   └── W_v: embed_dim (64) -> attention_dim (64)
├── GRU Cell: input (64+64=128) -> hidden_dim (128)
├── Policy Head: hidden_dim (128) -> n_actions
└── Aux Head (MLP): hidden_dim (128) -> 64 -> n_actions * (n_agents-1)
```

### Training Objective (Eq. 11)

```
L_total = L_policy + value_loss_coef * L_value + entropy_coef * L_entropy + aux_lambda * L_aux
```

Where:
- `L_policy`: PPO clipped surrogate objective
- `L_value`: MSE value loss with centralized critic
- `L_entropy`: Entropy bonus for exploration
- `L_aux`: Teammate action prediction loss (Eq. 10)

---

## Part 2: Environment Setup

### Step 1: Install Base Package

```bash
# Navigate to project directory
cd "C:\Users\Vishnu Kadiyala\VS Code\ICML"

# Install the package in development mode
pip install -e .
```

### Step 2: Install Environment Dependencies

**Option A: All environments (recommended)**
```bash
pip install -e ".[all]"
```

**Option B: Specific environments**
```bash
# SMAC only
pip install -e ".[smac]"

# SMAC V2 only
pip install -e ".[smacv2]"

# Overcooked only
pip install -e ".[overcooked]"
```

### Step 3: StarCraft II Setup (for SMAC/SMAC V2)

```bash
# Download StarCraft II (Linux)
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip SC2.4.10.zip -d ~/

# Set environment variable
export SC2PATH=~/StarCraftII

# For Windows, download from Blizzard and set:
# set SC2PATH=C:\Program Files (x86)\StarCraft II
```

### Step 4: Verify Installation

```bash
# Test SMAC
python -c "from smac.env import StarCraft2Env; env = StarCraft2Env(map_name='3m'); env.close(); print('SMAC OK')"

# Test Overcooked
python -c "from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld; print('Overcooked OK')"
```

---

## Part 3: Running Experiments

### Quick Test (Verify Everything Works)

```bash
# Run a quick training test with VABL on SMAC 3m
python -m marl_research.runners.train algorithm=vabl environment=smac environment.map_name=3m experiment.total_timesteps=10000 experiment.name=vabl_test
```

### Main Experiments

#### Experiment 1: SMAC (Partial Observability Test)

```bash
# Run VABL on easy maps
python -m marl_research.runners.train \
    algorithm=vabl \
    environment=smac \
    environment.map_name=3m \
    experiment.total_timesteps=2000000 \
    experiment.seed=42 \
    experiment.name=vabl_smac_3m

# Maps to test: 3m, 8m, 2s3z, 3s5z, 5m_vs_6m
```

#### Experiment 2: SMAC V2 (Generalization Test)

```bash
python -m marl_research.runners.train \
    algorithm=vabl \
    environment=smac_v2 \
    environment.scenario=terran_5_vs_5 \
    experiment.total_timesteps=2000000 \
    experiment.name=vabl_smacv2_terran5v5

# Scenarios: terran_5_vs_5, zerg_5_vs_5, protoss_5_vs_5
```

#### Experiment 3: Overcooked (Coordination Test)

```bash
python -m marl_research.runners.train \
    algorithm=vabl \
    environment=overcooked \
    environment.layout_name=cramped_room \
    experiment.total_timesteps=1000000 \
    experiment.name=vabl_overcooked_cramped

# Layouts: cramped_room, asymmetric_advantages, coordination_ring, forced_coordination
```

### Batch Experiment Script

Create and run experiments across multiple seeds:

```bash
# Using the provided scripts
bash scripts/run_smac.sh --map 3m --algorithm vabl --seeds 5
bash scripts/run_overcooked.sh --layout cramped_room --seeds 5
```

Or run all experiments:

```bash
# Edit run_all_experiments.sh to use ALGORITHM="vabl"
bash scripts/run_all_experiments.sh
```

---

## Part 4: Ablation Studies

### Ablation 1: Effect of Auxiliary Loss (lambda)

```bash
# Baseline: lambda=0 (no auxiliary loss - standard recurrent attention)
python -m marl_research.runners.train \
    algorithm=vabl \
    algorithm.aux_lambda=0.0 \
    environment=smac \
    environment.map_name=3m \
    experiment.name=vabl_ablation_lambda0

# Test different lambda values: 0.0, 0.1, 0.5, 1.0, 2.0
for lambda in 0.0 0.1 0.5 1.0 2.0; do
    python -m marl_research.runners.train \
        algorithm=vabl \
        algorithm.aux_lambda=$lambda \
        environment=smac \
        environment.map_name=3m \
        experiment.name=vabl_ablation_lambda${lambda}
done
```

### Ablation 2: Attention vs No Attention

Modify `vabl_networks.py` to skip attention:
```python
# In forward(), replace attention with zero context:
context = torch.zeros(batch_size, self.attention_dim, device=device)
```

### Ablation 3: Network Architecture

```bash
# Test different hidden dimensions
for hidden in 64 128 256; do
    python -m marl_research.runners.train \
        algorithm=vabl \
        algorithm.hidden_dim=$hidden \
        environment=smac \
        environment.map_name=3m \
        experiment.name=vabl_ablation_hidden${hidden}
done
```

---

## Part 5: Baselines for Comparison

### QMIX Baseline

```bash
python -m marl_research.runners.train \
    algorithm=qmix \
    environment=smac \
    environment.map_name=3m \
    experiment.name=qmix_smac_3m
```

### MAPPO Baseline (if implemented)

```bash
python -m marl_research.runners.train \
    algorithm=mappo \
    environment=smac \
    environment.map_name=3m \
    experiment.name=mappo_smac_3m
```

---

## Part 6: Monitoring and Visualization

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir results/

# View at http://localhost:6006
```

### Weights & Biases (Optional)

```bash
# Enable W&B logging
python -m marl_research.runners.train \
    algorithm=vabl \
    logging.use_wandb=true \
    logging.wandb_project=vabl_icml2026 \
    ...
```

### Key Metrics to Monitor

1. **Training Metrics:**
   - `train/reward`: Episode reward
   - `train/win_rate`: Battle win rate (SMAC)
   - `train/aux_loss`: Auxiliary prediction loss (should decrease)
   - `train/aux_accuracy`: Teammate action prediction accuracy

2. **Evaluation Metrics:**
   - `eval/reward_mean`: Mean evaluation reward
   - `eval/win_rate`: Evaluation win rate

---

## Part 7: Expected Results

Based on the paper claims:

1. **SMAC:** VABL should achieve higher win rates than QMIX/MAPPO on partial observability scenarios
2. **Overcooked:** VABL should coordinate faster (fewer timesteps to reach coordination)
3. **Auxiliary Loss:** As `aux_loss` decreases, `aux_accuracy` should increase, indicating the belief is learning to predict teammates
4. **Ablation:** `aux_lambda=0` (no auxiliary loss) should perform worse, demonstrating the importance of the Theory of Mind objective

---

## Part 8: Full Experiment Pipeline

### Complete Reproduction Script

```bash
#!/bin/bash
# reproduce_paper.sh

# Configuration
SEEDS=5
TIMESTEPS=2000000

# SMAC Experiments
for MAP in "3m" "8m" "2s3z" "3s5z" "5m_vs_6m"; do
    for ALGO in "vabl" "qmix"; do
        for SEED in $(seq 0 $((SEEDS-1))); do
            python -m marl_research.runners.train \
                algorithm=$ALGO \
                environment=smac \
                environment.map_name=$MAP \
                experiment.seed=$SEED \
                experiment.total_timesteps=$TIMESTEPS \
                experiment.name=${ALGO}_smac_${MAP}_seed${SEED}
        done
    done
done

# Overcooked Experiments
for LAYOUT in "cramped_room" "asymmetric_advantages" "coordination_ring"; do
    for SEED in $(seq 0 $((SEEDS-1))); do
        python -m marl_research.runners.train \
            algorithm=vabl \
            environment=overcooked \
            environment.layout_name=$LAYOUT \
            experiment.seed=$SEED \
            experiment.total_timesteps=1000000 \
            experiment.name=vabl_overcooked_${LAYOUT}_seed${SEED}
    done
done

# Ablation: Auxiliary loss weight
for LAMBDA in 0.0 0.1 0.5 1.0 2.0; do
    for SEED in $(seq 0 $((SEEDS-1))); do
        python -m marl_research.runners.train \
            algorithm=vabl \
            algorithm.aux_lambda=$LAMBDA \
            environment=smac \
            environment.map_name=3m \
            experiment.seed=$SEED \
            experiment.name=vabl_ablation_lambda${LAMBDA}_seed${SEED}
    done
done

echo "All experiments completed!"
```

---

## Troubleshooting

### Common Issues

1. **SMAC connection error:**
   ```bash
   export SC2PATH=/path/to/StarCraftII
   ```

2. **CUDA out of memory:**
   ```bash
   python -m marl_research.runners.train hardware.device=cpu ...
   ```

3. **Visibility masks not updating:**
   - Check that `get_visibility_masks()` is implemented in the environment wrapper
   - For SMAC, visibility is based on sight range

4. **Auxiliary loss not decreasing:**
   - Increase `aux_lambda` gradually
   - Check that `visibility_masks` are being passed correctly
   - Verify `next_actions` alignment in training step

---

## Summary

The VABL implementation is complete and ready to run. The key steps are:

1. **Install dependencies:** `pip install -e ".[all]"`
2. **Set up StarCraft II:** Export `SC2PATH`
3. **Run experiments:** Use the training commands above
4. **Monitor progress:** TensorBoard at `results/`
5. **Analyze results:** Compare VABL vs baselines, verify auxiliary loss behavior
