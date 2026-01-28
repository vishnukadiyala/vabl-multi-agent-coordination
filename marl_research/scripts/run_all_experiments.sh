#!/bin/bash
# Run all experiments for paper reproduction

set -e

SEEDS=5
ALGORITHM="qmix"

echo "=========================================="
echo "Running All MARL Experiments"
echo "=========================================="

# SMAC experiments
echo ""
echo "1. Running SMAC experiments..."
echo "=========================================="

for MAP in "3m" "8m" "2s3z" "3s5z" "5m_vs_6m"; do
    echo "Running SMAC map: $MAP"
    bash scripts/run_smac.sh --map $MAP --algorithm $ALGORITHM --seeds $SEEDS
done

# SMAC V2 experiments
echo ""
echo "2. Running SMAC V2 experiments..."
echo "=========================================="

for SCENARIO in "terran_5_vs_5" "zerg_5_vs_5" "protoss_5_vs_5"; do
    echo "Running SMAC V2 scenario: $SCENARIO"
    bash scripts/run_smac_v2.sh --scenario $SCENARIO --algorithm $ALGORITHM --seeds $SEEDS
done

# Overcooked experiments
echo ""
echo "3. Running Overcooked experiments..."
echo "=========================================="

for LAYOUT in "cramped_room" "asymmetric_advantages" "coordination_ring"; do
    echo "Running Overcooked layout: $LAYOUT"
    bash scripts/run_overcooked.sh --layout $LAYOUT --algorithm mappo --seeds $SEEDS --zero_shot
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
