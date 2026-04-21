#!/bin/bash
# Script to run SMAC V2 experiments with distribution shift

set -e

# Default values
SCENARIO="terran_5_vs_5"
ALGORITHM="qmix"
SEEDS=5
RACE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --race)
            RACE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Running SMAC V2 experiments"
echo "==========================="
echo "Scenario: $SCENARIO"
echo "Algorithm: $ALGORITHM"
echo "Seeds: $SEEDS"

if [ -n "$RACE" ]; then
    python -m marl_research.experiments.smac_v2.run_experiments \
        algorithm=$ALGORITHM \
        --race $RACE \
        --seeds $SEEDS
else
    python -m marl_research.experiments.smac_v2.run_experiments \
        algorithm=$ALGORITHM \
        environment.map_name=$SCENARIO \
        --scenario $SCENARIO \
        --seeds $SEEDS
fi
