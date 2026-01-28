#!/bin/bash
# Script to run SMAC experiments

set -e

# Default values
MAP="3m"
ALGORITHM="qmix"
SEEDS=5
DIFFICULTY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --map)
            MAP="$2"
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
        --difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Running SMAC experiments"
echo "========================"
echo "Map: $MAP"
echo "Algorithm: $ALGORITHM"
echo "Seeds: $SEEDS"

if [ -n "$DIFFICULTY" ]; then
    python -m marl_research.experiments.smac.run_experiments \
        algorithm=$ALGORITHM \
        --difficulty $DIFFICULTY \
        --seeds $SEEDS
else
    python -m marl_research.experiments.smac.run_experiments \
        algorithm=$ALGORITHM \
        environment.map_name=$MAP \
        --map $MAP \
        --seeds $SEEDS
fi
