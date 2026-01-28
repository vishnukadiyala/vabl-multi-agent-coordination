#!/bin/bash
# Script to run Overcooked experiments

set -e

# Default values
LAYOUT="cramped_room"
ALGORITHM="mappo"
SEEDS=5
CATEGORY=""
ZERO_SHOT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --layout)
            LAYOUT="$2"
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
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --zero_shot)
            ZERO_SHOT="--zero_shot_eval"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "Running Overcooked experiments"
echo "=============================="
echo "Layout: $LAYOUT"
echo "Algorithm: $ALGORITHM"
echo "Seeds: $SEEDS"

if [ -n "$CATEGORY" ]; then
    python -m marl_research.experiments.overcooked.run_experiments \
        algorithm=$ALGORITHM \
        --category $CATEGORY \
        --seeds $SEEDS \
        $ZERO_SHOT
else
    python -m marl_research.experiments.overcooked.run_experiments \
        algorithm=$ALGORITHM \
        environment.layout_name=$LAYOUT \
        --layout $LAYOUT \
        --seeds $SEEDS \
        $ZERO_SHOT
fi
