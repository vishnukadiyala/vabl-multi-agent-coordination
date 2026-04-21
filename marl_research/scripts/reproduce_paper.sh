#!/bin/bash
# reproduce_paper.sh
# Full reproduction script for VABL ICML 2026 paper experiments

set -e

# Configuration
SEEDS=5
TIMESTEPS=2000000
OVERCOOKED_TIMESTEPS=1000000

echo "=============================================="
echo "VABL Paper Reproduction Script"
echo "ICML 2026: Implicit Coordination via Attention-Driven"
echo "Latent Belief Representations"
echo "=============================================="

# Parse arguments
RUN_SMAC=true
RUN_OVERCOOKED=true
RUN_ABLATIONS=true
RUN_BASELINES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --smac-only)
            RUN_OVERCOOKED=false
            RUN_ABLATIONS=false
            shift
            ;;
        --overcooked-only)
            RUN_SMAC=false
            RUN_ABLATIONS=false
            shift
            ;;
        --ablations-only)
            RUN_SMAC=false
            RUN_OVERCOOKED=false
            shift
            ;;
        --no-baselines)
            RUN_BASELINES=false
            shift
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--smac-only] [--overcooked-only] [--ablations-only] [--no-baselines] [--seeds N] [--timesteps N]"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Seeds: $SEEDS"
echo "  Timesteps (SMAC): $TIMESTEPS"
echo "  Timesteps (Overcooked): $OVERCOOKED_TIMESTEPS"
echo "  Run SMAC: $RUN_SMAC"
echo "  Run Overcooked: $RUN_OVERCOOKED"
echo "  Run Ablations: $RUN_ABLATIONS"
echo "  Run Baselines: $RUN_BASELINES"
echo ""

# ============================================
# SECTION 1: SMAC Experiments
# ============================================
if [ "$RUN_SMAC" = true ]; then
    echo ""
    echo "=============================================="
    echo "SECTION 1: SMAC Experiments (Partial Observability)"
    echo "=============================================="

    SMAC_MAPS=("3m" "8m" "2s3z" "3s5z" "5m_vs_6m")

    for MAP in "${SMAC_MAPS[@]}"; do
        echo ""
        echo "--- Running VABL on SMAC map: $MAP ---"

        for SEED in $(seq 0 $((SEEDS-1))); do
            echo "  Seed $SEED..."
            python -m marl_research.runners.train \
                algorithm=vabl \
                environment=smac \
                environment.map_name=$MAP \
                experiment.seed=$SEED \
                experiment.total_timesteps=$TIMESTEPS \
                experiment.name=vabl_smac_${MAP}_seed${SEED} \
                logging.use_tensorboard=true
        done

        # Run baseline
        if [ "$RUN_BASELINES" = true ]; then
            echo ""
            echo "--- Running QMIX baseline on SMAC map: $MAP ---"
            for SEED in $(seq 0 $((SEEDS-1))); do
                echo "  Seed $SEED..."
                python -m marl_research.runners.train \
                    algorithm=qmix \
                    environment=smac \
                    environment.map_name=$MAP \
                    experiment.seed=$SEED \
                    experiment.total_timesteps=$TIMESTEPS \
                    experiment.name=qmix_smac_${MAP}_seed${SEED}
            done
        fi
    done
fi

# ============================================
# SECTION 2: Overcooked Experiments
# ============================================
if [ "$RUN_OVERCOOKED" = true ]; then
    echo ""
    echo "=============================================="
    echo "SECTION 2: Overcooked Experiments (Coordination)"
    echo "=============================================="

    LAYOUTS=("cramped_room" "asymmetric_advantages" "coordination_ring" "forced_coordination")

    for LAYOUT in "${LAYOUTS[@]}"; do
        echo ""
        echo "--- Running VABL on Overcooked layout: $LAYOUT ---"

        for SEED in $(seq 0 $((SEEDS-1))); do
            echo "  Seed $SEED..."
            python -m marl_research.runners.train \
                algorithm=vabl \
                environment=overcooked \
                environment.layout_name=$LAYOUT \
                experiment.seed=$SEED \
                experiment.total_timesteps=$OVERCOOKED_TIMESTEPS \
                experiment.name=vabl_overcooked_${LAYOUT}_seed${SEED}
        done
    done
fi

# ============================================
# SECTION 3: Ablation Studies
# ============================================
if [ "$RUN_ABLATIONS" = true ]; then
    echo ""
    echo "=============================================="
    echo "SECTION 3: Ablation Studies"
    echo "=============================================="

    # Ablation 1: Auxiliary loss weight (lambda)
    echo ""
    echo "--- Ablation 1: Auxiliary Loss Weight (lambda) ---"
    LAMBDAS=("0.0" "0.1" "0.5" "1.0" "2.0")

    for LAMBDA in "${LAMBDAS[@]}"; do
        echo "  Lambda = $LAMBDA"
        for SEED in $(seq 0 $((SEEDS-1))); do
            python -m marl_research.runners.train \
                algorithm=vabl \
                algorithm.aux_lambda=$LAMBDA \
                environment=smac \
                environment.map_name=3m \
                experiment.seed=$SEED \
                experiment.total_timesteps=$TIMESTEPS \
                experiment.name=ablation_lambda${LAMBDA}_seed${SEED}
        done
    done

    # Ablation 2: Hidden dimension
    echo ""
    echo "--- Ablation 2: Hidden Dimension ---"
    HIDDEN_DIMS=("64" "128" "256")

    for HIDDEN in "${HIDDEN_DIMS[@]}"; do
        echo "  Hidden dim = $HIDDEN"
        for SEED in $(seq 0 $((SEEDS-1))); do
            python -m marl_research.runners.train \
                algorithm=vabl \
                algorithm.hidden_dim=$HIDDEN \
                environment=smac \
                environment.map_name=3m \
                experiment.seed=$SEED \
                experiment.total_timesteps=$TIMESTEPS \
                experiment.name=ablation_hidden${HIDDEN}_seed${SEED}
        done
    done

    # Ablation 3: Attention dimension
    echo ""
    echo "--- Ablation 3: Attention Dimension ---"
    ATTN_DIMS=("32" "64" "128")

    for ATTN in "${ATTN_DIMS[@]}"; do
        echo "  Attention dim = $ATTN"
        for SEED in $(seq 0 $((SEEDS-1))); do
            python -m marl_research.runners.train \
                algorithm=vabl \
                algorithm.attention_dim=$ATTN \
                environment=smac \
                environment.map_name=3m \
                experiment.seed=$SEED \
                experiment.total_timesteps=$TIMESTEPS \
                experiment.name=ablation_attention${ATTN}_seed${SEED}
        done
    done
fi

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo ""
echo "Results saved in: results/"
echo "View with: tensorboard --logdir results/"
