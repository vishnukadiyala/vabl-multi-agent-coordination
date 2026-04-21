#!/bin/bash
# Generic Transformer Agent ablation: 4 configs × 5 seeds on Overcooked AA.
# Tests the pathology on a SECOND architecture (not VABL, not AERIAL).
# If pathology reproduces → proven architectural-class finding.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
SEEDS="0 1 2 3 4"
OUT_DIR=results/transformer

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/transformer_${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_transformer_vec \
        --layout asymmetric_advantages \
        --episodes "$N_EPISODES" \
        --horizon "$HORIZON" \
        --n-envs "$N_ENVS" \
        --seed "$seed" \
        --save "$save" \
        "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): $name seed=$seed"
    else
        log "DONE: $name seed=$seed"
    fi
    return $rc
}

log "============================================="
log "Transformer Agent Ablation on Overcooked AA"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_attn" "$seed" --aux-lambda 0.05 --no-attention
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

log "============================================="
log "Transformer Ablation COMPLETE"
log "============================================="
