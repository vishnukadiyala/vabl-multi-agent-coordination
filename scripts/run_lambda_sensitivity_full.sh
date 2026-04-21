#!/bin/bash
# Full lambda sensitivity + gradient diagnostic controls.
# 5 seeds for EVERYTHING. No shortcuts.
#
# Matrix:
#   Lambda sensitivity at 10M (25K episodes):
#     lambda=0.001 × 5 seeds
#     lambda=0.01  × 5 seeds
#     (lambda=0.05 already done in Phase 2 as A_full)
#
#   Lambda sensitivity at 50M (125K episodes):
#     lambda=0.01  × 5 seeds  (threshold test)
#     lambda=0.05  × 5 seeds  (does pathology persist at 5× budget?)
#
#   Gradient diagnostic controls:
#     A_full diagnostic × 5 seeds
#     A_no_aux diagnostic × 5 seeds
#
# Total: 30 runs. Crash recovery: skips existing JSONs.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
HORIZON=400
SEEDS="0 1 2 3 4"
OUT_DIR=results/lambda_sensitivity

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_vabl() {
    local name=$1
    local episodes=$2
    local seed=$3
    shift 3
    local save="$OUT_DIR/${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed (${episodes}ep)"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout asymmetric_advantages \
        --episodes "$episodes" \
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

run_diag() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START diagnostic: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_diagnostic \
        --layout asymmetric_advantages \
        --episodes 25000 \
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
log "FULL Lambda Sensitivity + Gradient Controls"
log "5 seeds for everything. No shortcuts."
log "============================================="

# === Lambda sensitivity at 10M (25K episodes) ===
log "=== Lambda 0.001 at 10M (5 seeds) ==="
for seed in $SEEDS; do
    run_vabl "lambda0001_10M" 25000 "$seed" --aux-lambda 0.001
done

log "=== Lambda 0.01 at 10M (5 seeds) ==="
for seed in $SEEDS; do
    run_vabl "lambda001_10M" 25000 "$seed" --aux-lambda 0.01
done

# === Lambda sensitivity at 50M (125K episodes) ===
log "=== Lambda 0.01 at 50M (5 seeds) ==="
for seed in $SEEDS; do
    run_vabl "lambda001_50M" 125000 "$seed" --aux-lambda 0.01
done

log "=== Lambda 0.05 at 50M (5 seeds) ==="
for seed in $SEEDS; do
    run_vabl "lambda005_50M" 125000 "$seed" --aux-lambda 0.05
done

# === Gradient diagnostic: A_full × 5 seeds ===
log "=== A_full gradient diagnostic (5 seeds) ==="
for seed in $SEEDS; do
    run_diag "grad_diag_full" "$seed" --aux-lambda 0.05
done

# === Gradient diagnostic: A_no_aux × 5 seeds ===
log "=== A_no_aux gradient diagnostic (5 seeds) ==="
for seed in $SEEDS; do
    run_diag "grad_diag_noaux" "$seed" --no-aux-loss --aux-lambda 0.0
done

log "============================================="
log "FULL Lambda Sensitivity + Controls COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
