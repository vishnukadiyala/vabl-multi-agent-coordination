#!/bin/bash
# SMAX 3v3 ablation: 4 VABL configs × 5 seeds.
# Third cooperative MARL environment (combat coordination).
# 50K episodes × 100 steps = 5M env steps per run.
# ~20 runs × estimated ~15 min each = ~5 hours total.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=50000
HORIZON=100
SEEDS="0 1 2 3 4"
OUT_DIR=results/smax

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/smax_${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_smax \
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
log "SMAX 3v3 ablation STARTED"
log "Episodes: $N_EPISODES | Horizon: $HORIZON | Seeds: $SEEDS"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_attn" "$seed" --aux-lambda 0.05 --no-attention
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

log "============================================="
log "SMAX 3v3 ablation COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
