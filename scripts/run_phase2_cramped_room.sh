#!/bin/bash
# Phase 2 cross-layout: Cramped Room ablation × 5 seeds × 4 configs.
#
# Tests whether the gradient-interference finding from AA generalizes to a
# second Overcooked layout. If yes → strong cross-layout claim for the paper.
# If only on AA → MARL-specific footnote.
#
# Total: 20 runs at ~8.5 min/run = ~3 hours overnight.
#
# Crash recovery: each run saves its own JSON; existing JSONs are skipped.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=cramped_room
SEEDS="0 1 2 3 4"
OUT_DIR=results/phase2_cramped

mkdir -p "$OUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/cramped_${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" \
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
log "Phase 2 Cramped Room ablation STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | N envs: $N_ENVS | Seeds: $SEEDS"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_attn" "$seed" --aux-lambda 0.05 --no-attention
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

log "============================================="
log "Phase 2 Cramped Room ablation COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
