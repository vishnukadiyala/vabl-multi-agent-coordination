#!/bin/bash
# Extra 5 seeds of A_full and A_neither on Overcooked AA (seeds 5-9).
# Goal: tighten the Neither vs Full bootstrap CI that crosses zero at n=5.
# Combined with existing seeds 0-4 this gives n=10 for the two cells that
# matter most for the Limitations "statistical power" claim.
#
# Total: 2 configs x 5 new seeds = 10 runs, ~1h on an RTX 5090 GPU.
#
# Crash recovery: skips existing JSONs.
#
# Usage (on the training GPU server):
#   screen -S extra_seeds
#   cd ~/aux-loss-considered-harmful
#   bash scripts/run_extra_seeds_aa.sh 2>&1 | tee results/extra_seeds_aa.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
NEW_SEEDS="5 6 7 8 9"
OUT_DIR=results/extra_seeds_aa

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/extra_${name}_seed${seed}.json"

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
log "Extra seeds (Full and Neither on AA) STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | Seeds: $NEW_SEEDS"
log "============================================="

for seed in $NEW_SEEDS; do
    run_one "A_full"    "$seed" --aux-lambda 0.05
    run_one "A_neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

log "============================================="
log "Extra seeds COMPLETE"
log "Results in: $OUT_DIR/"
log "Merge with existing phase2 seeds 0-4 for n=10 bootstrap analysis."
log "============================================="
