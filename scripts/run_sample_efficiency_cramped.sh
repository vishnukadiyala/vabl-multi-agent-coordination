#!/bin/bash
# Sample-efficiency curves on Overcooked Cramped Room.
# Cross-layout confirmation of the training-budget artifact.
#
# Matrix: 4 configs x 4 budgets x 5 seeds = 80 runs
# Budgets: 1250ep (500K), 6250ep (2.5M), 12500ep (5M), 25000ep (10M)
# Same episode budget as AA version for cross-layout comparison.
#
# Crash recovery: skips existing JSONs.
#
# Usage (on Celestia):
#   screen -S se_cramped
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_sample_efficiency_cramped.sh 2>&1 | tee results/se_cramped.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
HORIZON=400
SEEDS="0 1 2 3 4"
OUT_DIR=results/sample_efficiency_cramped

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local episodes=$2
    local seed=$3
    shift 3
    local budget_tag="${episodes}ep"
    local save="$OUT_DIR/se_cr_${name}_${budget_tag}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name ${budget_tag} seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout cramped_room \
        --episodes "$episodes" \
        --horizon "$HORIZON" \
        --n-envs "$N_ENVS" \
        --seed "$seed" \
        --save "$save" \
        "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): $name ${budget_tag} seed=$seed"
    else
        log "DONE: $name ${budget_tag} seed=$seed"
    fi
    return $rc
}

log "============================================="
log "Sample-efficiency (Cramped Room) STARTED"
log "Budgets: 1250ep 6250ep 12500ep 25000ep"
log "Configs: full no_attn no_aux neither"
log "Seeds: $SEEDS"
log "============================================="

for episodes in 1250 6250 12500 25000; do
    log "=== Budget: ${episodes} episodes ==="
    for seed in $SEEDS; do
        run_one "full"    "$episodes" "$seed" --aux-lambda 0.05
        run_one "no_attn" "$episodes" "$seed" --aux-lambda 0.05 --no-attention
        run_one "no_aux"  "$episodes" "$seed" --no-aux-loss --aux-lambda 0.0
        run_one "neither" "$episodes" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
    done
done

log "============================================="
log "Sample-efficiency (Cramped Room) COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
