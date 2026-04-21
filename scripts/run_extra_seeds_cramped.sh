#!/bin/bash
# Extra 5 seeds (5-9) of Full and No-Attn on Overcooked Cramped Room.
# Combined with existing seeds 0-4 (results/remote_pull/phase2_cramped/)
# this gives n=10 for the cells whose bootstrap CI currently crosses zero.
#
# Rationale: the Cramped Room No-Attention vs Full comparison has bootstrap
# CI [-0.84, +2.11] on Cohen's d at n=5. The qualitative pattern (pathology
# worse than mean+aux) is consistent with the AA finding, but the statistical
# test does not resolve the comparison. Need more seeds.
#
# Total: 2 configs x 5 seeds = 10 runs, ~1h on an RTX 5090 GPU (Cramped Room
# is smaller than AA, so should be faster per seed).
# Crash recovery: skips existing JSONs.
#
# Usage (on the training GPU server):
#   cd ~/aux-loss-considered-harmful
#   bash scripts/run_extra_seeds_cramped.sh 2>&1 | tee results/logs/extra_seeds_cramped.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=cramped_room
NEW_SEEDS="5 6 7 8 9"
OUT_DIR=results/extra_seeds_cramped

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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
log "Cramped Room extra seeds (Full and No-Attn, 5-9) STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | Seeds: $NEW_SEEDS"
log "============================================="

for seed in $NEW_SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_attn" "$seed" --aux-lambda 0.05 --no-attention
done

log "============================================="
log "Cramped Room extra seeds COMPLETE"
log "Merge with existing seeds 0-4 for n=10 bootstrap on No-Attn vs Full."
log "============================================="
