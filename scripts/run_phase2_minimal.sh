#!/bin/bash
# Phase 2 minimal diagnostic: fresh VABL runs on the finalized vabl_v2 code.
#
# Goal: confirm or refute that at 10M steps, Full VABL is worse than its own
# ablations on Overcooked AA, and test the three fix paths for the
# aux+attention interaction documented in wiki/concepts/aux_loss_bug.md.
#
# Matrix (35 runs total, ~3 days on one 5090):
#
#   Group A - 4-config ablation with constant lambda=0.05 (20 runs)
#     cfg: full, no_attn, no_aux, neither  |  seeds 0..4  |  layout AA
#
#   Group B - Full VABL with the three fix paths (15 runs)
#     cfg: full + annealed lambda 0.05->0 over first 50% of training
#     cfg: full + stop-gradient belief to aux head
#     cfg: full + both (annealed + stop-grad)
#     seeds 0..4 each | layout AA
#
# All runs use 25000 episodes x 64 envs = 1.6M envsteps/episode = ~10M total
# environment steps per run. Same horizon (400) as the rebuttal experiments.
#
# Usage (on the training GPU server):
#   screen -S phase2_minimal
#   cd ~/aux-loss-considered-harmful
#   bash scripts/run_phase2_minimal.sh 2>&1 | tee results/phase2_minimal.log
#
# Crash recovery: each run saves its own JSON. If a run already exists in
# results/phase2/, it is skipped. Re-running the script picks up where it left off.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/phase2

mkdir -p "$OUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/phase2_${name}_seed${seed}.json"

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
log "Phase 2 minimal diagnostic STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | N envs: $N_ENVS | Seeds: $SEEDS"
log "============================================="

# ===== Group A: 4-config ablation with constant lambda=0.05 =====
log "=== Group A: 4-config ablation ==="

for seed in $SEEDS; do
    # Full VABL (attn + aux, constant lambda=0.05)
    run_one "A_full"    "$seed" --aux-lambda 0.05

    # No Attention (mean pool + aux, constant lambda=0.05)
    run_one "A_no_attn" "$seed" --aux-lambda 0.05 --no-attention

    # No Aux (attn + aux off)
    run_one "A_no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0

    # Neither (mean pool + aux off)
    run_one "A_neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

# ===== Group B: Full VABL with three fix paths =====
log "=== Group B: Fix paths on Full VABL ==="

for seed in $SEEDS; do
    # B1: annealed lambda 0.05 -> 0 over first 50% of training
    run_one "B_anneal"       "$seed" --aux-lambda 0.05 --aux-anneal-fraction 0.5

    # B2: stop-gradient on belief into aux head
    run_one "B_stopgrad"     "$seed" --aux-lambda 0.05 --stop-gradient-belief

    # B3: both (annealed + stop-grad)
    run_one "B_anneal_stopgrad" "$seed" --aux-lambda 0.05 --aux-anneal-fraction 0.5 --stop-gradient-belief
done

log "============================================="
log "Phase 2 minimal diagnostic COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
