#!/bin/bash
# AERIAL fix-path experiment (reviewer-requested, 2026-04-20).
#
# Goal: test whether our paper's proposed stop-gradient intervention recovers
# AERIAL from the late-training pathology documented in Section 6.1.
#
# Note on methodology: our original AERIAL port (aerial_impl.py) omitted
# Phan et al.'s auxiliary recurrence loss and still produced the 10M-step
# late-training drop (Section 6.1). This script re-enables a VABL-style
# teammate-next-action auxiliary head on AERIAL so the stop-gradient fix
# can be tested in an apples-to-apples setup. We run two configurations,
# each at 3 seeds on Overcooked Asymmetric Advantages, 10M steps:
#
#   1. AERIAL + aux (Full):     attention over teammate beliefs + aux head
#                               with lambda=0.05, gradients flow back through
#                               the belief encoder.
#   2. AERIAL + aux + stopgrad: identical to (1) but with stop_gradient on
#                               the belief before the aux head.
#
# If the paper's mechanism story is correct, (2) should recover the no-aux
# regime; if it fails, the AERIAL pathology has a component our mechanism
# does not explain.
#
# Total: 2 configs x 3 seeds = 6 runs, ~1h on Celestia 5090.
# Crash recovery: skips existing JSONs.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_aerial_fix_path.sh 2>&1 | tee results/logs/aerial_fix_path.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2"
OUT_DIR=results/aerial_fix_path

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_unified \
        --algo aerial \
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
log "AERIAL fix-path experiment STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | Seeds: $SEEDS"
log "============================================="

for seed in $SEEDS; do
    # Full: AERIAL + aux (lambda=0.05), pathological variant
    run_one "aerial_aux_full" "$seed" --aux-lambda 0.05

    # Fix: AERIAL + aux + stop-gradient on belief into aux head
    run_one "aerial_aux_stopgrad" "$seed" --aux-lambda 0.05 --stop-gradient-belief
done

log "============================================="
log "AERIAL fix-path experiment COMPLETE"
log "Compare Final50 across aerial_aux_full vs. aerial_aux_stopgrad"
log "and against the existing AERIAL baseline (no-aux) in results/phase2_baselines/."
log "============================================="
