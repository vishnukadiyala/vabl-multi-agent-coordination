#!/bin/bash
# Phase 2 baselines: clean v2-code reruns of MAPPO/TarMAC/CommNet/AERIAL on
# both Overcooked AA and Cramped Room at 10M.
#
# Purpose: provide an apples-to-apples benchmark contribution for the NeurIPS
# paper. All methods on the same JAX code path, same training loop, same
# vmap-vectorized envs, same 5 seeds. The existing vec_10m_*_aa archive is
# from the same code generation as the buggy VABL runs and we want consistency
# even though the baselines themselves were not affected by the VABL aux bug.
#
# Total: 4 algos x 2 layouts x 5 seeds = 40 runs at ~8.5 min/run = ~5.7 hours.
#
# Crash recovery: each run saves its own JSON; existing JSONs are skipped.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
SEEDS="0 1 2 3 4"
ALGOS="mappo tarmac commnet aerial"
LAYOUTS="asymmetric_advantages cramped_room"
OUT_DIR=results/phase2_baselines

mkdir -p "$OUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_one() {
    local algo=$1
    local layout=$2
    local seed=$3
    local save="$OUT_DIR/baseline_${algo}_${layout}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $algo $layout seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_unified \
        --algo "$algo" \
        --layout "$layout" \
        --episodes "$N_EPISODES" \
        --horizon "$HORIZON" \
        --n-envs "$N_ENVS" \
        --seed "$seed" \
        --save "$save"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): $algo $layout seed=$seed"
    else
        log "DONE: $algo $layout seed=$seed"
    fi
    return $rc
}

log "============================================="
log "Phase 2 baselines STARTED"
log "Algos: $ALGOS | Layouts: $LAYOUTS | Seeds: $SEEDS"
log "============================================="

for algo in $ALGOS; do
    for layout in $LAYOUTS; do
        for seed in $SEEDS; do
            run_one "$algo" "$layout" "$seed"
        done
    done
done

log "============================================="
log "Phase 2 baselines COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
