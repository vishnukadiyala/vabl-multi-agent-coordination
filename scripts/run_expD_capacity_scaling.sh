#!/bin/bash
# Experiment D: aux-capacity scaling.
#
# If the pathology scales with aux-network capacity, that supports the
# capacity-consumption counter-narrative. If pathology is similar across
# aux hidden dims, capacity is not the driver.
#
# Matrix: aux_hidden_dim in {16, 32, 128} (default is 64, already run as
# Full). 3 variants x 5 seeds on Overcooked AA, 10M env steps.
# Total: 15 runs, ~15h on an RTX 5090 GPU.

set -u
PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64; N_EPISODES=25000; HORIZON=400; LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/expD_capacity_scaling
mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local hdim=$1; local seed=$2
    local save="$OUT_DIR/expD_auxhid${hdim}_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP: $save"; return 0; fi
    log "START: aux_hidden_dim=$hdim seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --aux-lambda 0.05 --aux-hidden-dim "$hdim"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc) hdim=$hdim seed=$seed" || log "DONE hdim=$hdim seed=$seed"
}

for hdim in 16 32 128; do
    for seed in $SEEDS; do
        run_one "$hdim" "$seed"
    done
done
log "ExpD COMPLETE"
