#!/bin/bash
# R1 (NeurIPS 2026 rebuttal): KL-instrumented reruns of the four expB
# conditions, adding direct consecutive-policy KL measurement (Sigma_pi) and
# denser gradient-cosine logging (every 5 iterations, was 25).
#
# Answers: yGKw Q1 (estimate Sigma_pi via consecutive-policy KL, correlate
# with cosine-std and Final50 degradation) and fXvf Q6 (measure
# consecutive-policy KL to connect drift with cosine variability).
#
# Matrix: 4 configs x 5 seeds on Overcooked AA, 10M env steps.
# ~10 min/run on the RTX 5090 -> ~3.5 h total. Crash recovery: skips existing.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_R1_kl_instrumented.sh 2>&1 | tee results/logs/R1_kl.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/R1_kl_instrumented
GRAD_LOG_INTERVAL=5

mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/R1_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --log-policy-kl \
        --log-gradient-decomp --grad-log-interval "$GRAD_LOG_INTERVAL" \
        "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $name seed=$seed" || log "DONE: $name seed=$seed"
    return $rc
}

log "R1 KL-instrumented runs STARTED"

for seed in $SEEDS; do
    run_one "full"     "$seed" --aux-lambda 0.05
    run_one "no_aux"   "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "stopgrad" "$seed" --aux-lambda 0.05 --stop-gradient-belief
    run_one "frozen"   "$seed" --aux-lambda 0.05 --aux-frozen-target-policy
done

log "R1 COMPLETE"
