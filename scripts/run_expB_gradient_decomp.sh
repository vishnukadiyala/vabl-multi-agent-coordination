#!/bin/bash
# Experiment B: multi-seed gradient decomposition under four conditions.
#
# Goal: make Figure 1 load-bearing instead of decorative. Currently the
# directional-interference claim rests on one seed, one config. This
# experiment logs the PPO vs aux gradient norms and their cosine at
# regular intervals for ALL 5 seeds across Full, No Aux, Stop-grad, and
# Anneal. The paper's prediction is specific and falsifiable:
#   - Full:     cosine has high variance (directional noise dominant)
#   - No Aux:   no aux gradient (cosine undefined, will log NaN)
#   - Stop-grad: cosine variance SHOULD COLLAPSE (aux severed from encoder)
#   - Anneal:   cosine variance SHOULD SHRINK over time as lambda -> 0
#
# If stop-grad does NOT collapse cosine variance, the directional story
# is wrong: the fix works via some other mechanism.
#
# Matrix: 4 configs x 5 seeds on Overcooked AA, 10M env steps, gradient
# logged every 25 iterations (~16 log points per run).
# Total: 20 runs, ~20h on an RTX 5090 GPU. Crash recovery: skips existing JSONs.
#
# Usage (on the training GPU server):
#   screen -S expB
#   cd ~/aux-loss-considered-harmful
#   bash scripts/run_expB_gradient_decomp.sh 2>&1 | tee results/logs/expB.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/expB_gradient_decomp
GRAD_LOG_INTERVAL=25

mkdir -p "$OUT_DIR"
mkdir -p results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/expB_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --log-gradient-decomp --grad-log-interval "$GRAD_LOG_INTERVAL" \
        "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $name seed=$seed" || log "DONE: $name seed=$seed"
    return $rc
}

log "============================================="
log "ExpB gradient decomposition STARTED"
log "============================================="

for seed in $SEEDS; do
    run_one "full"     "$seed" --aux-lambda 0.05
    run_one "no_aux"   "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "stopgrad" "$seed" --aux-lambda 0.05 --stop-gradient-belief
    run_one "anneal"   "$seed" --aux-lambda 0.05 --aux-anneal-fraction 0.5
done

log "============================================="
log "ExpB COMPLETE. Analyze via scripts/analyze_expB.py"
log "============================================="
