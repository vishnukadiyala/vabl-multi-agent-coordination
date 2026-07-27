#!/bin/bash
# R2 (NeurIPS 2026 rebuttal): PCGrad and GradNorm baselines in the Full
# setting (reviewer-requested: fXvf Q5, yGKw W3/Q5).
#
# Falsifiable prediction from the paper's mechanism: both methods target
# magnitude imbalance / persistent conflict; since I_mag ~= 0 in this setting,
# neither should reduce cross-seed Final50 variance. Either outcome is
# reported honestly.
#
# GradNorm note: norm-balancing variant (inverse-training-rate term dropped
# because PPO losses can be negative), weights initialized at (1, 1),
# lr 0.025, weight trajectory logged in the result JSON.
#
# Matrix: 2 methods x 5 seeds on Overcooked AA, 10M env steps. ~2 h total.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_R2_grad_surgery.sh 2>&1 | tee results/logs/R2_surgery.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/R2_grad_surgery

mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local method=$1
    local seed=$2
    local save="$OUT_DIR/R2_${method}_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $method seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --aux-lambda 0.05 --grad-surgery "$method" \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $method seed=$seed" || log "DONE: $method seed=$seed"
    return $rc
}

log "R2 gradient-surgery baselines STARTED"

for seed in $SEEDS; do
    run_one "pcgrad"   "$seed"
    run_one "gradnorm" "$seed"
done

log "R2 COMPLETE"
