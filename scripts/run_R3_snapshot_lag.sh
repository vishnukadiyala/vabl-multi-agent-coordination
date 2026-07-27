#!/bin/bash
# R3 (NeurIPS 2026 rebuttal): snapshot-lag continuum with soft targets.
#
# Isolates temporal drift from label entropy / predictability / state
# dependence (fXvf W1/Q5, yGKw W4/Q2): every condition uses the aux target
# policy's full action DISTRIBUTION (soft targets) evaluated on the same
# rollout inputs; the only difference is how often the target-policy snapshot
# is refreshed:
#   refresh=1   -> refreshed every iteration (co-learning endpoint, max drift)
#   refresh=25  -> intermediate drift
#   refresh=100 -> slow drift
#   refresh=-1  -> never refreshed (frozen endpoint, zero drift)
#
# Predicted (paper's mechanism): late-training instability increases
# monotonically with drift rate; the frozen-soft endpoint matches No-Aux-level
# stability. A non-monotone or flat ordering falsifies the identification.
#
# Matrix: 4 lags x 5 seeds on Overcooked AA, 10M env steps. ~3.5 h total.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_R3_snapshot_lag.sh 2>&1 | tee results/logs/R3_lag.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
LAGS="1 25 100 -1"
OUT_DIR=results/R3_snapshot_lag

mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local lag=$1
    local seed=$2
    local tag="lag${lag}"
    [[ "$lag" == "-1" ]] && tag="lagfrozen"
    local save="$OUT_DIR/R3_${tag}_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $tag seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --aux-lambda 0.05 \
        --aux-snapshot-refresh "$lag" --aux-soft-targets \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $tag seed=$seed" || log "DONE: $tag seed=$seed"
    return $rc
}

log "R3 snapshot-lag continuum STARTED"

for seed in $SEEDS; do
    for lag in $LAGS; do
        run_one "$lag" "$seed"
    done
done

log "R3 COMPLETE"
