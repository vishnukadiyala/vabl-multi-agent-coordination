#!/bin/bash
# R16 (yGKw final comment, 2026-08-02): representation-drift probe.
#
# The reviewer's first objection: fixed aux labels do not make the aux
# gradient stationary because the shared encoder keeps drifting, and the
# theory never puts representation drift in the causal chain. This run
# measures that term directly: encoder representation (new_belief, the aux
# head's input) drift on a FIXED probe batch, logged at the gradient-decomp
# cadence, in all three arms.
#
# Predicted decomposition (what the reply claims): feature drift is
# common-mode (similar in Full / frozen / No-Aux) while aux-gradient
# self-cosine separates frozen from Full. If feature drift instead differs
# by arm, the common-mode argument fails and we report that.
#
# Matrix: 3 configs x 5 seeds on Overcooked AA, 10M env steps.
# ~10 min/run on the RTX 5090 -> ~2.5 h total. Crash recovery: skips existing.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   git pull   # needs the log_feature_drift instrumentation
#   bash scripts/run_R16_repdrift.sh 2>&1 | tee results/logs/R16_repdrift.log

set -u

PYTHON=${PYTHON:-~/miniconda3/envs/icml2026/bin/python}
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/R16_repdrift
GRAD_LOG_INTERVAL=5

mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/R16_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --log-feature-drift --log-policy-kl \
        --log-gradient-decomp --grad-log-interval "$GRAD_LOG_INTERVAL" \
        "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $name seed=$seed" || log "DONE: $name seed=$seed"
    return $rc
}

log "R16 representation-drift runs STARTED"

for seed in $SEEDS; do
    run_one "full"   "$seed" --aux-lambda 0.05
    run_one "frozen" "$seed" --aux-lambda 0.05 --aux-frozen-target-policy
    run_one "no_aux" "$seed" --no-aux-loss --aux-lambda 0.0
done

log "R16 COMPLETE"
