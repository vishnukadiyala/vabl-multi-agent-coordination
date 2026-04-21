#!/bin/bash
# Hyperparameter sensitivity sweep for the gradient-interference pathology.
#
# Reviewer concern (self-review 2026-04-13): "The architecture sweep held
# attention_heads=4 and aux_hidden_dim=64 fixed. Is the pathology a finding
# about the design pattern, or an artifact of these specific hyperparameters?"
#
# This sweep tests the Full VABL config (attention + constant aux, lambda=0.05)
# across 3 head counts on Overcooked Asymmetric Advantages, 5 seeds each.
# heads=4 is our canonical data point and is NOT re-run here.
#
# Total: 15 runs (3 new head values x 5 seeds).
#
# Hypothesis: the late-training instability (Final50 drop, high cross-seed
# variance) should appear for all head counts in {1, 2, 4, 8}. If heads=1 or
# heads=8 reverse or eliminate the pathology, the framing shifts.
#
# Runtime: ~8 min per run at 101 ep/s scan rollout = ~2h total on the training GPU.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
SEEDS="0 1 2 3 4"
HEAD_VALUES="1 2 8"     # heads=4 already in canonical data
AUX_LAMBDA=0.05         # Full VABL setting
OUT_DIR=results/hyperparam_sweep

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local heads=$1
    local seed=$2
    local save="$OUT_DIR/full_heads${heads}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: heads=$heads seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_fast \
        --layout asymmetric_advantages \
        --episodes "$N_EPISODES" \
        --horizon "$HORIZON" \
        --n-envs "$N_ENVS" \
        --seed "$seed" \
        --aux-lambda "$AUX_LAMBDA" \
        --attention-heads "$heads" \
        --save "$save"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): heads=$heads seed=$seed"
    else
        log "DONE: heads=$heads seed=$seed"
    fi
    return $rc
}

log "=================================================================="
log "Hyperparameter sensitivity sweep: attention_heads"
log "Overcooked AA, 25000 episodes, Full VABL (attn + aux lambda=0.05)"
log "Heads: $HEAD_VALUES   Seeds: $SEEDS"
log "Total runs: 15 (3 heads x 5 seeds)"
log "=================================================================="

for heads in $HEAD_VALUES; do
    log "--- heads=$heads ---"
    for seed in $SEEDS; do
        run_one "$heads" "$seed"
    done
done

log "=================================================================="
log "Hyperparameter sweep COMPLETE"
log "=================================================================="
log "Results in $OUT_DIR/"
log ""
log "Next: analyze per-seed Final50 and compare drop-from-peak magnitude"
log "against the canonical heads=4 baseline."
