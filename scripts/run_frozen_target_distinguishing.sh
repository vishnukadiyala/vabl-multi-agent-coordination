#!/bin/bash
# Distinguishing experiment: directional interference vs. aux-capacity cost.
#
# Hypothesis under test: The pathology is driven by co-learning non-stationarity
# in aux targets (paper's story), not by the aux network consuming encoder
# capacity (counter-narrative).
#
# Design:
#   Full VABL (attn + aux, lambda=0.05) with aux TARGETS replaced by a FROZEN
#   snapshot of the initial agent policy. Aux capacity, architecture, and the
#   aux-to-encoder gradient pathway are IDENTICAL to Full. The only change is
#   that aux targets do not drift with teammate co-learning.
#
# Predictions:
#   - Paper's story (directional interference via Σε from co-learning):
#       pathology disappears → Final50 std returns to ~3-4 range (matching
#       A_no_aux = 3.62) because Σε → 0 for stationary targets.
#   - Counter (aux capacity consumption):
#       pathology persists → Final50 std stays elevated (~6-8) because the
#       aux network still consumes capacity regardless of target source.
#
# Matrix: 5 seeds on Overcooked Asymmetric Advantages, 10M env steps each.
# Compare directly against:
#   - canonical Full (Final50 463.21 ± 8.55, pathological)
#   - canonical No Aux (Final50 473.45 ± 3.62, shielded)
#
# Total: 5 runs, ~5h on an RTX 5090 GPU. Crash recovery: skips existing JSONs.
#
# Usage (on the training GPU server):
#   screen -S frozen_tgt
#   cd ~/aux-loss-considered-harmful
#   bash scripts/run_frozen_target_distinguishing.sh 2>&1 | tee results/logs/frozen_tgt.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/frozen_target_distinguishing

mkdir -p "$OUT_DIR"
mkdir -p results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local seed=$1
    local save="$OUT_DIR/frozen_tgt_full_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: frozen_tgt_full seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" \
        --episodes "$N_EPISODES" \
        --horizon "$HORIZON" \
        --n-envs "$N_ENVS" \
        --seed "$seed" \
        --save "$save" \
        --aux-lambda 0.05 \
        --aux-frozen-target-policy
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): frozen_tgt_full seed=$seed"
    else
        log "DONE: frozen_tgt_full seed=$seed"
    fi
    return $rc
}

log "============================================="
log "Frozen-target distinguishing experiment STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | Seeds: $SEEDS"
log "Purpose: distinguish directional-interference from aux-capacity-cost"
log "============================================="

for seed in $SEEDS; do
    run_one "$seed"
done

log "============================================="
log "Frozen-target distinguishing experiment COMPLETE"
log ""
log "Analysis: compare Final50 mean and std to the two references."
log "  - If std ~ 3-4 (near A_no_aux): directional-interference story confirmed."
log "  - If std ~ 6-8 (near A_full):   aux-capacity counter-narrative wins."
log "============================================="
