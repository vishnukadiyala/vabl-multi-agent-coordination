#!/bin/bash
# Vision experiment PILOT: 4-config Group A x 1 seed only.
#
# Goal: verify the aux+attention pathology direction at 1 seed before
# committing the full 35-run matrix. ~2 hours of GPU on a 5090.
#
# If A_full's Final5 looks meaningfully lower than A_no_attn / A_no_aux /
# A_neither at this single seed, the pathology is plausibly reproducing and
# we launch the full matrix. If not, we investigate before burning more compute.

set -u

PYTHON=${PYTHON:-~/miniconda3/envs/icml2026/bin/python}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
SEED=0
OUT_DIR=vision_experiment/results

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    shift
    local save="$OUT_DIR/vision_${name}_seed${SEED}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$SEED"
    "$PYTHON" -u vision_experiment/train_vision_aux.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --seed "$SEED" \
        --save "$save" \
        "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): $name seed=$SEED"
    else
        log "DONE: $name seed=$SEED"
    fi
    return $rc
}

log "============================================="
log "Vision PILOT (4 Group A configs, seed 0 only)"
log "Epochs: $EPOCHS | Batch: $BATCH"
log "============================================="

run_one "A_full"    --aux-lambda 0.05
run_one "A_no_attn" --aux-lambda 0.05 --no-attention
run_one "A_no_aux"  --no-aux-loss --aux-lambda 0.0
run_one "A_neither" --no-aux-loss --no-attention --aux-lambda 0.0

log "============================================="
log "Vision PILOT complete. Inspect Final5 for Group A configs:"
log "  - If A_full < A_no_attn/A_no_aux/A_neither by ~5+ points: pathology reproduces"
log "  - If all 4 are within ~2 points: no pathology, investigate before scaling up"
log "============================================="
