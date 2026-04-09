#!/bin/bash
# Vision cross-domain experiment - mirrors Phase 2 minimal exactly.
#
# 7 configs x 5 seeds = 35 runs on CIFAR-100 with a small ViT + RotNet aux head.
# Each run is ~30 min on a 5090, total ~17.5 hours.
#
# Goal: demonstrate that the constant aux + attention pathology found in MARL
# also appears in a non-MARL setting, and that the same fix paths recover it.
#
# Results land in vision_experiment/results/. Each run saves its own JSON;
# existing JSONs are skipped, so re-running picks up where it left off.

set -u

PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
SEEDS=${SEEDS:-"0 1 2 3 4"}
OUT_DIR=${OUT_DIR:-vision_experiment/results}

mkdir -p "$OUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/vision_${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u vision_experiment/train_vision_aux.py \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --seed "$seed" \
        --save "$save" \
        "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "FAIL (rc=$rc): $name seed=$seed"
    else
        log "DONE: $name seed=$seed"
    fi
    return $rc
}

log "============================================="
log "Vision experiment STARTED"
log "Epochs: $EPOCHS | Batch: $BATCH | Seeds: $SEEDS"
log "============================================="

# Group A: 4-config ablation, constant lambda=0.05
log "=== Group A: 4-config ablation ==="

for seed in $SEEDS; do
    run_one "A_full"    "$seed" --aux-lambda 0.05
    run_one "A_no_attn" "$seed" --aux-lambda 0.05 --no-attention
    run_one "A_no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "A_neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

# Group B: Full encoder + 3 fix paths
log "=== Group B: Fix paths ==="

for seed in $SEEDS; do
    run_one "B_anneal"          "$seed" --aux-lambda 0.05 --aux-anneal-fraction 0.5
    run_one "B_stopgrad"        "$seed" --aux-lambda 0.05 --stop-gradient-belief
    run_one "B_anneal_stopgrad" "$seed" --aux-lambda 0.05 --aux-anneal-fraction 0.5 --stop-gradient-belief
done

log "============================================="
log "Vision experiment COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
