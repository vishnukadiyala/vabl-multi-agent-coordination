#!/bin/bash
# CIFAR-100 4-config 2x2 ablation, 5 seeds each (20 runs total).
#
# Addresses the n=1 per-config pilot reported in the paper appendix.
# Matches the MARL 2x2 (Full / No Attn / No Aux / Neither) exactly so
# cross-domain comparison is apples-to-apples.
#
# Each run: 200 epochs small ViT (4 layers, 192 dim) + RotNet aux head,
# batch 256, ~25 min on RTX 5090. Total: ~8-10h.
#
# Crash recovery: skips existing JSONs in vision_experiment/results/.

set -u
PYTHON=${PYTHON:-~/miniconda3/envs/icml2026/bin/python}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
SEEDS=${SEEDS:-"0 1 2 3 4"}
OUT_DIR=vision_experiment/results_5seed

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/vision_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP: $save"; return 0; fi
    log "START: $name seed=$seed"
    "$PYTHON" -u vision_experiment/train_vision_aux.py \
        --epochs "$EPOCHS" --batch-size "$BATCH" \
        --seed "$seed" --save "$save" "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc) $name seed=$seed" || log "DONE $name seed=$seed"
    return $rc
}

log "============================================="
log "Vision 4-config 5-seed STARTED"
log "============================================="

for seed in $SEEDS; do
    run_one "A_full"    "$seed" --aux-lambda 0.05
    run_one "A_no_attn" "$seed" --aux-lambda 0.05 --no-attention
    run_one "A_no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "A_neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

log "============================================="
log "Vision 4-config 5-seed COMPLETE"
log "============================================="
