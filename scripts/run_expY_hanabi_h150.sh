#!/bin/bash
# Experiment Y (re-run): VABL on Hanabi with horizon=150 for higher absolute scores.
#
# First pass at horizon=80 confirmed the pathology pattern clearly in
# relative terms (Aux-ON 0.14-0.26 vs Aux-OFF 1.84-2.81, Cohen's d=+2.81
# for No-Aux vs Full) but with low absolute scores due to games getting
# truncated. Horizon=150 should let full games complete, giving scores
# in the 5-15 range expected for this benchmark.
#
# Same 2x2 matrix; separate output directory so both passes are kept.
# Total: 20 runs, ~8h on RTX 5090. Crash recovery: skips existing JSONs.

set -u
PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=32
N_EPISODES=25000
HORIZON=150
SEEDS="0 1 2 3 4"
OUT_DIR=results/expY_hanabi_h150
mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1; local seed=$2
    shift 2
    local save="$OUT_DIR/expY_hanabi150_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP: $save"; return 0; fi
    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_hanabi \
        --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc) $name seed=$seed" || log "DONE $name seed=$seed"
}

log "============================================="
log "ExpY Hanabi (horizon=150) STARTED"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "no_attn" "$seed" --no-attention --aux-lambda 0.05
    run_one "neither" "$seed" --no-attention --no-aux-loss --aux-lambda 0.0
done

log "============================================="
log "ExpY Hanabi h150 COMPLETE"
log "============================================="
