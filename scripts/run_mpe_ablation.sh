#!/bin/bash
# MPE simple_spread ablation: 4 VABL configs × 5 seeds.
# Tests whether the gradient-interference pathology reproduces on a second
# environment family (cooperative navigation, 3 agents, 25-step episodes).
#
# Total: 20 runs × ~27 min each = ~9 hours on a 5090.
# Results in results/mpe/.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=100000   # 100K episodes × 25 steps = 2.5M env steps
HORIZON=25
SEEDS="0 1 2 3 4"
OUT_DIR=results/mpe

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/mpe_${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_mpe \
        --episodes "$N_EPISODES" \
        --horizon "$HORIZON" \
        --n-envs "$N_ENVS" \
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
log "MPE simple_spread ablation STARTED"
log "Episodes: $N_EPISODES | Horizon: $HORIZON | Seeds: $SEEDS"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_attn" "$seed" --aux-lambda 0.05 --no-attention
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "neither" "$seed" --no-aux-loss --no-attention --aux-lambda 0.0
done

log "============================================="
log "MPE simple_spread ablation COMPLETE"
log "Results in: $OUT_DIR/"
log "============================================="
