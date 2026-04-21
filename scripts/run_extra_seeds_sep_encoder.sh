#!/bin/bash
# Extra 5 seeds (5-9) of separate-encoder VABL on Overcooked AA.
# Combined with seeds 0-4 from run_separate_encoder.sh this gives n=10.
#
# Rationale: at n=5 the bootstrap CIs on Cohen's d for separate vs. every
# reference cell (A_full, A_no_aux, B_stopgrad) all cross zero. The point
# estimate (Final50 ~ 468) sits between pathology (~463) and shielded
# (~473) but can't be statistically distinguished from either at n=5.
# Running to n=10 to resolve whether the partial-recovery pattern is real.
#
# Total: 5 runs, ~1h on an RTX 5090 GPU. Crash recovery: skips existing JSONs.
#
# Usage (on the training GPU server):
#   screen -S sep_enc_extra
#   cd ~/aux-loss-considered-harmful
#   bash scripts/run_extra_seeds_sep_encoder.sh 2>&1 | tee results/logs/sep_encoder_extra.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
NEW_SEEDS="5 6 7 8 9"
OUT_DIR=results/separate_encoder

mkdir -p "$OUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/${name}_seed${seed}.json"

    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi

    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" \
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
log "Separate-encoder extra seeds (5-9) STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | Seeds: $NEW_SEEDS"
log "============================================="

for seed in $NEW_SEEDS; do
    run_one "sep_encoder_full" "$seed" \
        --aux-lambda 0.05 \
        --separate-aux-encoder
done

log "============================================="
log "Separate-encoder extra seeds COMPLETE"
log "Merge with seeds 0-4 in results/separate_encoder/ for n=10 analysis."
log "============================================="
