#!/bin/bash
# Architecture Sweep: Which components cause the gradient-interference pathology?
#
# Tests 6 architecture variants on Overcooked Asymmetric Advantages,
# each with and without aux loss, 5 seeds = 60 runs.
#
# Architectures:
#   1. lstm+cross_attn    — LSTM instead of GRU (is recurrence type the factor?)
#   2. gru+self_attn      — self-attention instead of cross-attention
#   3. gru+additive       — Bahdanau additive attention
#   4. gru+mean_pool      — no learned attention, just mean pooling
#   5. none+cross_attn    — no recurrence (is recurrence required for pathology?)
#   6. maac_critic_aux    — aux on critic not actor (does it generalize?)
#
# Expected: pathology should appear whenever constant-weight aux + shared params
# + co-learning non-stationarity are present, regardless of specific components.

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
SEEDS="0 1 2 3 4"
OUT_DIR=results/arch_sweep

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
    "$PYTHON" -u -m marl_research.algorithms.jax.train_configurable_vec \
        --layout asymmetric_advantages \
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
log "Architecture Sweep on Overcooked AA"
log "============================================="

# ---- 1. LSTM + cross_attn (with/without aux) ----
log "--- LSTM + cross_attn ---"
for seed in $SEEDS; do
    run_one "lstm_cross_aux"    "$seed" --recurrence lstm --attention cross_attn --aux-lambda 0.05
    run_one "lstm_cross_no_aux" "$seed" --recurrence lstm --attention cross_attn --no-aux-loss --aux-lambda 0.0
done

# ---- 2. GRU + self_attn (with/without aux) ----
log "--- GRU + self_attn ---"
for seed in $SEEDS; do
    run_one "gru_self_aux"    "$seed" --recurrence gru --attention self_attn --aux-lambda 0.05
    run_one "gru_self_no_aux" "$seed" --recurrence gru --attention self_attn --no-aux-loss --aux-lambda 0.0
done

# ---- 3. GRU + additive (with/without aux) ----
log "--- GRU + additive ---"
for seed in $SEEDS; do
    run_one "gru_add_aux"    "$seed" --recurrence gru --attention additive --aux-lambda 0.05
    run_one "gru_add_no_aux" "$seed" --recurrence gru --attention additive --no-aux-loss --aux-lambda 0.0
done

# ---- 4. GRU + mean_pool (with/without aux) ----
log "--- GRU + mean_pool ---"
for seed in $SEEDS; do
    run_one "gru_pool_aux"    "$seed" --recurrence gru --attention mean_pool --aux-lambda 0.05
    run_one "gru_pool_no_aux" "$seed" --recurrence gru --attention mean_pool --no-aux-loss --aux-lambda 0.0
done

# ---- 5. none + cross_attn (with/without aux) ----
log "--- FF (no recurrence) + cross_attn ---"
for seed in $SEEDS; do
    run_one "ff_cross_aux"    "$seed" --recurrence none --attention cross_attn --aux-lambda 0.05
    run_one "ff_cross_no_aux" "$seed" --recurrence none --attention cross_attn --no-aux-loss --aux-lambda 0.0
done

# ---- 6. MAAC-style: critic aux (simple actor, aux on critic) ----
log "--- MAAC-style (critic aux) ---"
for seed in $SEEDS; do
    # critic-aux with GRU+cross_attn actor (VABL-like actor, aux on critic)
    run_one "maac_critic_aux"    "$seed" --recurrence gru --attention cross_attn --critic-aux --aux-lambda 0.05
    # control: same actor, no aux anywhere
    run_one "maac_no_aux"        "$seed" --recurrence gru --attention cross_attn --no-aux-loss --aux-lambda 0.0
done

log "============================================="
log "Architecture Sweep COMPLETE"
log "============================================="
log "Results in $OUT_DIR/"
log "Total runs: 60 (6 architectures x 2 aux x 5 seeds)"
