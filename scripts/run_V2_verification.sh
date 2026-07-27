#!/bin/bash
# V2-environment re-verification (2026-07-25). Same code, same machine, same
# jaxmarl (0.1.0): only the numerical stack changes (jax 0.6.2 -> 0.10.2,
# flax 0.10.7 -> 0.12.8, numpy 2.2.6 -> 2.4.6, python 3.10 -> 3.11).
# Tests whether the canonical AA collapse phenomenon and the diagnostics are
# sensitive to the numerical environment.
#
# Priority order: AA main effect n=20 -> frozen x10 -> R3 continuum -> SMAX.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_V2_verification.sh 2>&1 | tee -a results/logs/V2_verification.log

set -u

PY=~/miniconda3/envs/icml2026_v2/bin/python
OUT=results/V2_verification
mkdir -p "$OUT" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

aa() {
    local save=$1; shift
    [[ -f "$save" ]] && { log "SKIP $save"; return 0; }
    log "START $save"
    "$PY" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout asymmetric_advantages --episodes 25000 --horizon 400 \
        --n-envs 64 --save "$save" \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5 "$@" \
        || log "FAIL $save"
}

smax() {
    local save=$1; shift
    [[ -f "$save" ]] && { log "SKIP $save"; return 0; }
    log "START $save"
    "$PY" -u -m marl_research.algorithms.jax.train_vabl_vec_smax \
        --episodes 50000 --horizon 100 --n-envs 64 --save "$save" \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5 "$@" \
        || log "FAIL $save"
}

log "===== V2 verification STARTED (jax $($PY -c 'import jax; print(jax.__version__)')) ====="

# Phase 1: AA main effect, n=20 per arm
for seed in $(seq 0 19); do
    aa "$OUT/V2_full_seed${seed}.json"   --seed "$seed" --aux-lambda 0.05
    aa "$OUT/V2_no_aux_seed${seed}.json" --seed "$seed" --no-aux-loss --aux-lambda 0.0
done
echo "[$(date)] V2_PHASE1_AA_COMPLETE" >> results/logs/rebuttal_chain.log

# Phase 2: frozen-target n=10 (diagnostic re-check: cosine-std frozen vs full)
for seed in $(seq 0 9); do
    aa "$OUT/V2_frozen_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 --aux-frozen-target-policy
done

# Phase 3: R3 snapshot-lag continuum re-check
for seed in 0 1 2 3 4; do
    for lag in 1 25 100 -1; do
        tag="lag${lag}"; [[ "$lag" == "-1" ]] && tag="lagfrozen"
        aa "$OUT/V2_${tag}_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 \
           --aux-snapshot-refresh "$lag" --aux-soft-targets
    done
done

# Phase 4: SMAX re-check (E[cos] + gap)
for seed in 0 1 2 3 4; do
    smax "$OUT/V2_smax_full_seed${seed}.json"     --seed "$seed" --aux-lambda 0.05
    smax "$OUT/V2_smax_no_aux_seed${seed}.json"   --seed "$seed" --no-aux-loss --aux-lambda 0.0
    smax "$OUT/V2_smax_stopgrad_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 --stop-gradient-belief
done

echo "[$(date)] V2_VERIFICATION_COMPLETE" >> results/logs/rebuttal_chain.log
log "===== V2 verification COMPLETE ====="
