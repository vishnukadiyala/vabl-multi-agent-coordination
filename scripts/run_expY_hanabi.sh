#!/bin/bash
# Experiment Y: VABL on Hanabi 2-player — 4th MARL environment.
#
# Addresses the "too narrow" critique by testing whether the pathology
# reproduces on a canonical turn-based, partial-information coordination
# benchmark beyond Overcooked/SMAX/MPE. Hanabi is widely regarded as a
# defining test for MARL belief/prediction methods.
#
# Matrix: 2x2 ablation (attn x aux) with 5 seeds on JaxMARL Hanabi-2p.
#   Full    (attn + aux, lambda=0.05):   expected pathological if claim generalizes
#   No Aux  (attn only):                  healthy reference
#   No Attn (mean + aux):                 isolates attention's role
#   Neither (mean only):                  baseline
#
# Total: 20 runs, ~5-8h on RTX 5090 (Hanabi's larger obs + obs_dim makes
# it somewhat slower per episode than Overcooked).
#
# Crash recovery: skips existing JSONs.

set -u
PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=32   # smaller than Overcooked due to Hanabi's 658-dim obs
N_EPISODES=25000
HORIZON=80
SEEDS="0 1 2 3 4"
OUT_DIR=results/expY_hanabi
mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1; local seed=$2
    shift 2
    local save="$OUT_DIR/expY_hanabi_${name}_seed${seed}.json"
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
log "ExpY Hanabi 2x2 STARTED"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "no_attn" "$seed" --no-attention --aux-lambda 0.05
    run_one "neither" "$seed" --no-attention --no-aux-loss --aux-lambda 0.0
done

log "============================================="
log "ExpY Hanabi COMPLETE"
log "============================================="
