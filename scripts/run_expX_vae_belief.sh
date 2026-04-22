#!/bin/bash
# Experiment X-A: Dynamic-Belief-style VAE belief encoder in VABL.
#
# Goal: address the "VABL is your own construction" reviewer critique by
# showing the pathology reproduces when VABL's belief encoder is swapped
# for Dynamic Belief's architectural signature (reparameterized VAE
# belief with KL-to-N(0, I) regularizer, beta=0.005 per Zhai et al.).
#
# If Full VAE is still pathological and severing the aux recovers it,
# the mechanism is not VABL-specific: it reproduces on an architecture
# whose encoder matches a published method.
#
# Matrix: 2x2 ablation with VAE belief on Overcooked AA, 5 seeds each.
#   Full (attn + aux + VAE):      expected pathological
#   No Aux (attn + VAE, no aux):  expected shielded (matches No Aux in main ablation)
#   No Attn (mean + aux + VAE):   mean pooling, expected non-pathological
#   Neither (mean + VAE, no aux): expected baseline
#
# Total: 20 runs, ~5h on RTX 5090. Crash recovery: skips existing JSONs.

set -u
PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64; N_EPISODES=25000; HORIZON=400; LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/expX_vae_belief
mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1; local seed=$2
    shift 2
    local save="$OUT_DIR/expX_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP: $save"; return 0; fi
    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --use-vae-belief --vae-kl-weight 0.005 \
        "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc) $name seed=$seed" || log "DONE $name seed=$seed"
}

log "============================================="
log "ExpX VAE-belief 2x2 STARTED"
log "============================================="

for seed in $SEEDS; do
    run_one "full"    "$seed" --aux-lambda 0.05
    run_one "no_aux"  "$seed" --no-aux-loss --aux-lambda 0.0
    run_one "no_attn" "$seed" --no-attention --aux-lambda 0.05
    run_one "neither" "$seed" --no-attention --no-aux-loss --aux-lambda 0.0
done

log "============================================="
log "ExpX COMPLETE"
log "============================================="
