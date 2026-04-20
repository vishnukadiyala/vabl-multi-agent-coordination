#!/bin/bash
# Separate-encoder VABL control (intra-actor).
# Goal: test whether the pathology is specifically due to the SHARED encoder.
#
# The aux predictor uses its own parallel feature encoders (phi_aux, psi_aux)
# and a projection to a separate belief-dim representation. The policy's
# belief encoder is entirely shielded from auxiliary gradients.
#
# If the pathology DISAPPEARS in this configuration, the mechanism is
# confirmed as "aux gradients flowing into the shared encoder". If the
# pathology PERSISTS, the finding refines to attention+aux-anywhere.
#
# Matrix: Full (attn+aux with separate aux encoder) x 5 seeds on Overcooked AA.
# Total: 5 runs, ~1h on Celestia 5090.
#
# Crash recovery: skips existing JSONs.
#
# Usage (on Celestia):
#   screen -S sep_encoder
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_separate_encoder.sh 2>&1 | tee results/separate_encoder.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
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
log "Separate-encoder VABL (intra-actor control) STARTED"
log "Layout: $LAYOUT | Episodes: $N_EPISODES | Seeds: $SEEDS"
log "============================================="

for seed in $SEEDS; do
    # Full config (attn + aux constant lambda=0.05) with separate aux encoder.
    # This is the pathological config with the shared-encoder pathway removed.
    run_one "sep_encoder_full" "$seed" \
        --aux-lambda 0.05 \
        --separate-aux-encoder
done

log "============================================="
log "Separate-encoder control COMPLETE"
log "Compare Final50 against A_full (pathological) and A_no_aux (shielded)"
log "in results/phase2/. If Final50 ~ 473, the shared-encoder hypothesis is"
log "confirmed. If Final50 ~ 463, the pathology survives this intervention."
log "============================================="
