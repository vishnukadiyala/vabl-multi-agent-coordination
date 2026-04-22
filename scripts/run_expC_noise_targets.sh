#!/bin/bash
# Experiment C: aux with UNIFORM RANDOM targets.
#
# Completes the target-source distinguishing tripod together with
# Full (teammate co-learning targets) and ExpA (frozen-policy targets):
#   - Full:   co-learning, semantic signal present  -> pathological
#   - ExpA:   stationary, semantic signal present   -> ???
#   - ExpC:   non-stationary, NO semantic signal    -> ???
#
# If ExpC pathology matches Full -> aux capacity/gradient-presence is the
# driver (counter-narrative wins). If ExpC is stable like No Aux ->
# pathology requires structured targets, not just an aux pathway.
#
# Matrix: 5 seeds on Overcooked AA, 10M env steps.
# Total: 5 runs, ~5h on an RTX 5090 GPU.

set -u
PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64; N_EPISODES=25000; HORIZON=400; LAYOUT=asymmetric_advantages
SEEDS="0 1 2 3 4"
OUT_DIR=results/expC_noise_targets
mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for seed in $SEEDS; do
    save="$OUT_DIR/expC_noise_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP: $save"; continue; fi
    log "START: noise_targets seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --aux-lambda 0.05 --aux-noise-targets
    rc=$?; [[ $rc -ne 0 ]] && log "FAIL (rc=$rc) seed=$seed" || log "DONE seed=$seed"
done
log "ExpC COMPLETE"
