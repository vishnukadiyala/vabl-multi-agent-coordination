#!/bin/bash
# Experiment E: SMAX intermediate-lambda diagnostic.
#
# Current puzzle: on SMAX, the three fix paths reduce cross-seed variance
# but do NOT recover the mean (Full=10.60, No Aux=11.15, fixes=10.54-10.63).
# Two hypotheses:
#   (a) aux is actively HARMFUL on SMAX (mechanism different from AA)
#   (b) aux provides USEFUL regularization on SMAX that fix paths destroy
#
# Test: run Full at half-lambda (0.025). If mean drops below lambda=0.05 ->
# evidence for (b): the aux has an active benefit that gets lost when
# weakened. If mean stays near 0.05 or rises -> evidence for (a): aux is
# just harmful and reducing lambda reduces harm.
#
# Matrix: 1 lambda value (0.025) x 5 seeds on SMAX 3v3.
# Total: 5 runs, ~5h on an RTX 5090 GPU.

set -u
PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64; N_EPISODES=50000; HORIZON=100
SEEDS="0 1 2 3 4"
OUT_DIR=results/expE_smax_intermediate_lambda
mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for seed in $SEEDS; do
    save="$OUT_DIR/expE_smax_lam0.025_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP: $save"; continue; fi
    log "START: smax lam=0.025 seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_smax \
        --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --seed "$seed" --save "$save" \
        --aux-lambda 0.025
    rc=$?; [[ $rc -ne 0 ]] && log "FAIL (rc=$rc) seed=$seed" || log "DONE seed=$seed"
done
log "ExpE COMPLETE"
