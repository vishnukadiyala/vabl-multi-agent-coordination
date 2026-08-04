#!/bin/bash
# R17 (yGKw final comment, defense round): J_pi on MPE simple_spread.
#
# R15 falsified "slow symmetric drift" (MPE Sigma_pi is 2.2-2.7x AA's). The
# proposition's noise term Sigma_eps = J_pi Sigma_pi J_pi^T has two other
# factors; this run measures the pathway-sensitivity factor with the exact
# protocol used on AA (target flips at eps in {0.05, 0.1, 0.2}, relative
# aux-gradient response; AA Full reference: 2.47 at eps=0.1) plus the
# aux/policy gradient-norm ratio.
#
# If MPE's J_pi response and/or norm ratio is much smaller than AA's, the
# MPE null is explained WITHIN the theory (high drift x low sensitivity =>
# small Sigma_eps) and the reply defends the proposition while withdrawing
# only the informal intro attribution.
#
# Matrix: full arm x 5 seeds, exact R15 settings + instrumentation.
#
# Usage:
#   bash scripts/run_R17_mpe_jpi.sh 2>&1 | tee results/logs/R17_mpe_jpi.log

set -u

PYTHON=${PYTHON:-python}
N_ENVS=64
N_EPISODES=100000
SEEDS="0 1 2 3 4"
OUT_DIR=results/R17_mpe_jpi

mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

for seed in $SEEDS; do
    save="$OUT_DIR/R17_full_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        continue
    fi
    log "START: full seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_mpe \
        --episodes "$N_EPISODES" --n-envs "$N_ENVS" --seed "$seed" \
        --log-interval 200 --aux-lambda 0.05 \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5 --log-jpi \
        --save "$save"
    rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): full seed=$seed" || log "DONE: full seed=$seed"
done

log "R17 COMPLETE"
