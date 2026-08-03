#!/bin/bash
# R15 (yGKw final comment, 2026-08-02): measure Sigma_pi on MPE simple_spread.
#
# The reviewer's second objection: the paper's MPE boundary claim ("symmetric
# slow drift => no pathology") attributes the null to small Sigma_pi without
# ever measuring it there. This run adds the consecutive-policy KL
# measurement (ported from train_vabl_vec.py into train_vabl_vec_mpe.py) to
# the exact original MPE ablation settings (100K episodes, 64 envs, horizon
# 25, VABL v2, lambda=0.05).
#
# Arms: full (attn+aux, the arm whose live policy IS the aux target) and
# no_aux (control: shows drift level absent any aux pathway). 5 seeds each.
# Compare late-training KL to the Overcooked AA reference 2.42-2.58e-3 nats
# (R1). If MPE late KL is NOT small, the small-Sigma_pi attribution is
# falsified and the MPE null gets reported as an open boundary case.
#
# Usage (local CPU or Celestia):
#   cd <repo root>
#   bash scripts/run_R15_mpe_kl.sh 2>&1 | tee results/logs/R15_mpe_kl.log

set -u

PYTHON=${PYTHON:-python}
N_ENVS=64
N_EPISODES=100000
SEEDS="0 1 2 3 4"
OUT_DIR=results/R15_mpe_kl

mkdir -p "$OUT_DIR" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1
    local seed=$2
    shift 2
    local save="$OUT_DIR/R15_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec_mpe \
        --episodes "$N_EPISODES" --n-envs "$N_ENVS" --seed "$seed" \
        --log-interval 200 --log-policy-kl --save "$save" "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $name seed=$seed" || log "DONE: $name seed=$seed"
    return $rc
}

log "R15 MPE Sigma_pi runs STARTED"

for seed in $SEEDS; do
    run_one "full"   "$seed" --aux-lambda 0.05
done
for seed in $SEEDS; do
    run_one "no_aux" "$seed" --no-aux-loss --aux-lambda 0.0
done

log "R15 COMPLETE"
