#!/bin/bash
# Lambda x stationarity interaction (2026-07-25). THE discriminating experiment.
#
# Problem this solves. The GradNorm result (learned w_aux ~ 0.95, i.e. ~19x the
# paper's lambda = 0.05, all 5 seeds degraded, worst 415) shows that a large
# auxiliary gradient is harmful. By itself that does NOT discriminate the
# paper's DIRECTIONAL account from a plain MAGNITUDE account, because scaling
# the auxiliary weight is exactly a magnitude intervention. A sharp reviewer
# (yGKw, confidence 4) will make this objection, and it is correct.
#
# The fix is a 2x2: cross the auxiliary weight with target stationarity.
#
#                    drifting targets      frozen (stationary) targets
#   lambda = 0.05    Full          (have: 467.99 +/- 2.96, n=20)
#                                        frozen (have: 468.8 +/- 2.5, n=10)
#   lambda = 0.95    NEW: hi_drift        NEW: hi_frozen
#
# Predictions:
#   Directional account (Sigma_eps = lambda^2 * J Sigma_pi J^T): the damage at
#     high lambda requires DRIFT. hi_drift should degrade badly; hi_frozen
#     should stay near baseline, because Sigma_pi ~ 0 makes Sigma_eps ~ 0
#     regardless of lambda. Prediction: large interaction.
#   Magnitude account: damage follows the auxiliary gradient norm irrespective
#     of target source. hi_drift and hi_frozen should degrade similarly.
#     Prediction: no interaction.
#
# This is a genuine falsification test of the paper's central distinction, and
# either outcome is reportable. lambda = 0.95 matches GradNorm's converged
# auxiliary weight so the comparison is weight-matched to that baseline.
#
# 10 runs, ~100 min on the RTX 5090.

set -u

REPO=$HOME/projects/VABL/vabl-multi-agent-coordination
PYTHON=$HOME/miniconda3/envs/icml2026/bin/python
OUT=$REPO/results/lambda_stationarity_2x2
HI_LAMBDA=0.95

mkdir -p "$OUT" "$REPO/results/logs"
cd "$REPO" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1 seed=$2; shift 2
    local save="$OUT/L2x2_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP $save"; return 0; fi
    log "START $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout asymmetric_advantages --episodes 25000 --horizon 400 \
        --n-envs 64 --seed "$seed" --save "$save" \
        --log-gradient-decomp --grad-log-interval 5 --log-policy-kl \
        "$@" || log "FAIL $name seed=$seed"
}

log "===== lambda x stationarity 2x2 START (hi lambda = $HI_LAMBDA) ====="

for seed in 0 1 2 3 4; do
    # High weight, co-adapting targets: magnitude AND drift both present.
    run_one "hi_drift"  "$seed" --aux-lambda "$HI_LAMBDA"
    # High weight, stationary targets: magnitude present, drift removed.
    run_one "hi_frozen" "$seed" --aux-lambda "$HI_LAMBDA" --aux-frozen-target-policy
done

echo "[$(date)] LAMBDA_2X2_COMPLETE" >> "$REPO/results/logs/rebuttal_chain.log"
log "===== lambda x stationarity 2x2 COMPLETE ====="
