#!/bin/bash
# R4 + R5 (NeurIPS 2026 rebuttal).
#
# R4: drift-gated aux controller prototype (yGKw Q6): stop-gradient the
#     aux->encoder pathway when rolling cosine-std (window 10) exceeds tau.
#     Two taus bracket the Full-VABL cosine-std of 0.185: tau=0.10 (gates
#     often) and tau=0.15 (gates only in the high-drift regime).
# R5: seeds 5-9 for the headline Full / No-Aux / frozen-target comparisons,
#     lifting the claim-bearing contrasts from n=5 to n=10 (fXvf W2/Q6).
#     Same instrumentation as R1 so the new seeds also feed the KL analysis
#     (R5 outputs land in the R1 directory with the R1 naming scheme).
#
# Matrix: R4 = 2 taus x 5 seeds; R5 = 3 configs x 5 seeds. ~4.5 h total.
#
# Usage (on Celestia):
#   cd ~/projects/VABL/vabl-multi-agent-coordination
#   bash scripts/run_R4_R5.sh 2>&1 | tee results/logs/R4_R5.log

set -u

PYTHON=~/miniconda3/envs/icml2026/bin/python
N_ENVS=64
N_EPISODES=25000
HORIZON=400
LAYOUT=asymmetric_advantages
OUT_R4=results/R4_drift_gate
OUT_R5=results/R1_kl_instrumented

mkdir -p "$OUT_R4" "$OUT_R5" results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
    local save=$1
    shift
    if [[ -f "$save" ]]; then
        log "SKIP (exists): $save"
        return 0
    fi
    log "START: $save"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout "$LAYOUT" --episodes "$N_EPISODES" --horizon "$HORIZON" \
        --n-envs "$N_ENVS" --save "$save" "$@"
    local rc=$?
    [[ $rc -ne 0 ]] && log "FAIL (rc=$rc): $save" || log "DONE: $save"
    return $rc
}

log "R4 drift-gate + R5 extra seeds STARTED"

# R4: drift gate, seeds 0-4
for seed in 0 1 2 3 4; do
    for tau in 0.10 0.15; do
        run "$OUT_R4/R4_gate_tau${tau}_seed${seed}.json" \
            --seed "$seed" --aux-lambda 0.05 \
            --drift-gate-tau "$tau" --drift-gate-window 10 \
            --log-policy-kl
    done
done

# R5: seeds 5-9 on the headline configs, R1-instrumented
for seed in 5 6 7 8 9; do
    run "$OUT_R5/R1_full_seed${seed}.json" \
        --seed "$seed" --aux-lambda 0.05 \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5
    run "$OUT_R5/R1_no_aux_seed${seed}.json" \
        --seed "$seed" --no-aux-loss --aux-lambda 0.0 \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5
    run "$OUT_R5/R1_frozen_seed${seed}.json" \
        --seed "$seed" --aux-lambda 0.05 --aux-frozen-target-policy \
        --log-policy-kl --log-gradient-decomp --grad-log-interval 5
done

log "R4 + R5 COMPLETE"
