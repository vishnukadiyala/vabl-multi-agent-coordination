#!/bin/bash
# Rebuttal round 2 (2026-07-26): everything the reviewers asked that was not
# yet run, in reviewer-value priority order.
#   R8:  J_pi finite-difference + belief rank, AA (Full, stopgrad, No-Aux x5)
#        -> yGKw Q3 (J_pi measured, pathway contrast) + PYCT Q1 in the primary env
#   R9:  aux-task variants (latent = drifting, recon = stationary) x5
#        -> yGKw Q4 as a mini replication of the central contrast
#   R10: random-gate control p=0.70 x5 -> closes yGKw Q6 (duty-cycle vs trigger)
#   R11: schedules cosine / exp / kl_adaptive x5 -> yGKw Q8
#   R12: SMAX with belief-rank logging (Full, No-Aux x5) -> PYCT Q1 where asked
# Then resumes the paused V2 phases 2-4 (supplementary).
#
# ~60 runs, ~10 h. Crash recovery: skips existing JSONs.

set -u
PY=$HOME/miniconda3/envs/icml2026/bin/python
REPO=$HOME/projects/VABL/vabl-multi-agent-coordination
cd "$REPO" || exit 1
mkdir -p results/R8_jpi_rank results/R9_auxdef results/R10_randgate \
         results/R11_schedules results/R12_smax_rank results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

aa() {
    local save=$1; shift
    [[ -f "$save" ]] && { log "SKIP $save"; return 0; }
    log "START $save"
    "$PY" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout asymmetric_advantages --episodes 25000 --horizon 400 \
        --n-envs 64 --save "$save" "$@" || log "FAIL $save"
}

smax() {
    local save=$1; shift
    [[ -f "$save" ]] && { log "SKIP $save"; return 0; }
    log "START $save"
    "$PY" -u -m marl_research.algorithms.jax.train_vabl_vec_smax \
        --episodes 50000 --horizon 100 --n-envs 64 --save "$save" \
        --log-gradient-decomp --grad-log-interval 5 --log-policy-kl "$@" \
        || log "FAIL $save"
}

log "===== ROUND 2 STARTED ====="

# R8: J_pi + rank (Full and stopgrad get jpi; No-Aux gets decomp+rank only)
for seed in 0 1 2 3 4; do
    aa "results/R8_jpi_rank/R8_full_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 \
        --log-gradient-decomp --grad-log-interval 5 --log-policy-kl --log-jpi
    aa "results/R8_jpi_rank/R8_stopgrad_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 \
        --stop-gradient-belief --log-gradient-decomp --grad-log-interval 5 --log-policy-kl --log-jpi
    aa "results/R8_jpi_rank/R8_no_aux_seed${seed}.json" --seed "$seed" --no-aux-loss --aux-lambda 0.0 \
        --log-gradient-decomp --grad-log-interval 5 --log-policy-kl
done
echo "[$(date)] R8_COMPLETE" >> results/logs/rebuttal_chain.log

# R9: aux-task variants
for seed in 0 1 2 3 4; do
    aa "results/R9_auxdef/R9_latent_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 \
        --aux-task latent --log-policy-kl
    aa "results/R9_auxdef/R9_recon_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 \
        --aux-task recon --log-policy-kl
done
echo "[$(date)] R9_COMPLETE" >> results/logs/rebuttal_chain.log

# R10: random-gate control at the tau=0.10 duty cycle
for seed in 0 1 2 3 4; do
    aa "results/R10_randgate/R10_rand070_seed${seed}.json" --seed "$seed" --aux-lambda 0.05 \
        --drift-gate-random 0.70 --log-policy-kl
done
echo "[$(date)] R10_COMPLETE" >> results/logs/rebuttal_chain.log

# R11: schedules
for seed in 0 1 2 3 4; do
    for sched in cosine exp kl_adaptive; do
        aa "results/R11_schedules/R11_${sched}_seed${seed}.json" --seed "$seed" \
            --aux-lambda 0.05 --aux-schedule "$sched" --log-policy-kl
    done
done
echo "[$(date)] R11_COMPLETE" >> results/logs/rebuttal_chain.log

# R12: SMAX rank
for seed in 0 1 2 3 4; do
    smax "results/R12_smax_rank/R12_full_seed${seed}.json" --seed "$seed" --aux-lambda 0.05
    smax "results/R12_smax_rank/R12_no_aux_seed${seed}.json" --seed "$seed" --no-aux-loss --aux-lambda 0.0
done
echo "[$(date)] R12_COMPLETE" >> results/logs/rebuttal_chain.log

log "===== ROUND 2 COMPLETE; resuming V2 phases 2-4 ====="
bash scripts/run_V2_verification.sh >> results/logs/V2_verification.log 2>&1
echo "[$(date)] ROUND2_AND_V2_COMPLETE" >> results/logs/rebuttal_chain.log
