#!/bin/bash
# R13 (2026-07-26): EMA-distilled auxiliary targets. The target-DESIGN answer
# to yGKw limitation 3 ("interventions are gradient-path engineering,
# addressing symptoms rather than the root cause"): predict soft actions of an
# EMA copy of the policy instead of the live co-learning policy. Drift is
# reduced at the target source by design; the task stays relevant because the
# EMA tracks the live policy slowly.
#
# Completes the designed target spectrum: live (drift) -> EMA (slowed) ->
# periodic snapshot (piecewise) -> frozen (zero drift).
#
# Two alphas x 5 seeds. Waits for R12 to finish; pauses the supplementary V2
# resume if it has started, restarts it afterward.

set -u
PY=$HOME/miniconda3/envs/icml2026/bin/python
REPO=$HOME/projects/VABL/vabl-multi-agent-coordination
CHAIN=$REPO/results/logs/rebuttal_chain.log
cd "$REPO" || exit 1
mkdir -p results/R13_ema results/logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] R13: $*"; }

log "waiting for R12 to complete"
while ! grep -q R12_COMPLETE "$CHAIN" 2>/dev/null; do sleep 120; done

# If the round-2 script already resumed V2, pause it for R13 (bracket
# patterns so pkill cannot match this script's own command line).
pkill -f "[r]un_V2_verification.sh" 2>/dev/null
sleep 3
pkill -f "[i]cml2026_v2/bin/python" 2>/dev/null
sleep 5
log "GPU claimed"

for seed in 0 1 2 3 4; do
    for alpha in 0.995 0.99; do
        save="results/R13_ema/R13_ema${alpha}_seed${seed}.json"
        [[ -f "$save" ]] && { log "SKIP $save"; continue; }
        log "START $save"
        "$PY" -u -m marl_research.algorithms.jax.train_vabl_vec \
            --layout asymmetric_advantages --episodes 25000 --horizon 400 \
            --n-envs 64 --seed "$seed" --save "$save" \
            --aux-lambda 0.05 --aux-ema-alpha "$alpha" --log-policy-kl \
            || log "FAIL $save"
    done
done
echo "[$(date)] R13_COMPLETE" >> "$CHAIN"

log "resuming V2 phases"
bash scripts/run_V2_verification.sh >> results/logs/V2_verification.log 2>&1
echo "[$(date)] R13_AND_V2_COMPLETE" >> "$CHAIN"
