#!/bin/bash
# GPU orchestrator for 2026-07-25. Runs as a detached background job.
#
# Order of business (single GPU, no concurrency):
#   1. Let the in-flight V2 Phase 1 (AA n=20 both arms) finish. It is the
#      strongest current result and is nearly done.
#   2. Stop the V2 chain so Phases 2-4 do not start yet.
#   3. Run the pre-VAE replication (the reproducibility control).
#   4. Restart the V2 chain, which skips completed files and continues with
#      Phase 2 (frozen n=10, the cosine-diagnostic recheck), then 3 and 4.
#
# Written as a file (not an inline ssh command) so that pkill patterns cannot
# match this script's own command line, which broke two earlier attempts.

set -u

REPO=$HOME/projects/VABL/vabl-multi-agent-coordination
CHAIN_LOG=$REPO/results/logs/rebuttal_chain.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ORCH: $*"; }

log "waiting for V2 Phase 1 to complete"
while ! grep -q V2_PHASE1_AA_COMPLETE "$CHAIN_LOG" 2>/dev/null; do
    sleep 60
done
log "V2 Phase 1 complete"

# Stop the V2 driver and any V2 training child, then wait for the GPU to clear.
pkill -f 'bash scripts/run_V2_verification.sh' 2>/dev/null
sleep 5
pkill -f 'icml2026_v2/bin/python' 2>/dev/null
sleep 10
log "V2 chain paused"

log "starting pre-VAE replication"
bash "$REPO/scripts/run_prevae_replication.sh" >> "$REPO/results/logs/prevae_replication.log" 2>&1
log "pre-VAE replication finished"

log "starting lambda x stationarity 2x2 (discriminating experiment)"
bash "$REPO/scripts/run_lambda_stationarity_2x2.sh" >> "$REPO/results/logs/lambda_2x2.log" 2>&1
log "lambda x stationarity 2x2 finished"

log "resuming V2 chain (phases 2-4)"
cd "$REPO" || exit 1
bash scripts/run_V2_verification.sh >> "$REPO/results/logs/V2_verification.log" 2>&1
log "V2 chain finished"

echo "[$(date)] ORCHESTRATION_COMPLETE" >> "$CHAIN_LOG"
