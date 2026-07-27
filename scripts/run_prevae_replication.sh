#!/bin/bash
# Pre-VAE replication (2026-07-25). THE reproducibility control.
#
# The paper's headline Phase-2 table (canonical_phase2.json) records
# code_commit 8e0c854. Commit 6b37a34 (2026-04-22) later changed the trainer's
# RNG structure (init split 3-way -> 4-way, plus in-loop vae splits), so the
# CURRENT trainer cannot reproduce any April run even at a matched seed: the
# random stream diverges from step zero. Every "seed-matched rerun" done on
# 2026-07-24/25 with current code was therefore an independent sample path,
# not a replicate.
#
# This script runs the ORIGINAL code (git worktree at 8e0c854) in the ORIGINAL
# env (icml2026, jax 0.6.2) with the canonical Phase-2 flags, so seeds mean
# what they meant in April.
#
# Decisive predictions:
#   - If A_full seeds 1 and 2 return ~456 and ~451 (canonical collapses):
#     canonical numbers are real, deterministic given (code, seed); the
#     "collapse is a rare stochastic event" reading is WRONG and the paper's
#     per-seed data stands.
#   - If they return ~468: canonical per-seed outcomes are not reproducible
#     even under their own code, which is a far more serious finding.
#
# 10 runs, ~100 min on the RTX 5090.

set -u

WORKTREE=${WORKTREE:-$HOME/projects/VABL/prevae_worktree}
PYTHON=$HOME/miniconda3/envs/icml2026/bin/python
OUT=${OUT:-$HOME/projects/VABL/vabl-multi-agent-coordination/results/prevae_replication}

mkdir -p "$OUT"
cd "$WORKTREE" || exit 1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
    local name=$1 seed=$2; shift 2
    local save="$OUT/prevae_${name}_seed${seed}.json"
    if [[ -f "$save" ]]; then log "SKIP $save"; return 0; fi
    log "START $name seed=$seed"
    "$PYTHON" -u -m marl_research.algorithms.jax.train_vabl_vec \
        --layout asymmetric_advantages --episodes 25000 --horizon 400 \
        --n-envs 64 --seed "$seed" --save "$save" "$@" || log "FAIL $name seed=$seed"
}

log "===== PRE-VAE REPLICATION START (worktree $(git rev-parse --short HEAD)) ====="

# Canonical Phase-2 seeds first: these are the ones with known targets.
#   A_full  canonical per-seed: 469.1 456.5 450.6 465.6 474.2
#   A_no_aux canonical per-seed: 475.9 475.1 477.8 469.8 468.6
for seed in 0 1 2 3 4; do
    run_one "A_full"   "$seed" --aux-lambda 0.05
    run_one "A_no_aux" "$seed" --no-aux-loss --aux-lambda 0.0
done

echo "[$(date)] PREVAE_REPLICATION_COMPLETE" \
    >> "$HOME/projects/VABL/vabl-multi-agent-coordination/results/logs/rebuttal_chain.log"
log "===== PRE-VAE REPLICATION COMPLETE ====="
