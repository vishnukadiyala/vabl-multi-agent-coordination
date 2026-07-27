---
tags: [neurips2026, rebuttal, claim-audit, empirical-integrity]
status: active
related: [neurips2026_rebuttal_plan, neurips2026_rebuttal_draft]
---

# Rebuttal evidence audit (2026-07-25, after R1-R5)

All 65 chain runs + 20-run seed extension in flight. This page records what
the fresh data actually supports, before any rebuttal text is finalized.
Collapse threshold: Final50 < 460 (canonical collapsed seeds: 450.6-456.5;
fresh non-collapsed minimum: 463).

## What the fresh data says

| Condition (AA, 10M) | n | Final50 | Collapses |
|---|---|---|---|
| Full (fresh, R1+R5) | 10 | 469.1 +/- 3.0 | 0/10 |
| No-Aux (fresh) | 10 | 469.9 +/- 2.2 | 0/10 |
| Stop-grad (fresh) | 5 | 469.3 +/- 0.8 | 0/5 |
| Frozen (fresh) | 10 | 468.8 +/- 2.5 | 0/10 |
| Full (canonical Phase2 + expB) | 10 | — | 3/10 |
| PCGrad (R2) | 5 | 461.0 +/- 5.0 | 2/5 |
| GradNorm (R2, w_aux -> ~0.95) | 5 | 446.2 +/- 15.8 | 5/5 |
| Lag continuum soft (R3): lag1/25/100/frozen | 5 ea | 468.0 / 466.0 / 466.8 / 470.5 | 0,0,1,0 |
| Drift gate tau=0.10 (gated 70%) | 5 | 471.9 +/- 1.7 | 0/5 |
| Drift gate tau=0.15 (gated 30%) | 5 | 467.2 +/- 1.1 | 0/5 |

Config parity between fresh Full and canonical expB Full verified key-by-key
(only new metadata keys differ). Fresh runs perturb the main RNG stream
(grad-log every 5 vs 25), so they are new sample paths, not seed-matched
replicates. Exact seed-matched reruns of collapsed expB seed 3 (x2) queued to
distinguish seed-anchored from purely stochastic collapse.

## Scorecard vs the paper's claims

SUPPORTS the mechanism:
- GradNorm (equalizes aux gradient norm, ~19x canonical weight): all 5 seeds
  degrade, one severely (415). Aux-gradient magnitude near convergence is
  clearly harmful. Strongest new evidence.
- PCGrad does not fix it (2/5 collapse-ish): conflict-projection surgery does
  not remove the failure mode, consistent with "not a persistent-conflict
  problem".
- Drift gate tau=0.10 is the best condition overall (471.9 +/- 1.7): a
  mechanism-motivated controller works (though at 70% gating it approximates
  stop-gradient).
- Soft-target aux self-cos: frozen 0.974 vs drifting 0.79-0.86: aux gradient
  direction IS more stable for stationary targets when targets are smooth
  distributions. Direction consistent with Sigma_eps = J Sigma_pi J^T.

WEAKENS the paper's framing:
- Fresh Full runs: 0/10 collapses, mean gap to No-Aux only 0.8 points.
  Canonical: 3/10 and ~6-10 points. The AA pathology is rarer/weaker than the
  paper reports; at n=5 the variance claim does not replicate reliably.
- R3 continuum is FLAT (no monotone drift-rate effect on Final50); the single
  collapse occurred at lag100, not at max drift.
- Between-task cosine-std does not separate frozen (0.226!) from full (0.160);
  with HARD targets, aux self-cos is ~0 for BOTH full and frozen: target
  SAMPLING noise, not policy drift, dominates aux-gradient direction
  variability in the canonical setting.
- Consecutive-policy KL is ~2.5e-3 in all conditions (it measures live-policy
  drift; target drift differs by construction, but the measured quantity does
  not itself separate conditions).

## Resolution (2026-07-25 morning) — SUPERSEDED IN PART, see corrections below

1. Seed extension: fresh Full incidence is 0/20 (min 464.1). Full vs No-Aux
   at n=20: gap 1.92 points, Welch t p=0.044, Cohen's d=0.68.
2. ~~Seed-3 exact reruns: 469.5 and 468.1 vs canonical 454.6. The canonical
   collapse does NOT reproduce seed-matched. Collapse is a purely stochastic
   rare event. Fisher exact p = 0.024.~~ **RETRACTED 2026-07-25 afternoon.**
   See correction 1.

## CORRECTIONS (2026-07-25 afternoon)

**Correction 1: the "exact seed-matched rerun" was not seed-matched.**
Commit 6b37a34 (2026-04-22 01:00) changed the trainer's RNG structure: the
init split went 3-way -> 4-way and in-loop `split` calls were added. The random
stream therefore diverges from step zero between April code and current code,
so current code CANNOT reproduce any April run at a matched seed. Verified by
config fingerprint: the VAE commit added `use_vae_belief` to the recorded
config, and `expB_full_seed3.json` lacks that key, so the collapsed canonical
run came from pre-VAE code. The seed-3 "reruns" were independent sample paths.
Consequences:
  - The Fisher test pooling canonical 3/10 against fresh 0/22 treated two code
    versions as exchangeable. Withdrawn.
  - "GPU nondeterminism" was an unnecessary explanation for the earlier
    validation mismatch; a deterministic one was available.
  - Genuine fixed-seed nondeterminism under IDENTICAL code is only ~1.4 points
    (two same-day reruns: 469.5, 468.1), well under the 2.96 between-seed sd.
    The harness is reproducible; the cross-era comparison was the broken part.
  - Also found: the expB batch is internally split across code versions (seeds
    0-3 pre-VAE, seed 4 post-VAE in every condition). The paper's headline
    canonical_phase2.json is clean (all seven configs from 8e0c854).
  - A definitive replication under the ORIGINAL code (worktree at 8e0c854,
    original env) is running. Results PENDING.

**Correction 2: the effect is stronger than the old-stack runs suggested.**
The V2 environment (jax 0.10.2 vs 0.6.2, flax 0.12.8, numpy 2.4.6, py3.11,
jaxmarl pinned) gives, at n=16/arm so far:
  Full 468.37 +/- 3.38, No-Aux 472.39 +/- 1.86
  gap 4.02, bootstrap 95% CI [2.30, 5.96], Welch p = 0.0004, d = 1.47
  variance ratio Full/No-Aux = 3.30, p = 0.027
versus the old stack at n=20/arm: gap 1.92 [0.17, 3.66], p = 0.044, d = 0.66,
variance ratio 1.07 (p = 0.88).
So the deficit replicates in DIRECTION on both stacks, and the paper's actual
variance-asymmetry claim replicates on the V2 stack (3.30x) and in canonical
data (5.58x) but not in the fresh old-stack set. Magnitude is stack-sensitive.
This materially improves the defensible position relative to the morning read,
and Vishnu's instinct to re-verify on a second stack was correct.

**Correction 3: statistical presentation.** All effect claims now carry
bootstrap CIs (see VERIFIED_NUMBERS.md). The collapse threshold (<460) was
chosen after seeing the data and is descriptive only; no inferential test may
be computed on collapse counts.

**Correction 4: claims withdrawn from the rebuttal.** Drift gate is not "best
in study" (vs No-Aux: diff 1.99, p = 0.097). GradNorm is not the strongest
evidence for the directional mechanism (a magnitude-scaling intervention
producing a magnitude dose-response does not discriminate directional from
magnitude accounts; PCGrad's null is the discriminating result). Run count is
147, not "130+". No assertion about NeurIPS organizers inserting text.
3. R7 SMAX (fresh, canonical config): full 10.25+/-0.29, no_aux 10.56+/-0.56,
   stopgrad 10.36+/-0.14 (canonical gap direction reproduces, small). Late
   E[cos] = +0.006 for full, 0.000 others: NO meaningful directional bias.
   The zero-mean noise assumption HOLDS on SMAX; the residual mean gap is NOT
   explained by gradient-direction bias (answer for PYCT Q1/yGKw Q7: bias
   ruled out, secondary mechanism remains open). aux_self_cos ~ 0 on SMAX
   hard targets, same sampling-noise dominance as AA.

## What survives (replicable, defensible)

- Dose-response in effective aux weight: No-Aux/stopgrad/frozen 0 collapses
  in 70+ runs; Full (lambda=0.05) small deficit (d=0.68) + ~9% collapse rate;
  PCGrad 2/5 degraded; GradNorm (w_aux -> ~1, i.e. ~19x) 5/5 degraded, worst
  415. Clean monotone chain across effective aux-gradient influence.
- Drift-gated controller: best condition in the study (471.9 +/- 1.7).
- Soft-target self-cosine: frozen 0.974 vs drifting 0.79-0.86 (mechanism
  signature present when targets are smooth distributions).

## What does not survive

- Canonical AA effect magnitudes (10-pt gap, 8.5-vs-3.6 variance contrast,
  bimodality at n=5).
- Between-task cosine-std as the mechanism's diagnostic (frozen 0.226 vs full
  0.160: does not separate; hard-target sampling noise dominates direction
  variability).
- Monotone drift-rate dose-response (R3 continuum flat on Final50).
- SMAX mean-gap-from-bias explanation (E[cos] ~= 0).

## Framing options for the rebuttal (decision needed: Vishnu +/- Atiq)

A. Full disclosure + reframe: report all reviewer-requested results including
   the weak fresh replication; reframe the contribution around (i) collapse
   as a rare stochastic event whose incidence scales with effective aux
   weight (No-Aux 0/25 < Full ~3/30 < PCGrad 2/5 < GradNorm 5/5), and (ii)
   the drift-gate/stop-grad fixes. Honest; strongest on the weight-scaling
   axis; concedes the AA headline is overstated at n=5.
B. Present requested experiments candidly but keep the paper's framing,
   arguing pooled evidence (3/30 Full vs 0/50+ shielded, plus GradNorm
   amplification) still supports the mechanism, with the seed fragility
   stated as a limitation the camera-ready will address.
C. Withdraw/hold: if seed reruns show canonical collapses don't reproduce at
   all, the AA evidence base may be too weak to defend; discuss with Atiq.

Recommendation (Claude, 2026-07-25): between A and B pending the two in-flight
checks; the GradNorm result makes A viable and arguably stronger than the
original framing. Do NOT submit rebuttal text asserting the canonical AA
numbers without acknowledging the fresh replication set.
