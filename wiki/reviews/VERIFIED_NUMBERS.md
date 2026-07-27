---
tags: [neurips2026, rebuttal, verified-data]
status: active
related: [neurips2026_rebuttal_draft, rebuttal_evidence_audit]
---

# VERIFIED NUMBERS — single source of truth for the rebuttal

Every number here was computed from result JSONs on 2026-07-25 and re-checked.
**No number may enter the rebuttal unless it appears on this page.**
Numbers marked PENDING are not yet available and must not be asserted.

Threshold note: "collapse" = Final50 < 460. This threshold was chosen AFTER
seeing the data (canonical collapses 450-457, fresh minimum 463). It is
therefore DESCRIPTIVE ONLY. Do not compute or quote inferential p-values on
collapse counts.

Final50 = mean reward over the last 3200 episodes (50 iterations x 64 envs).

---

## 1. The headline comparison, three data generations

| Set | Code | Stack | n/arm | Full | No-Aux | Gap [95% CI] | p | d | Var ratio (F/NA) |
|---|---|---|---|---|---|---|---|---|---|
| Canonical (paper) | 8e0c854 | jax 0.6.2 | 5 | 463.21 ± 8.55 | 473.45 ± 3.62 | 10.24 | — | — | 5.58 |
| Fresh old-stack | current | jax 0.6.2 | 20 | 467.99 ± 2.96 | 469.91 ± 2.86 | 1.92 [0.17, 3.66] | 0.044 | 0.66 | 1.07 (p=0.88) |
| Fresh V2-stack | current | jax 0.10.2 | **20 FINAL** | 468.14 ± 3.49 | 471.62 ± 2.54 | 3.47 [1.68, 5.37] | 0.00098 | 1.14 | 1.89 (p=0.175) |

V2 is COMPLETE at n=20 (2026-07-25 evening). CIs are 20k-sample bootstrap; p is
Welch; d is pooled-sd. Collapses (<460): V2 Full 1/20, No-Aux 0/20.

**INTERIM FIGURES NOW SUPERSEDED. DO NOT USE.** The n=16 V2 snapshot read
gap 4.02 [2.30, 5.96], p=0.0004, d=1.47, var ratio 3.30 (p=0.027). At the final
n=20 the mean gap holds (3.47, p=0.001, d=1.14) but **the variance ratio falls to
1.89 and is no longer significant (p=0.175)**.

**Consequence: the paper's variance-asymmetry claim does NOT replicate at n=20 on
EITHER stack** (old 1.07 p=0.88; V2 1.89 p=0.175). It appears only in the n=5
canonical set (5.58) and in the underpowered n=16 V2 snapshot. Do not claim it.
What replicates is the MEAN deficit, on both stacks.

Canonical per-seed Final50:
- A_full: 469.1, 456.5, 450.6, 465.6, 474.2
- A_no_aux: 475.9, 475.1, 477.8, 469.8, 468.6

Collapse counts (<460, descriptive): canonical Full 2/5; fresh old-stack Full
0/20, No-Aux 0/20; V2 Full 1/16, No-Aux 0/16.

**Key reading.** The Full-vs-No-Aux deficit replicates in DIRECTION on both
numerical stacks. Its MAGNITUDE is stack-sensitive (1.9 vs 4.0 points). The
variance asymmetry the paper claims (Full more variable than No-Aux)
replicates on the V2 stack (3.30x, p=0.027) and on canonical (5.58x) but NOT
on the fresh old-stack set (1.07x, p=0.88).

## 2. Reproducibility of the April runs (critical caveat)

- Commit 6b37a34 (2026-04-22 01:00) changed the trainer's RNG structure:
  init split 3-way -> 4-way, plus added in-loop `split` calls. The random
  stream therefore diverges from step zero between April code and current code.
- **Current code cannot reproduce any April run, even at a matched seed.**
  Any "seed-matched rerun" performed with current code is an independent
  sample path, not a replicate.
- Verified by config fingerprint (the VAE commit added `use_vae_belief` to the
  recorded config): `expB_full_seed3.json` has NO vae keys -> pre-VAE code.
- The expB batch is internally SPLIT across code versions: seeds 0-3 of every
  condition are pre-VAE; seed 4 of every condition is post-VAE.
- The paper's headline `canonical_phase2.json` is internally CLEAN: all seven
  configs from commit 8e0c854.
- Genuine fixed-seed nondeterminism under IDENTICAL code: two same-day reruns
  of seed 3 gave 469.5 and 468.1, a spread of 1.4 points, which is small
  relative to the 2.96 between-seed sd. The harness is reproducible; the
  cross-era comparison is what was broken.
- **RETRACTED:** the earlier claim "canonical collapse does not reproduce
  seed-matched, therefore collapse is purely stochastic (Fisher p=0.024)".
  Its premise was false. Do not use it.

## 0. THE STABLE FINDING (read this first; computed 2026-07-25 night, all sets final)

The Full-vs-No-Aux mean deficit is CONSISTENT across every replication set and
every code/stack configuration. The canonical n=5 magnitude is the outlier:

| Set | Code | Stack | n/arm | Gap (No-Aux minus Full) |
|---|---|---|---|---|
| Fresh old-stack | current | jax 0.6.2 | 20 | 1.92 [0.17, 3.66], p=0.044 |
| Fresh V2-stack | current | jax 0.10.2 | 20 | 3.47 [1.68, 5.37], p=0.00098 |
| Pre-VAE replication | 8e0c854 | jax 0.6.2 | 5 | 2.56 (no test at n=5) |
| Canonical (paper) | 8e0c854 | jax 0.6.2 | 5 | 10.24 (2 collapses drive it) |

Collapse census (Final50 < 460, post hoc threshold, descriptive only):
plain Full pooled across all five sets: 4/55 = 7.3%. Shielded or stationary
arms (no-aux, stop-grad, frozen, anneal, all sets): 0/80.

So: a real, replicated deficit of roughly 2 to 3.5 points (0.4 to 0.7 percent
of the ~470 baseline), plus a rare collapse mode (~7 percent of Full runs) that
never appears when the pathway is severed or targets are stationary. The
canonical numbers were an unlucky draw containing 2 of the 4 collapses ever
observed in 55 plain-Full runs. Everything that "moved" during the discussion
window was small-n readings of variance/collapse patterns; nothing about the
mean deficit ever moved outside [1.9, 3.5].

## 3c. Pre-VAE replication, COMPLETE (10/10, 2026-07-25 night)

Original code (8e0c854) + original env + matched seeds:

| Cond | Seed | Replication | Canonical | Diff |
|---|---|---|---|---|
| A_full | 0 | 469.3 | 469.1 | +0.2 |
| A_full | 1 | 467.0 | 456.5 | +10.5 (collapse did NOT reproduce) |
| A_full | 2 | 471.2 | 450.6 | +20.6 (collapse did NOT reproduce) |
| A_full | 3 | 467.1 | 465.6 | +1.5 |
| A_full | 4 | 467.8 | 474.2 | -6.4 |
| A_no_aux | 0-4 | 471.06 mean | 473.44 mean | diffs -6.2 to +2.6 |

Replication A_full: 468.50 +/- 1.79. Replication gap at n=5: 2.56.

Readings:
1. NEITHER canonical collapse reproduced under the original code, env, and
   seeds. Collapse is a rare stochastic bifurcation, not seed-anchored and not
   code-era-specific. (V2 produced one collapse of its own: 459.3, seed-set
   independent.)
2. Fixed-seed reproducibility across a 3-month gap is LOOSER than the same-day
   estimate: non-collapse seed diffs span -6.4 to +2.6 (same-day estimate was
   1.4 from one seed twice). Likely XLA autotuning/driver variation (autotune
   warnings present in the July logs). Quote the honest range, not the 1.4.
3. The canonical 463.21 +/- 8.55 is a real observation but an unrepresentative
   draw; the pre-VAE replication mean (468.50 +/- 1.79) matches the fresh sets.

## 3d. Lambda x stationarity 2x2 — COMPLETE (10/10, 2026-07-25 night)

| Cell | n | Final50 | Learning-curve shape |
|---|---|---|---|
| lambda=0.05, drifting (Full, ref) | 20 | 467.99 ± 2.96 | normal; rare late collapse (~7%) |
| lambda=0.05, frozen (ref) | 10 | 468.8 ± 2.5 | normal; 0 collapses |
| lambda=0.95, drifting (hi_drift) | 5 | 440.5 ± 14.6 | learns fast (~250 by fifth 1), plateaus ~445; NO late collapse shape |
| lambda=0.95, frozen (hi_frozen) | 5 | 337.5 ± 67.4 | crippled from the START (fifth 1: 37-61); 4/5 plateau ~300-320; 1 seed escapes to 456 |

Per-seed hi_drift: 416 439 446 450 451. Per-seed hi_frozen: 289 303 319 321 456.

**Interpretation (constrained; use exactly this reading):**
1. The high-lambda row does NOT vindicate the directional mechanism: at
   lambda=0.95, damage does not require drift, and the frozen cell is far WORSE.
2. BUT the high-lambda row is also not a clean test of the paper's phenomenon.
   The learning curves show the frozen cell fails in EARLY training: at
   lambda=0.95 the aux loss dominates, and predicting a frozen random-policy
   snapshot is a pathological task that fights policy learning from step one
   (hi_frozen fifth-1 means 37-61 vs hi_drift 244-267). That is target
   learnability at initialization, not late-training interference near
   convergence, which is the paper's claim.
3. hi_drift shows a ~27-point deficit with a learned-and-held shape, again not
   the late-collapse phenomenon.
4. The coherent two-regime summary supported by ALL data: at HIGH aux weight,
   magnitude dominates and target drift is not required for damage (GradNorm's
   19x result now reads the same way). At the paper's LOW weight (0.05), the
   stationarity contrast is the operative one: drifting targets show the
   2-3.5 point deficit and all 4 observed collapses (4/55), frozen/stationary
   targets show neither (0/80). The directional-noise account is therefore
   scoped to the low-lambda near-convergence regime, and the 2x2 does not
   extend it upward.
5. One hi_frozen seed escaping to 456 is consistent with stochastic
   bifurcation in optimization.

DO NOT present the 2x2 as vindication of the mechanism. DO present it as the
requested weight-matched control, honestly read: it bounds the mechanism's
scope from above and confirms that GradNorm-scale damage is a magnitude effect.

## 3b. Pre-VAE replication, PARTIAL (superseded by 3c)

Original code (worktree 8e0c854), original env (jax 0.6.2), canonical flags,
matched seeds. This is the only valid replication of the canonical per-seed data.

| Cond | Seed | Replication | Canonical | Diff |
|---|---|---|---|---|
| A_full | 0 | 469.3 | 469.1 | +0.2 |
| A_full | 1 | 467.0 | 456.5 | +10.5 (canonical collapse did not reproduce) |
| A_full | 2 | 471.2 | 450.6 | +20.6 (canonical collapse did not reproduce) |
| A_no_aux | 0 | 471.2 | 475.9 | -4.7 |
| A_no_aux | 1 | 468.9 | 475.1 | -6.2 |

**Reading (partial, seeds 3-4 still running).** Two facts sit together and both
matter. A non-collapsed run reproduced almost exactly (seed 0, +0.2), so the
harness is close to deterministic given code, seed, and environment. But NEITHER
canonical collapse reproduced. The collapse is therefore a rare stochastic
bifurcation rather than a deterministic property of a seed: once a run tips into
it the trajectory is not reproducible, while runs that do not tip are.

This restores, on valid grounds, the conclusion that was retracted in section 2.
The earlier version of this claim was withdrawn because its test used current
code and so was never seed-matched. This test IS seed-matched (same commit, same
env, same flags), and it reaches the same answer.

Consequence for the paper: the canonical A_full mean of 463.21 with sd 8.55 and
2/5 collapses is a real observation but an unlucky sample of a heavy-left-tailed
distribution. Pooled plain-Full collapse rate across all matched runs is roughly
5 to 10 percent, so 2/5 was high. Report the collapse as a low-rate stochastic
event whose rate rises with effective auxiliary weight, and do not present
canonical per-seed values as reproducible.

## 3. Pre-VAE replication (original entry, superseded by 3b)

Original code (worktree at 8e0c854) + original env (jax 0.6.2) + canonical
flags, seeds 0-4, A_full and A_no_aux. This is the only valid test of whether
the canonical per-seed numbers reproduce. Targets to compare against are the
canonical per-seed values in section 1. PENDING.

## 3e. ROUND-2 PRE-REGISTRATION (written 2026-07-26 before results)

Timing disclosure: R8 had 3 of 15 result files on disk when this was written
(none inspected beyond the smoke test); R9-R12 had zero. Per
rigorous-experiments, refutation criteria are stated here BEFORE analysis.

**R8 (J_pi finite-difference, Full vs stop-grad vs No-Aux).**
Claim tested: the aux gradient responds to target-policy changes through the
shared encoder (the pathway of Sigma_eps = J Sigma_pi J^T).
Supports: jpi_rel scales approximately linearly in eps (ratio eps0.2/eps0.05
in ~[3, 5]) and the response is materially smaller for stop-grad (encoder
pathway severed) than Full.
Refutes/weakens: response flat in eps, or stop-grad response >= Full response.
Metric: jpi_rel_eps* averaged over late-training log points, per seed.

**R9 (aux-task variants; mini replication of the central contrast).**
Prediction stated in advance: latent (structured, drifting) shows a Final50
deficit vs No-Aux (469.91 +/- 2.86, n=20 reference); recon (structured,
stationary) shows no deficit.
Refutes: recon deficit >= latent deficit, or latent shows no deficit.
Caveat pre-stated: variant heads add parameters No-Aux lacks and gradient
magnitudes are not norm-matched across tasks; comparison is qualitative at
n=5; report mean +/- sd, no inferential claim.

**R10 (random gating at p=0.70).**
Claim tested: whether the tau=0.10 drift gate's benefit is duty cycle or
trigger timing. Reference: gate tau=0.10 471.90 +/- 1.93 (n=5).
Trigger matters: random-gate mean clearly below 471.90 (non-overlapping
mean +/- sd bands at n=5 is the most we can say).
Duty cycle explains it: random-gate within the gate's band. Either outcome is
reportable; the second retires the gate as mechanism evidence permanently.

**R11 (schedules: cosine, exp, kl_adaptive).**
Claim tested: the fix family is robust to schedule choice (fXvf W5 "fixes
inconsistent"; yGKw Q8). Supports: all three within or above the No-Aux band
with no sub-460 run. Refutes/weakens: any schedule with a sub-460 run or
clearly below Full. KL_REF = 2.5e-3 in kl_adaptive was calibrated on the
round-1 KL measurements (disclosed; not tuned on round-2 outcomes).

**R12 (belief effective rank, SMAX and AA).**
Question (PYCT Q1): does Full show reduced late-training representation rank
vs No-Aux where the residual SMAX gap lives?
Supports secondary-pathology hypothesis: Full late rank clearly below No-Aux
on SMAX. Null: ranks comparable -> representation collapse loses its lead-
candidate status and we say so. No pre-commitment on AA.

Language rule for all round-2 reporting: n=5 results are labeled
"directional" (mean +/- sd); "significant" is reserved for n=20 bootstrap-CI
results. This applies retroactively to how existing n=5 numbers are labeled
in the rewrite.

## 3g. R8 RESULTS (complete 2026-07-26, analyzed against pre-registration 3e)

J_pi finite-difference (late-training mean of jpi_rel, relative aux-gradient
response to target perturbation), n=5/arm, AA:

| Cond | eps=0.05 | eps=0.1 | eps=0.2 | linearity (0.2/0.05) | belief rank | F50 |
|---|---|---|---|---|---|---|
| Full | 1.251 | 2.473 | 4.940 | 3.95 | 21.7 | 466.1 |
| Stop-grad | 0.770 | 1.522 | 3.041 | 3.95 | 15.8 | 471.9 |
| No-Aux | (no aux gradient) | — | — | — | 16.1 | 471.1 |

**Verdicts (both pre-registered support criteria HIT):**
1. Linearity 3.95, inside the pre-registered [3, 5] band: the aux gradient
   responds linearly to target-policy perturbation, as a first-order J_pi
   model predicts.
2. Pathway contrast: Full 2.473 vs stop-grad 1.522 at eps=0.1, with
   NON-OVERLAPPING per-seed ranges (Full min 1.917 > stop-grad max 1.719).
   Severing the encoder pathway reduces the aux-gradient response by ~40%;
   the encoder pathway carries the excess sensitivity.
CONSEQUENCE: J_pi is now MEASURED. Remove every "J_pi remains unmeasured"
concession from the responses and replace with this result. This is the
campaign's first clean pre-registered mechanism win; label it "directional,
n=5" but note the non-overlapping ranges.

3. Belief effective rank (descriptive, no pre-commitment): Full 21.7 is
   HIGHER than No-Aux 16.1 and stop-grad 15.8. The auxiliary loss INFLATES
   representation rank on AA; the deficit coexists with richer, not
   collapsed, representations. Representation collapse is disfavored as the
   explanation on AA; final word for SMAX awaits R12. Do not overclaim: one
   environment, n=5, descriptive.

## 3h. R9-R12 RESULTS (complete 2026-07-26 night, analyzed against pre-reg 3e)

All n=5, directional, mean +/- sd. References: No-Aux n=20 469.91 +/- 2.86;
Full n=20 467.99 +/- 2.96; drift gate tau=0.10 471.90 +/- 1.93.

**R9 aux-task variants: REFUTES the pre-registered prediction.**
latent (drifting, separate-head MSE): 471.85 +/- 1.43. recon (stationary,
separate-head MSE): 472.95 +/- 1.26. BOTH sit above No-Aux; neither shows a
deficit; zero sub-460. The pre-registered prediction (latent deficit, recon
none) fails on the latent arm. NOTE: the analysis script's automatic verdict
printed "SUPPORTS" due to a sign error; the correct verdict is REFUTES, and
the script output must not be quoted.
Honest interpretation (pre-stated caveat applies): the variants use a
separate head and MSE losses whose gradient scale was NOT norm-matched to
the action-prediction CE loss, and variant runs did not log gradient norms.
Therefore this result cannot distinguish "the deficit is specific to the
in-network action-prediction formulation" from "the variant aux gradients
were simply too small to matter at lambda=0.05". Scope statement for the
rebuttal: the deficit is demonstrated for next-action prediction with CE
through the shared encoder; a belief-MSE variant did not reproduce it at
n=5, with gradient-scale not matched. No stronger claim either way.

**R10 random-gate control: trigger timing MATTERS (supports the gate).**
Random gating at p=0.70: 466.72 +/- 2.52, clearly below the drift gate
tau=0.10 (471.90 +/- 1.93; per-seed overlap only at one point) and at
plain-Full level. Gating on the cosine-std signal at matched duty cycle
outperforms gating randomly. Consequence: the earlier reading "the gate
approximates stop-gradient by duty cycle" is WITHDRAWN; the within-run
timing signal carries information even though the between-condition
cosine-std diagnostic failed. State both facts together; do not resurrect
the between-condition diagnostic.

**R11 schedules: SUPPORTS (fix family robust to schedule choice).**
cosine 471.65 +/- 1.54; exp 472.99 +/- 3.47 (best arm in round 2);
kl_adaptive 468.38 +/- 1.77. Zero sub-460 anywhere. All within or above the
No-Aux band. Directly answers yGKw Q8 and fXvf W5.

**R12 SMAX rank: representation collapse DISFAVORED where PYCT asked.**
Late belief effective rank: Full 35.22 +/- 2.73 vs No-Aux 30.39 +/- 2.57
(Full HIGHER, same direction as AA in 3g). Also F50 in this batch: Full
10.57 vs No-Aux 10.49, i.e. the SMAX "gap" reversed sign relative to the
earlier fresh batch (10.25 vs 10.56): further confirmation that SMAX shows
no stable effect. Answer to PYCT Q1: we measured their hypothesis; the
collapse account is disfavored on both environments; the SMAX gap itself is
not a stable phenomenon.

## 4. Gradient-surgery baselines (old stack, n=5 each)

- PCGrad: 461.0 ± 5.0. Per-seed 455, 456, 462, 466, 466.
- GradNorm: 446.2 ± 15.8. Per-seed 415, 450, 452, 454, 460. Learned weights
  converge to w_policy 1.053 / w_aux 0.947, i.e. the auxiliary term is
  effectively weighted ~19x the paper's lambda = 0.05.

**Interpretation constraint (important).** GradNorm degrading performance
shows that a large auxiliary gradient is harmful. It does NOT by itself
discriminate directional interference from magnitude interference; if
anything a magnitude dose-response is the natural reading of a
magnitude-scaling intervention. Do NOT present GradNorm as the strongest
evidence for the directional mechanism. The DISCRIMINATING result is PCGrad:
projection removes pairwise conflict, so if persistent conflict were the
mechanism, PCGrad should have repaired the deficit and it did not.

## 5. Snapshot-lag continuum, soft targets (old stack, n=5 each)

| Condition | Final50 | aux self-cosine |
|---|---|---|
| lag 1 (fastest drift) | 468.0 ± 1.8 | 0.861 |
| lag 25 | 466.0 ± 3.3 | 0.791 |
| lag 100 | 466.8 ± 5.6 | 0.832 |
| frozen (no drift) | 470.5 ± 2.3 | 0.974 |

Endpoints separate in the predicted direction (frozen best). The INTERIOR is
not ordered and not resolved at n=5: state this honestly. The self-cosine
diagnostic does separate stationary (0.974) from drifting (0.79-0.86) targets
under matched soft labels.

## 6. Drift-gated controller (old stack, n=5 each)

- tau = 0.10 (gate active ~70% of iterations): 471.90 ± 1.93
- tau = 0.15 (gate active ~30%): 467.2 ± 1.1

**Constraint.** vs No-Aux (469.91 ± 2.86, n=20): difference 1.99, Welch
p = 0.097. NOT significant. Do NOT write "best configuration in our study" or
"exceeds No-Aux". Correct claim: the gate matches the best shielded
configurations and beats un-gated Full. Also disclose that at 70% duty cycle
it approximates stop-gradient.

## 7. SMAX (old stack, fresh instrumented, n=5 each)

- Full 10.25 ± 0.29, late-training E[cos] = +0.0062
- No-Aux 10.56 ± 0.56, E[cos] = 0.000
- Stop-grad 10.36 ± 0.14, E[cos] = 0.000

Reading: the zero-mean noise assumption HOLDS (E[cos] ~ 0). Directional bias
does NOT explain the residual SMAX mean gap. The SMAX effect is small and
should not be presented as a strong instance of the pathology.

## 8. Direct drift measurement (old stack)

- Consecutive-policy KL, late training: ~2.4e-3 to 2.6e-3 nats in ALL
  conditions (full 2.46, no_aux 2.49, stopgrad 2.42, frozen 2.58).
- corr(late KL, cosine-std) = 0.60 across 15 aux-ON runs.
- Honest caveat: this KL measures LIVE-policy drift, which is near-identical
  across conditions by construction. It equals TARGET drift only for Full.
- Between-task cosine-std: full 0.160, frozen 0.226, no_aux 0.000,
  stopgrad 0.000. **The between-task cosine-std does NOT separate frozen from
  full.** The per-task self-cosine (section 5) is the diagnostic that works,
  and only under soft targets.

## 9. Run counts (verified by file count, 2026-07-25)

R1_kl_instrumented 55, R2_grad_surgery 10, R3_snapshot_lag 20,
R4_drift_gate 10, R7_smax_instrumented 15, validation 4, V2_verification 33
(growing). **Total completed full-length runs: 147.** Do not write "130+"
or any other unverified figure; use 145+ only while the count is at least that.

## 9b. Evaluation-window sensitivity (answers fXvf Q6)

Full vs No-Aux gap under different evaluation windows:

| Stack | Final25 | Final50 | Final100 | Final200 |
|---|---|---|---|---|
| old (jax 0.6.2), n=20 | 1.78 | 1.92 | 2.45 | 3.70 |
| V2 (jax 0.10.2), n=20 FINAL | 3.61 | 3.47 | 3.05 | 3.29 |

(Superseded n=17 V2 window row: 3.74/3.89/3.28/3.58. Do not quote. The V2
range at n=20 is 3.05 to 3.61.)

The conclusion does not depend on the window. On the old stack the gap GROWS
with a longer window, so Final50 is if anything conservative. This is the
answer to "justify the Final50 window": it is not a cherry-picked operating
point, and the contrast is reported at three other windows.

Peak-to-final drop (smoothed peak minus Final50):

| Stack / condition | mean | sd | max |
|---|---|---|---|
| old / Full (n=20) | 4.77 | 2.47 | 9.51 |
| old / No-Aux (n=20) | 4.82 | 1.83 | 8.41 |
| V2 / Full (n=17) | 4.29 | 2.29 | 10.05 |
| V2 / No-Aux (n=17) | 3.13 | 1.26 | 5.99 |

Honest reading: on the old stack the peak-to-final drop is the SAME for Full
and No-Aux (4.77 vs 4.82), so that metric does not distinguish them there. On
the V2 stack Full drops further and more variably (4.29 +/- 2.29, max 10.05)
than No-Aux (3.13 +/- 1.26, max 5.99). Report both, including the old-stack null.

## 9c. Facts about the submitted paper itself (verified against source, 2026-07-25)

- **The accessibility appendix is Appendix L, not M.** The submission's
  appendices run A through L (`\section{Figure and Reader Accessibility}` is the
  last one). Earlier drafts said "Appendix M", which does not exist. Any
  commitment to remove it must name Appendix L.
- **The word "prompt" occurs exactly once in the entire source**, in Appendix L
  ("the title opens with a curiosity prompt before stating the mechanism"). It
  is VISIBLE text and it is about our own paper, whereas fXvf described HIDDEN
  and UNRELATED text. Report the overlap as a possible source of the reviewer's
  observation; do not assert it is the explanation.
- **The CIFAR-100 / supervised result is NOT a clean null and the paper says so.**
  Source text: "this as a weak boundary residual rather than a clean null"; the
  cleaner null is on Best accuracy, d = -0.16 with CI [-1.78, +1.19]. That CI is
  extremely wide, so this is an UNDERPOWERED null, not a confirmed prediction.
  Do NOT describe it as a passed prediction or a clean boundary confirmation,
  and do not quote "CI crossing zero" as if that established the boundary.

## 10. Claims that must NOT appear in the rebuttal

1. Any assertion that NeurIPS organizers inserted text into the PDF, unless an
   official announcement is cited. Verified and safe to say: our LaTeX source
   and both submitted PDFs contain no hidden text (checked by text-layer
   extraction on 2026-07-24).
2. "Collapse does not reproduce seed-matched" / the Fisher p = 0.024.
3. Drift gate as "best in study" or "exceeds No-Aux".
4. GradNorm as the strongest evidence for the directional mechanism.
5. Priority claims ("first to ...").
6. Any n=20 V2 figure until V2 actually reaches n=20.

## 3f. R13 PRE-REGISTRATION: EMA-distilled targets (written 2026-07-26, before results)

The target-design answer to yGKw limitation 3 (root cause vs symptom).
Scheme: auxiliary targets are soft actions of an EMA copy of the policy
(alpha 0.995 and 0.99, 5 seeds each), completing the designed spectrum
live -> EMA -> periodic snapshot -> frozen.
Supports (pre-stated): Final50 at or above the plain-Full band (467.99 +/-
2.96 reference) with zero sub-460 runs; ideally within the No-Aux band
(469.91 +/- 2.86).
Refutes/weakens: any sub-460 run, or mean clearly below plain Full (the
scheme would then be harmful, not protective).
Label: directional at n=5, mean +/- sd, no inferential claim.

## 3i. R13 RESULTS (complete 2026-07-26 night, analyzed against pre-reg 3f)

EMA-distilled auxiliary targets (target-design intervention), n=5 each,
directional. References: Full n=20 467.99 +/- 2.96; No-Aux n=20 469.91 +/- 2.86.

| alpha | Final50 | per-seed | sub-460 |
|---|---|---|---|
| 0.995 (slow EMA) | 469.88 +/- 3.43 | 466 467 469 473 473 | 0 |
| 0.99 (faster EMA) | 466.69 +/- 3.71 | 463 463 467 469 472 | 0 |

Verdict vs pre-registration: alpha=0.995 hits the top support tier (zero
sub-460 AND mean at the No-Aux level: 469.88 vs 469.91). alpha=0.99 is
within the Full band with zero sub-460 (not refuted, weak). The designed
target spectrum now reads, at the endpoints: frozen-soft 470.5 > EMA-0.995
469.88 ~ No-Aux 469.91 > Full 467.99 > EMA-0.99 466.69 (last inversion
within n=5 noise). Claim for the rebuttal: a slow-EMA distilled target
restores No-Aux-level performance while RETAINING the auxiliary task, with
zero sub-460 runs across the design family; this is a root-cause,
target-level intervention. Label directional at n=5; do not claim the full
spectrum is monotone.
