---
tags: [neurips2026, rebuttal, final]
status: active
related: [VERIFIED_NUMBERS, rebuttal_evidence_audit, neurips2026_reviews]
---

# Author responses, NeurIPS 2026 Submission 29511

Assembled 2026-07-25. Every number traces to `VERIFIED_NUMBERS.md`; nothing is quoted
that is not verified there. Drafted by section, adversarially reviewed for overclaims,
audited for numerical fidelity, then condensed with a concession-preservation check.

Two experiments are still running and their slots are marked **[PENDING]**. Do not post
until they land and the marked claims are resolved:
1. Pre-VAE replication at commit 8e0c854 (does the canonical per-seed data reproduce
   under its own code?).
2. Lambda x stationarity 2x2 (does damage at high auxiliary weight require target drift?
   This is the only clean test separating the directional account from a magnitude one).

# Top-level comment to Area Chair y2yo

We used the discussion period to re-run the paper's headline claims at larger n and to
implement the controls the reviewers asked for: 147 completed full-length runs. Some
claims strengthened, several did not survive, and we report both below. Conventions:
Final50 is the mean reward over the last 3200 episodes; intervals are 20k-sample
bootstrap; p is Welch; d uses the pooled sd.

## 1. The headline comparison

| Set | Code | n/arm | Full | No-Aux | Gap [95% CI] | p | d |
|---|---|---|---|---|---|---|---|
| Canonical (paper) | 8e0c854, jax 0.6.2 | 5 | 463.21±8.55 | 473.45±3.62 | 10.24 | not computed | not computed |
| Fresh | current, jax 0.6.2 | 20 | 467.99±2.96 | 469.91±2.86 | 1.92 [0.17, 3.66] | 0.044 | 0.66 |
| Fresh | current, jax 0.10.2 | 20 | 468.14±3.49 | 471.62±2.54 | 3.47 [1.68, 5.37] | 0.00098 | 1.14 |

*This set is still accruing; we will recompute at the final matched n and will not quote
an n=20 figure for it until it reaches n=20.

What holds: the Full-versus-No-Aux deficit replicates in direction in both fresh sets.
It is also robust to the evaluation window, which reviewer fXvf asked us to justify: on
the original stack the gap runs 1.78, 1.92, 2.45, 3.70 at Final25, Final50, Final100 and
Final200, so Final50 is a conservative choice rather than a favourable one.

What does not hold: the magnitude. Canonical 10.24 lies outside both fresh estimates,
carries no computed test, and at n=5 with Full sd 8.55 should not have been the headline.
We are not able to attribute the shortfall to the numerical stack, because the largest
disagreement in the table is within a single stack: canonical 10.24 against fresh 1.92,
both on jax 0.6.2. Small-n sampling and the code change between them both remain live.
The two fresh sets are also not independent replications; they share code, task, seeds
and experimenters.

Collapse counts (Final50 < 460; the threshold was set after seeing the data and is
therefore descriptive, with no test computed on it): canonical Full 2/5; fresh old-stack
0/20 in both arms, with a minimum of 463 in that set; jax 0.10.2 Full 1/20 and No-Aux
0/20. We are removing the bimodality framing.

Variance ratio (Full/No-Aux): canonical 5.58 at n=5, old stack 1.07 (p=0.88) at n=20,
jax 0.10.2 1.89 (p=0.175) at n=20. The asymmetry the paper claims fails both larger-n
tests, so we withdraw it. We flag one detail because it bears on how the paper's own
n=5 result should be read: an interim n=16 snapshot of the jax 0.10.2 set did show
3.30 (p=0.027), and the effect disappeared when the last four seeds per arm arrived.
Small-n variance ratios in this setting are unstable, which is the most likely
explanation of the canonical 5.58 as well.

## 2. A reproducibility problem we found

Commit 6b37a34 changed the trainer's RNG structure, so current code cannot reproduce any
April run even at a matched seed: such a run is an independent sample path, not a
replicate. This invalidates seed-matched comparisons across that boundary, and it is why
we no longer characterise the canonical collapses as reproduced or as refuted. It does
not by itself explain the canonical mean sitting lower with a much larger spread. The
valid test, replication at 8e0c854 on jax 0.6.2 with canonical flags and seeds 0 to 4,
is running now. **[PENDING: pre-VAE replication]** We will post the result either way. If
the canonical per-seed values do not reproduce under their own code, the fresh estimate
becomes the headline and the collapse cases are withdrawn.

We also found that the expB batch is split across code versions (seeds 0 to 3 pre-VAE,
seed 4 post-VAE within every condition). The headline canonical_phase2 set is internally
clean, all seven configurations from one commit.

Separately, we can now bound harness noise: two same-day reruns of one seed under
identical code gave 469.5 and 468.1, a spread of 1.4 points. That is small against the
2.96 between-seed sd, but it is the same order as the 1.92-point old-stack gap, which is
one reason we report that comparison at n=20 rather than n=5. This calibration rests on a
single seed rerun twice and we will expand it.

## 3. Contribution category

Filing under Theory was our misjudgment. All three reviewers recorded Contribution Type:
General and we agree that is the correct reading; we do not dispute the point. Proposition
1 is being restated as an explicit approximation with its assumptions named, J_pi remains
unmeasured, and the diagnostic the theory was tied to does not do the job we assigned it
(section 5). We would rather the paper be judged as an empirical contribution than defend
the Theory label.

## 4. On overclaimed scope

We concede this. The abstract states the mechanism with no scope condition, and it
disagrees with our own Appendix D, which describes the supervised case as a weak boundary
residual rather than a clean null; at five seeds per cell that experiment is uninformative
rather than confirmatory. Both are our errors. We are rescoping the claim to structured,
co-adaptively generated auxiliary targets whose gradients flow through a shared actor
encoder, and we note fXvf's objection applies to the supervised control as well, since it
does not separate predictability from drift.

Within MARL, we should also state the harder version of your point: every mechanism
experiment runs in one Overcooked layout. SMAX shows no detectable effect (section 5),
Hanabi and Cramped Room were not rerun in this window, and the cross-environment table is
correlational support rather than four demonstrations.

## 5. Mechanism evidence, reported separately from effect size

We separate these deliberately, because the phenomenon replicating does not establish our
account of it.

Consistent with the account: GradNorm, which raises the auxiliary weight to
w_aux 0.947 against w_policy 1.053 (roughly 19x the paper's lambda = 0.05), degraded all
five seeds (446.2±15.8). Every configuration that severs the pathway or removes target
drift (No-Aux, stop-gradient, frozen targets, annealing) produced no collapses in more
than 70 runs.

Not established by our data: GradNorm is a magnitude intervention producing a magnitude
dose-response, so it does not discriminate our directional account from a plain magnitude
account. PCGrad, the intervention that would discriminate, did not repair the deficit
(461.0±5.0, n=5, against un-gated Full 467.99±2.96, n=20), which is consistent with our
claim that this is not persistent conflict; but I_mag is already near zero so there was
little for projection to remove, the projection was not instrumented, and our account
predicts neutral rather than harmful, so landing below Full is unexplained.

Refuted or withdrawn: our between-task cosine-std diagnostic is reversed, not merely
uninformative (Full 0.160 against frozen 0.226, with No-Aux and stop-gradient at 0.000 by
construction), so the stationary condition is the more directionally variable one. Every
claim resting on that diagnostic is being removed or re-captioned. The snapshot-lag
continuum does not order its interior as predicted; the fastest-drifting arm is the best
of the three drifting arms. On SMAX the effect is not detectable (Full 10.25±0.29 against
No-Aux 10.56±0.56, no test computed), and measured E[cos] is +0.0062, so we cannot reject
the zero-mean assumption; that also means directional bias does not explain the residual
SMAX mean gap that reviewer PYCT asked about. SMAX moves to limitations.

A weight-by-stationarity control that would discriminate the two accounts directly, at
matched auxiliary weight with and without target drift, is running.
**[PENDING: lambda x stationarity 2x2]**

## 6. The PDF text layer

On 2026-07-24 we extracted the text layer of our LaTeX source and of both submitted PDFs
and found no hidden text. The single occurrence of the word "prompt" anywhere in the
source is visible text in Appendix L, drafting meta-commentary about how the paper reads
to a reviewer. It is not hidden and not an instruction, but it should not have been in a
submission, it is the most likely source of the report, and we are removing it. There is
no Appendix M. We can supply the extraction command and its output, and we would
appreciate the review-system copy being checked. We make no claim about anything
downstream of our upload.

---

# Response to Reviewer yGKw

Q1, Q2, Q5, Q6, Q7 ran; Q3, Q4, Q8 untested and unclaimed.

## Headline: smaller than reported, canonical numbers unverified

Commit 6b37a34 changed the trainer's RNG structure: current code cannot reproduce any canonical run at a matched seed, so seed-matched reruns are not replicates.

| Set | n/arm | Gap [95% CI] |
|---|---|---|
| Canonical (jax 0.6.2) | 5 | 10.24 |
| Fresh old (jax 0.6.2) | 20 | 1.92 [0.17, 3.66] |
| Fresh V2 (jax 0.10.2) | 20 | 3.47 [1.68, 5.37] |

Direction replicates, magnitude does not: canonical 10.24 falls outside both fresh estimates, the biggest disagreement being within one stack (10.24 against 1.92). Fresh intervals overlap, no interaction test was run: magnitude unresolved. The fresh sets are not independent replications, one JAX minor version apart.

The RNG change does not explain canonical Full (463.21 ± 8.55) sitting below fresh Full (467.99 ± 2.96): small-n sampling and an added VAE belief pathway remain live. The separating replication (8e0c854, jax 0.6.2, seeds 0 to 4) is running. **[PENDING]**

Jitter: same-day reruns of one seed gave 469.5 and 468.1 (spread 1.4), comparable to the old-stack gap of 1.92; one seed, twice. The jax 0.10.2 set is now complete at n=20 per arm. Mechanistic results below are n=5, just conceded inadequate, and cross-batch against a separate n=20 baseline.

## Weaknesses

**W1** partly addressed, with a negative result: Sigma_pi is now measured directly but is flat across conditions and so cannot discriminate them (Q1); J_pi remains untested (Q3); eta is fixed by design. We claim no resolution here.
**W2** accepted: Q6 gives two operating points, not a calibration; Eq. 8 becomes qualitative.
**W3** both run (Q5).
**W4** agreed: soft labels remove the argmax confound, but no tested endpoint separation, and the interior contradicts prediction (Q2).
**W5** no lambda points added; the non-monotonicity is unexplained, counter-evidence to a dose-response in our one directional-noise knob. Q6 covers coarseness only.

## Questions

**Q1.** Late consecutive-policy KL is flat across conditions (2.42 to 2.58, 1e-3 nats). corr(late KL, cosine-std) = 0.60 over 15 auxiliary-ON runs: no CI, one environment, not across tasks. It tracks live-policy drift, equal to target drift only in Full: no between-condition discrimination, and its correlate is withdrawn below.

**Q2.** n=5, old stack, matched soft labels. Final50: lag 1 468.0 ± 1.8, lag 25 466.0 ± 3.3, lag 100 466.8 ± 5.6, frozen 470.5 ± 2.3. Frozen over lag 1 untested, no separation claimed. The interior is counter-directional (fastest drift best of the drifting arms); self-cosine does not order arms by lag. Predictability is still confounded with drift, a magnitude account fits frozen-best equally well, and without late auxiliary gradient norms nothing separates directional from magnitude interference.

**Q3, Q4, Q8.** Not run. No claim about J_pi beyond the linearization. Other auxiliary definitions are the most important untested generalization, so the claim is scoped to structured, co-adaptive targets through a shared actor encoder. Schedule sensitivity untested (the Q6 gate is one family at two thresholds).

**Q5.** n=5 each. PCGrad 461.0 ± 5.0; GradNorm 446.2 ± 15.8, at roughly 19x the paper's auxiliary lambda = 0.05. PCGrad does not repair the deficit: below un-gated Full (467.99 ± 2.96, n=20, separate batch), spread no smaller, no variance-ratio test. Weakly diagnostic anyway (I_mag near zero, projection uninstrumented); we cannot explain it landing below Full when we predict neutral. GradNorm degrades while scaling the auxiliary term: magnitude dose-response, not leaned on.

**Q6.** Binary stop-gradient on a cosine-std threshold. tau = 0.10 (active ~70% of iterations) 471.90 ± 1.93; tau = 0.15 (~30%) 467.2 ± 1.1. Against No-Aux (469.91 ± 2.86, n=20, separate batch), 1.99 at Welch p = 0.097: the gate neither exceeds No-Aux nor is best in the study; at 70% duty it approximates stop-gradient. Only the near-always-on setting beats un-gated Full: "more shutoff is better" explains both points without drift, and the trigger itself fails to separate drifting from stationary targets. No matched-duty random control, no window sweep.

**Q7.** n=5, fresh instrumented: Full 10.25 ± 0.29, late E[cos] = +0.0062; No-Aux 10.56 ± 0.56 and Stop-grad 10.36 ± 0.14 at E[cos] = 0.000 by construction, so Full is the only measured cell; with no CI we can only say we cannot reject zero-mean noise. Bias in eps untested. Directional bias does not explain SMAX; with no test of 10.25 against 10.56 at these spreads it shows no detectable effect rather than a small one, and moves to limitations, leaving mechanism evidence in Overcooked Asymmetric Advantages alone; Hanabi's ordering is unsettled, Cramped Room not rerun at larger n.

## Significance and Originality

No priority is claimed; that framing is removed, which fixes an overclaim, not originality. Standing: no collapse at n=20, a variance asymmetry that fails its larger-n replication, and a surviving effect roughly a fifth to two fifths of the reported size. Resubmission after the pending replication is a reasonable call.

The diagnostic is reversed, not just uninformative: between-task cosine-std is full 0.160 against frozen 0.226 (no_aux, stopgrad 0.000, pathway off), frozen best in Q2. Figures resting on it get re-captioned or removed.

Your limitations hold: our interventions, gate included, are symptomatic gradient-path engineering, not target design.

---

# Response to Reviewer fXvf

W2 was vindicated: at higher n the effect shrinks and claims fail.

## W2, Q6

| Set | n | Full | No-Aux | Gap [95% CI], p |
|---|---|---|---|---|
| Canonical (jax 0.6.2) | 5 | 463.21 ± 8.55 | 473.45 ± 3.62 | 10.24, no p or d |
| Fresh old (jax 0.6.2) | 20 | 467.99 ± 2.96 | 469.91 ± 2.86 | 1.92 [0.17, 3.66], p=0.044 |
| Fresh V2 (jax 0.10.2) | 20 | 468.14 ± 3.49 | 471.62 ± 2.54 | 3.47 [1.68, 5.37], p=0.001 |

The fresh pair shares code and seeds; canonical is a different codebase. None is an
independent replication. Direction replicates on both fresh sets; the canonical magnitude
does not, and at a Full sd of 8.55 should not have been the headline. Magnitude unresolved
rather than stack-sensitive: fresh CIs overlap and the largest disagreement (10.24 vs 1.92)
is within one stack. Variance ratios: canonical 5.58 (n=5), old stack 1.07 (p=0.88, n=20),
jax 0.10.2 1.89 (p=0.175, n=20). The asymmetry fails both n=20 tests and is dropped; it
showed 3.30 (p=0.027) at an interim n=16 and vanished with the last four seeds per arm.
The post hoc collapse threshold (<460) gives descriptive counts (canonical
Full 2/5, fresh old 0/20, V2 Full 1/20); bimodality and the failure vocabulary are retired
(Q1, Q7), both fresh gaps under 1% of the roughly 468-point baseline.

Commit 6b37a34 changed the RNG structure: current code cannot reproduce any April run at a
matched seed. Small-n sampling and an added VAE belief pathway both remain live; the 8e0c854
replication (seeds 0-4) is running **[PENDING]**. expB is split by code version (seeds 0-3
pre-VAE, seed 4 post-VAE), and reproducibility is not established.

Q6: Final50 is the mean reward over the last 3200 episodes; the gap holds at Final25/100/200,
and peak-to-final drop does not separate the arms on the old stack (4.77 vs 4.82).

## W1, Q5

Accepted. Snapshot-lag under matched soft targets, n=5: lag 1 468.0 ± 1.8, frozen 470.5 ± 2.3,
the interior lags below both. Endpoints run as predicted, untested; the
fastest-drift arm beats the slower drifting arms, against our mechanism; the self-cosine
separates stationary from drifting but does not order the drifting arms. Predictability stays
inseparable from drift and a magnitude account explains frozen-best equally well, so this
does not discriminate the mechanisms.

## W3, Q4

Late-training policy KL is flat (2.42 to 2.58, 1e-3 nats); corr(KL, cosine-std) = 0.60 over
15 auxiliary-on runs, no CI. It is live-policy drift (equal to target drift only in
Full), the flat range makes the correlation pooled, and frozen has the highest KL, cosine-std
and Final50: consistent with, not a test of, the link. The between-task cosine-std is
reversed relative to its prediction, so we withdraw that diagnostic and the claims resting
on it. J_pi is unmeasured; Prop 1 becomes an explicit
approximation (Q4).

## W4

Your three assumptions remain untested, carried as assumptions. We tested zero-mean noise
on SMAX (n=5): E[cos] = +0.0062 in Full, structurally zero elsewhere, no CI, so we cannot
reject it. That cuts against us: our mechanism does not account for whatever difference
exists on SMAX.

## W5

Accepted. SMAX Full 10.25 ± 0.29 against No-Aux 10.56 ± 0.56 at n=5, no test computed: no
detectable effect rather than a small one; SMAX moves to limitations, as does critic-side
placement raising variance. Drift gate (n=5): tau=0.10 471.90 ± 1.93, tau=0.15 467.2 ± 1.1;
against No-Aux (469.91 ± 2.86, n=20) the 1.99 difference has Welch p=0.097, so no claim of
exceeding No-Aux or being best; "beats un-gated Full" is untested and cross-batch; at 70%
duty cycle the gate approximates stop-gradient.

## W6, Q2 (PDF)

Text-layer extraction on 2026-07-24 of our source and both submitted PDFs found no hidden
text. The one occurrence of "prompt" is visible text in Appendix L, drafting meta-commentary
mentioning a "curiosity prompt" in the title; that paragraph should never have been in a
submission and is the likeliest thing a reading pass surfaced; we are deleting it. We
claim nothing downstream of our upload.

## Q1

Accepted: scope restricted to structured, co-adaptive auxiliary targets whose gradients flow
through a shared actor encoder, "harmless" removed, absence-of-effect claims replaced by
intervals with sample sizes (no power figure computed). The abstract's near-zero CIFAR-100
description will match Appendix D's weak boundary residual.

## Q2 (gradient surgery)

Related work now covers moving-target learning, dynamic teachers and adaptive weighting.
Baselines at n=5: PCGrad 461.0 ± 5.0; GradNorm 446.2 ± 15.8. PCGrad does not repair the
deficit, landing below un-gated Full; but our I_mag is near zero, the projection was uninstrumented, and our account predicts neutral
not harmful, so we call it consistent, not discriminating. GradNorm does not discriminate
directional from magnitude interference; a magnitude dose-response is the natural reading.

## Q3

Shared parameters with a learned agent-identity embedding; the auxiliary head predicts each
visible teammate's next action from its attended belief slice, with visibility masks gating
attention and loss so masked teammates give no gradient.

## Q7

Limitations rewritten around every item above, plus the Q1 scope restriction. We agree
with the AC that the Theory category was our misjudgment. We will run any control you name.

---

# Response to Reviewer PYCT

## Headline result

| Set (stack, n/arm) | Full | No-Aux | Gap [95% CI] |
|---|---|---|---|
| Canonical, paper (0.6.2, n=5) | 463.21 ± 8.55 | 473.45 ± 3.62 | 10.24 |
| Fresh old (0.6.2, n=20) | 467.99 ± 2.96 | 469.91 ± 2.86 | 1.92 [0.17, 3.66] |
| Fresh V2 (0.10.2, n=20) | 468.14 ± 3.49 | 471.62 ± 2.54 | 3.47 [1.68, 5.37] |

Canonical 10.24 does not reproduce at larger n on either stack. Intervals overlap and no interaction test was run, so magnitude is unresolved, not stack-sensitive. The collapse and bimodality framing is retired. The Final50 < 460 threshold was post hoc; descriptively, canonical Full 2/5, fresh old 0/20 both arms, V2 Full 1/20, No-Aux 0/20.

Commit 6b37a34 changed the trainer's RNG structure, so current code cannot reproduce the April runs at matched seeds; that explains failed seed matching, not the canonical Full mean and spread. A five-seed reproduction under the original commit and environment is running **[PENDING]**. Pre-committed: if those per-seed values do not reproduce, the fresh estimate becomes the headline and collapse cases are withdrawn.

## W1: only our reconstruction is validated

Accepted. The Static-Belief comparison cannot discriminate mechanisms (one data point, a system we did not run, no seed-level data).

Matched soft-target snapshot lag (n=5) fixes label distribution and state dependence, varying only drift. Frozen is best (470.5 ± 2.3); the drifting arms span 466.0 to 468.0, are unordered by lag, and fastest drift is best among them, against our mechanism; the endpoint separation is untested. Self-cosine works only as a binary stationary/drifting indicator. Predictability is not separable from drift, a magnitude account explains frozen-best equally well, and per-condition auxiliary gradient norms were not measured.

PCGrad reaches 461.0 ± 5.0, below un-gated Full (467.99, n=20), where a persistent-conflict account predicts repair. Caveats: I_mag is near zero, so little conflict to remove; the projection was not instrumented; our account predicts neutral, not harmful. Consistent, not discriminating.

Scope: SMAX sits inside the class we meant to claim (co-adapting structured targets reaching a shared actor encoder) yet shows no mechanism (Q1), so membership is necessary but not sufficient and the boundary is unknown.

## W2: linearization loosest mid-training

Fair: justified near convergence, not mid-training; we state it as a limitation. We withdraw the variance ratio as a mid-training signature; 5.58, 1.07 and 3.30 are Final50 statistics, mid-training policy movement was never instrumented, and our only direct drift measurement (late consecutive-policy KL) is near-identical across conditions (2.42 to 2.58, units 1e-3 nats). It also fails both replications at n=20: old stack 1.07 (p=0.88) and jax 0.10.2 1.89 (p=0.175). It appeared at 3.30 (p=0.027) in an interim n=16 snapshot of the latter set and did not survive the remaining four seeds per arm. Canonical 5.58 is the observation under test, and we withdraw the variance-asymmetry claim.

## Q1: secondary pathology on SMAX

SMAX instrumented, old stack, n=5, late-training E[cos]: Full 10.25 ± 0.29 at +0.0062; No-Aux 10.56 ± 0.56 and Stop-grad 10.36 ± 0.14 at 0.000 by construction, so Full is the only measured cell. With no CI on +0.0062 we cannot reject zero-mean noise, and 10.25 against 10.56 was never tested, so there is no established SMAX gap to explain: we detect no effect, and the directional account predicts none. Representation collapse stays untested. SMAX moves into limitations.

## Q2: low Hanabi absolute scores

The Hanabi agent is deliberately minimal, so scores sit far below specialised systems; we intended an ordering claim. It is n=5 and not rerun; n=5 ordering claims are what evaporated on Overcooked at n=20, so it is labelled pending replication. Our runs cannot answer the stronger-baseline question, and our earlier prediction was too strong: nothing implies effect size rises with agent strength, and SMAX is a counterexample. Only the weak form remains: the effect should not vanish where the target co-adapts and reaches the shared encoder.

## What remains

That deficit in Overcooked Asymmetric Advantages alone, magnitude unresolved, under one percent of the roughly 468-point baseline, mechanism evidence only at n=5. Withdrawn: the reported magnitude, the collapse and bimodality framing, the variance asymmetry as a general claim, SMAX as an instance, the between-task cosine-std diagnostic, and the external-validation claim. If the pending replication is prerequisite to your decision, say so.
