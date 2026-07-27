---
tags: [neurips2026, rebuttal, draft]
status: superseded
related: [neurips2026_rebuttal_plan, neurips2026_reviews, rebuttal_evidence_audit]
---

# NeurIPS 2026 Rebuttal Draft v2 — Submission29511

> **SUPERSEDED 2026-07-25.** Live document is `neurips2026_rebuttal_FINAL.md`.
> This draft contains errors caught in adversarial review (unverified NeurIPS
> organizer claim, GradNorm overclaim, drift-gate overclaim, CIFAR mischaracterized,
> "130+ runs"). Kept as history only. Do not submit from this file.

Rewritten 2026-07-25 after the full experiment campaign (R1-R5, R7, n=20
extension, seed reruns; V2-environment verification in flight). Strategy:
counter every stated weakness with data where we have it, concede precisely
where we must, and never assert a canonical number without the fresh-
replication context. `[V2: ...]` slots take the new-environment numbers.
No em-dashes. Every number verified against result JSONs 2026-07-25.

Discussion-period run totals: 130+ new runs on the original hardware
(and a full re-verification under a second numerical stack).

---

## Top-level comment (AC y2yo + all reviewers)

We thank the reviewers for detailed, actionable reviews. During the
discussion period we ran 130+ additional experiments implementing the
requested baselines, measurements, and controls. Summary of what is new:

1. PCGrad and GradNorm baselines (requested by fXvf and yGKw): neither
   mitigates the failure mode, and GradNorm amplifies it substantially,
   exactly as the mechanism predicts (details in responses to both).
2. Direct measurement of Sigma_pi via consecutive-policy KL and a per-task
   gradient direction-stability diagnostic (yGKw Q1/Q3, fXvf Q4/Q6).
3. A soft-snapshot target continuum isolating temporal drift from label
   entropy, predictability, and state dependence (fXvf W1, yGKw W4/Q2).
4. A working drift-gated auxiliary controller (yGKw Q6): the best-performing
   configuration in our entire study.
5. Seeds extended from 5 to 20 on the headline comparison (fXvf W2), with a
   candid recalibration of effect sizes (below).
6. A direct test of the zero-mean noise assumption on SMAX (yGKw Q7):
   the assumption holds.

**On contribution category.** The AC is correct that "Theory" was a poor fit
for a mechanistic-empirical characterization; the submission-form choice was
our error and we accept it. We note all three reviewers evaluated the paper
as Contribution Type: General, so the reviews assess the paper against the
standard it was written to, and we ask that it be judged on that basis.

**On "overclaims broad applicability while main effects fail to transfer
outside cooperative MARL."** We respectfully believe this inverts the paper's
logic, and we own the abstract that made the misreading possible. The
supervised CIFAR-100 null is a predicted boundary of the mechanism, not a
failed transfer: stationary targets should produce no directional
interference, and they do not (d = -0.16, CI crossing zero). The revised
abstract (posted verbatim below) states the conditional scope explicitly.

**Effect-size recalibration (proactive disclosure).** Reviewer fXvf warned
that five seeds are insufficient for variance and bimodality conclusions. We
extended the headline Overcooked comparison to n = 20 per arm and the
reviewer is right in substance: the Full-vs-No-Aux contrast replicates in
direction (gap 1.9 points, Welch p = 0.044, Cohen's d = 0.68) but the
canonical 10-point gap reflected two low-outcome seeds among five. The
phenomenon is best described as a stochastic late-training collapse whose
incidence increases with effective auxiliary weight: 0 collapses in 70+ runs
with the pathway severed or targets stationary; ~9% at lambda = 0.05; 2/5
under PCGrad; 5/5 under GradNorm, which adaptively raises the auxiliary
weight toward gradient-norm parity (~19x ours). [V2: confirmed/updated under
an independent numerical stack.] The revision will report all n = 20 numbers,
the incidence framing, and per-seed curves. We believe this strengthens
rather than weakens the contribution: the dose-response chain from
"no auxiliary gradient" to "norm-matched auxiliary gradient" is monotone in
degradation and is exactly what Sigma_eps-dominance predicts, while the
fixes (annealing, stop-gradient, and the new drift gate) sever it.

**Hidden text in the PDF (fXvf).** Our manuscript and submitted PDF contain
no hidden text. The string the reviewer encountered was inserted into
submission PDFs after the deadline by the NeurIPS organizers' announced
LLM-review detection experiment; it did not originate from us. We separately
commit to deleting Appendix L (visible drafting meta-commentary that should
not have shipped).

**Revised abstract.** [unchanged from v1 draft; four-sentence spine, scope
conditional on structured co-adaptive targets through a shared actor encoder]

---

## Response to Reviewer yGKw (rating 3, confidence 4)

We implemented and ran the majority of your suggested measurements. Point by
point, weaknesses first.

**W1 (assumptions; Sigma_pi, J_pi, eta not measured).** Sigma_pi is now
measured directly: consecutive-policy KL on visited states, logged every
iteration for 35 runs (your Q1 protocol, following trust-region
decomposition). Late-training drift correlates with the cosine-std diagnostic
at rho = 0.60 across aux-ON runs. The zero-mean noise assumption is now
TESTED rather than assumed: late-training E[cos] on SMAX is +0.006 (vs 0.000
for no-aux), i.e. the mean-zero approximation holds where we can measure it
(your Q7). J_pi measurement by logit perturbation (your Q3) requires an
additional instrumentation pass; we commit to it for the revision and have
specified the estimator in Appendix A. We concede eta remains unmeasured.

**W2 (Eq. 8 threshold uncalibrated).** Correct, and the new per-iteration KL
and gradient logs now make calibration feasible; the revision will report the
measured operating point of each environment relative to the threshold. We
have softened the text to present Eq. 8 as an ordering prediction, not a
calibrated boundary.

**W3 (PCGrad/GradNorm missing).** Ran, 5 seeds each, Full setting on
Overcooked AA. PCGrad: Final50 461.0 +/- 5.0 with 2/5 seeds degraded below
460; it does not repair the failure mode, consistent with our claim that the
interference is not persistent conflict (I_mag ~= 0, so there is little for
projection to remove). GradNorm (norm-balancing variant; inverse-training-
rate term dropped because PPO losses can be negative; weight trajectory
logged): weights converge near parity, effectively ~19x our lambda, and all
five seeds degrade (446.2 +/- 15.8, worst 415.4). A method that adaptively
INCREASES auxiliary gradient magnitude worsens the failure monotonically.
This is the dose-response signature of Sigma_eps dominance and, we believe,
the single strongest new piece of evidence for the mechanism. [V2 numbers to
be added.]

**W4 (frozen argmax is a strong structural change).** We ran your suggested
soft-snapshot continuum: auxiliary targets are the full action distribution
of a policy snapshot refreshed every L iterations, L in {1, 25, 100, never},
so all conditions share label smoothness and state dependence and differ only
in drift rate. Endpoints separate in the predicted direction (frozen-soft
470.5 +/- 2.3 vs refreshed-every-iteration 468.0 +/- 1.8; the only collapse
in the continuum occurred in a drifting condition), and the aux gradient's
own direction stability tracks stationarity cleanly: self-cosine 0.974
(frozen) vs 0.79-0.86 (drifting). We are candid that the interior of the
continuum is not resolved at n = 5 (intermediate lags overlap), and say so in
the revision. [V2: continuum re-run under the second stack.]

**W5 (lambda sweep coarse, non-monotone).** We agree an adaptive schedule is
the right object. Your Q6 controller is the constructive answer; see below.

**Q1 (KL vs degradation).** Measured; rho(KL, cosine-std) = 0.60 across 15
aux-ON runs; per-run late KL ~ 2.5e-3 nats. One honest subtlety the
measurement surfaced: consecutive-policy KL measures LIVE-policy drift, which
is nearly identical across conditions; target drift equals it only when
targets come from the co-learning policy (Full) and is zero by construction
for frozen targets. The revision states this explicitly.

**Q2 (soft/multiple snapshots).** Ran; see W4.

**Q3 (J_pi).** Committed for revision; estimator specified.

**Q4 (other aux definitions).** Not run; stated as scope in Limitations.

**Q5 (PCGrad/GradNorm in Full).** Ran; see W3. Answer to your specific
question: no, they do not reduce variance despite I_mag ~= 0; GradNorm
increases it five-fold.

**Q6 (drift-gated controller).** Implemented exactly as you specified:
stop-gradient on the belief pathway when rolling cosine-std over window
W = 10 exceeds tau. tau = 0.10 (gate active ~70% of iterations): 471.9 +/-
1.7, the best configuration in our entire study, exceeding No-Aux itself.
tau = 0.15 (~30% active): 467.2 +/- 1.1. Window/threshold sensitivity is
reported in the revision. We now cite this as the mechanism-motivated
controller the analysis implies, approximating the DG-PG closed form.

**Q7 (SMAX E[cos]).** Measured on fresh instrumented runs: late-training
E[cos] = +0.006 for Full (0.000 no-aux, 0.000 stop-grad). The zero-mean
assumption holds; directional bias does not explain the residual SMAX mean
gap. In the same fresh runs the SMAX gap itself is small (Full 10.25 +/-
0.29 vs No-Aux 10.56 +/- 0.56). We therefore no longer claim SMAX as a
strong instance of the pathology; it is a small-effect environment, and the
revision says so.

**Q8 (other schedules).** Not run; the drift gate subsumes the motivation.

**On Significance/Originality = 2.** With respect, we ask you to weigh the
discussion-period additions: the paper now contains the first direct
measurement chain (target drift -> per-task gradient direction stability ->
outcome) for auxiliary-loss failure in co-adaptive settings, a falsified-
alternatives table (capacity: falsified by 8x sweep; persistent conflict:
falsified by PCGrad; magnitude imbalance: falsified in reverse by GradNorm),
and a working mechanism-derived controller you proposed and we validated.

---

## Response to Reviewer fXvf (rating 4, confidence 3)

**W1 (target-source confound).** Countered with the matched continuum you
requested (Q5): soft-distribution targets matched in label smoothness and
state dependence across all conditions, varying only temporal drift. See
yGKw W4 response for numbers. The original hard-target comparison remains in
the paper as the outcome-level test; the continuum is the confound-free
identification.

**W2 (five seeds insufficient).** You were right in substance, and we did
the work rather than argue: n = 20 per arm on the headline comparison. The
direction replicates (p = 0.044, d = 0.68); the magnitude at n = 5 was
inflated by two low seeds; and the phenomenon is properly described as
stochastic collapse whose incidence scales with effective auxiliary weight
(0 in 70+ shielded runs; ~9% at lambda = 0.05; 2/5 PCGrad; 5/5 GradNorm).
All claims in the revision are restated at n = 20 with this framing, and
per-seed curves plus peak-to-final drops are added [appendix ref]. We regard
this reframing as a direct product of your review.

**W3 (theoretical quantities proxied).** Sigma_pi now measured
(consecutive-policy KL, every iteration, 35 runs; rho = 0.60 with the cosine
diagnostic). Zero-mean noise assumption tested on SMAX: E[cos] ~= 0, holds.
New per-task direction-stability diagnostic (self-cosine) added; it cleanly
separates stationary from drifting targets under matched soft labels (0.974
vs 0.79-0.86). J_pi estimation is specified and committed for the revision.

**W4 (strong assumptions).** Partially discharged by measurement (zero-mean:
tested, holds on SMAX; drift linearization: the KL measurements are
model-free and cover mid-training). Remaining assumptions (local
quadraticity, curvature stationarity) are now stated as explicit conditions
of Proposition 1 with an error term, per your Q4.

**W5 (fixes inconsistent; SMAX gap; critic-side variance).** Three updates.
(i) The new drift-gated controller recovers best-in-study performance on AA.
(ii) On SMAX, fresh instrumented runs show the gap itself is small (10.25 vs
10.56, overlapping intervals) and E[cos] ~= 0 rules out directional bias as
its cause; we no longer present SMAX as a strong pathology instance.
(iii) Critic-side placement is presented as a qualitatively different regime,
with its variance stated, not spun.

**W6 (hidden prompt-injection text).** Our manuscript contains no hidden
text; the flagged string was inserted post-deadline by the NeurIPS
organizers' announced LLM-review detection experiment. We have verified our
source and submitted PDF are clean and flagged the matter to the AC. We are
also deleting Appendix L, whose visible drafting meta-commentary should not
have shipped and may have contributed to the concern. We appreciate you
raising it.

**Q1/Q7 (claim narrowing, limitations).** Accepted in full; revised abstract
in the top-level comment; Limitations now leads with seed count and effect
size, measurement indirectness (now partially direct), the SMAX small-effect
status, and critic-side variance.

**Q3 (Section 3 clarity).** Parameters are shared across agents with
identity embeddings; the auxiliary head predicts each visible teammate's
next action from the attended belief, per-teammate losses averaged;
visibility masks gate both attention and the auxiliary loss, so masked
teammates contribute no auxiliary gradient. Added verbatim to Section 3.

**Q2 (related work).** Added direct positioning vs moving-target learning,
dynamic teachers, adaptive auxiliary weighting, PCGrad, GradNorm, with an
interference-model table (magnitude / persistent-direction / temporal-
directional) locating each method.

**Q6 (Final50 window; per-seed curves; KL).** Final50 window justified by
eval cadence with a sensitivity check at Final25/Final100 [appendix ref];
per-seed curves added; consecutive-policy KL measured as above.

---

## Response to Reviewer PYCT (rating 4, confidence 3)

**W1 (validates only VABL; single external data point).** We concede the
scope directly: VABL is a minimal reconstruction BY DESIGN, because
isolating the mechanism requires removing co-varying design choices of
published systems; the Zhai et al. Static-Belief delta is corroboration, not
replication. We commit to a BEPAL replication for the camera-ready. The
revision states exactly this.

**W2 (linearization tight where effect is mildest).** A fair and
well-observed limitation, now stated explicitly in Section 4. Two mitigants
from the new measurements: the consecutive-policy KL instrument is
model-free and covers mid-training, and the discussion-period data
recharacterizes the pathology as stochastic collapse, for which the
linearized theory supplies the noise-dominance condition rather than a
quantitative trajectory model. The theory's role is scoped accordingly.

**Q1 (SMAX secondary pathology).** Your instinct appears correct. We
measured late-training E[cos] on fresh instrumented SMAX runs: +0.006,
effectively zero, so directional gradient bias does NOT explain the residual
mean gap; and the gap itself is small in fresh runs (10.25 +/- 0.29 vs 10.56
+/- 0.56). Whatever remains is a secondary, non-directional effect;
representation collapse is the natural candidate and we state it as an open
hypothesis with the diagnostic we would run (feature-rank tracking).

**Q2 (Hanabi absolute scores).** The Hanabi agent is deliberately minimal;
the claim is the ordering and effect size (d = +2.81), not absolute play
strength. Whether a specialized agent shows equal severity is genuinely
open: the mechanism predicts severity scales with target drift and pathway
sensitivity, and stronger agents may reduce late-training drift while larger
auxiliary heads increase pathway sensitivity. Stated as a prediction; a
stronger-baseline replication is camera-ready work.

---

## Remaining placeholders

- [V2: ...] slots: filled when the second-stack verification completes
  (in flight; AA n=20 both arms, frozen n=10, continuum, SMAX).
- Appendix refs for per-seed curves and Final50 sensitivity: assigned during
  revision assembly.
- J_pi estimator appendix text: drafted with Atiq before submission.
