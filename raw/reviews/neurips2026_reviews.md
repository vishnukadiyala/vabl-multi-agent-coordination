# NeurIPS 2026 Reviews — Submission29511 (verbatim)

Received 2026-07-24 (pasted by Vishnu from OpenReview). Paper: "When Auxiliary
Losses Fail: Non-Stationary Targets Induce Directional Gradient Noise."

---

## Meta Review — Area Chair y2yo (22 Jul 2026, modified 23 Jul 2026)

Metareview:
This paper identifies a failure mode in cooperative multi-agent RL, attributing
late-training policy-gradient degradation due to auxiliary losses to target
drift among co-learning agents. While reviewers appreciated the practical
relevance of the findings and the proposed gradient-pathway interventions,
significant concerns remain regarding both presentation and core framing.
Specifically, the manuscript is submitted under the "Theory" category, yet the
core evidence is predominantly empirical, with a dense, poorly structured
abstract that overclaims broad applicability across general deep learning while
main effects fail to transfer outside cooperative MARL settings.

Given these presentation deficiencies, the misalignment with the declared
contribution type, and low overall reviewer evaluation, I believe there is
insufficient argument to justify the acceptance based on the initial reviews.

---

## Reviewer fXvf (18 Jul 2026) — Rating 4 (Borderline accept), Confidence 3

Quality 3 / Clarity 3 / Significance 3 / Originality 3

Summary:
This paper investigates late-training performance degradation caused by
auxiliary losses in cooperative multi-agent reinforcement learning and argues
that when auxiliary targets are both task-structured and non-stationary due to
co-adapting teammate policies, their gradients introduce directional noise that
can dominate the diminishing policy gradient near convergence; the authors
validate this mechanism through the VABL framework, target-source controls,
auxiliary-capacity sweeps, cross-environment experiments, and gradient-pathway
interventions, showing that loss-weight annealing, stop-gradient, and
critic-side auxiliary placement can alleviate instability in some environments.

Contribution Type: General.

Strengths:
- Relatively novel explanation of auxiliary-loss failure via target drift and
  temporal variation in gradient direction, distinct from gradient-magnitude
  conflict and capacity limitations.
- Co-learning / frozen-policy / random target comparison thoughtfully designed;
  partially rules out capacity-consumption explanation.
- Theory, gradient-cosine diagnostic, cross-seed variance, and final-performance
  results form a reasonably coherent explanatory chain.
- Remedies (annealing, stop-gradient) simple to implement; clear improvements in
  Overcooked.
- Evaluation covers several cooperative MARL environments and a supervised
  boundary case.

Weaknesses:
- Target-source experiment simultaneously changes label entropy, predictability,
  and state dependence; does not strictly establish non-stationarity alone as
  cause.
- Most key experiments use only five seeds; insufficient for cross-seed
  variance, bimodality, or absence-of-difference conclusions.
- Central theoretical quantities not measured directly; proxied by temporal
  gradient-cosine variability.
- Locally quadratic objective, positive-definite Hessian, approximately
  independent noise are strong assumptions for co-adapting MARL.
- Fixes do not consistently recover performance (SMAX mean gap; critic-side
  placement higher variance).
- "The PDF text layer contains an unrelated hidden prompt-injection
  instruction, which would constitute a serious submission-integrity concern if
  it originated from the official manuscript."

Questions (numbered for rebuttal reference):
1. Narrow Abstract/Intro claims to structured, co-adaptive auxiliary targets
   with gradients through a shared actor encoder; replace "harmless" with
   wording reflecting 5-seed statistical power.
2. Related Work: direct comparison with moving-target learning, dynamic
   teachers, adaptive auxiliary-task weighting, PCGrad, GradNorm; remove the
   hidden prompt-injection text immediately.
3. §3: clarify VABL implementation (parameter sharing, multiple teammates in aux
   head, identity embeddings, visibility masks in gradient pathway).
4. §4: directly estimate Sigma_pi and J_pi; reformulate Prop 1 as precise
   approximation/bound with explicit assumptions and constants.
5. §5.1–5.2: add PCGrad, GradNorm, adaptive aux weighting, and target controls
   matched in label distribution/predictability/state dependence differing only
   in temporal drift.
6. §5.3/§6: more seeds; justify Final50 window; per-seed curves and
   peak-to-final drops; measure consecutive-policy KL to connect drift with
   cosine variability and degradation.
7. §7–8: treat seed count, indirect measurement, SMAX gap, critic-side variance
   as central limitations; avoid implying generalization to all aux-learning
   settings.

Limitations noted: 5 seeds; VABL-style cooperative MARL scope; strong local
assumptions with unmeasured key quantities; interventions inconsistent across
environments.

---

## Reviewer PYCT (06 Jul 2026) — Rating 4 (Borderline accept), Confidence 3

Quality 3 / Clarity 3 / Significance 3 / Originality 3

Summary:
Identifies the mechanism by which constant-weight auxiliary prediction losses
degrade late-stage MARL performance; structured + non-stationary targets inject
directional gradient noise dominating the vanishing policy gradient near
convergence; verified via first-order mechanistic model and ablations across
four MARL benchmarks; mitigated by stop-gradient / weight annealing.

Contribution Type: General: Most submissions will fall into this type.

Strengths:
- Directional (temporal-variance, mean-zero) vs magnitude/persistent-conflict
  distinction is novel; careful positioning vs PCGrad/GradNorm/DG-PG/GAC (§2,
  App A.6). Target-source distinguishing test is an original identification
  method.
- Theory operationalizes directional interference as temporal variance of
  gradient alignment vs magnitude/persistent interference (Eq. 4); isolates a
  signature the gradient-surgery literature does not target.

Weaknesses:
- Claims to explain published systems (BEPAL, Dynamic Belief) but validates only
  its own minimal reconstruction (VABL); external corroboration rests on one
  data point (Zhai et al. Static-Belief, Traffic Junction 73.4% -> 71.9%).
- Linearization valid where effect is weakest: tight late in training, loose
  mid-training where worst cross-seed instability is reported.

Questions:
1. SMAX residual mean gap: secondary pathology (e.g., representation collapse)
   coexisting?
2. Hanabi: relative ordering supports hypothesis but absolute scores low; would
   pathology manifest with same severity under a stronger specialized baseline?

---

## Reviewer yGKw (21 Jun 2026) — Rating 3 (Borderline reject), Confidence 4

Quality 3 / Clarity 3 / Significance 2 / Originality 2

Summary:
Identifies failure mode of aux losses in deep RL/MARL (structured +
non-stationary targets inject directional gradient noise dominating vanishing
policy gradients near convergence). Introduces VABL, first-order model
Sigma_eps ~= J_pi Sigma_pi J_pi^T, validated via target-source test, 8x
capacity sweep, cross-environment studies, three gradient-pathway interventions.

Contribution Type: General: Most submissions will fall into this type.

Strengths:
- Clean mechanistic framing of directional (temporal) interference distinct from
  magnitude imbalance and persistent conflict.
- Simple, insightful linearization tying Sigma_pi and J_pi to Sigma_eps with a
  testable stationarity/pathway/target-ordering triad.
- Practical mechanism-targeted fixes with empirical validation.

Weaknesses:
- i.i.d. noise + local quadraticity assumptions; core predictions not directly
  validated with measured J_pi, Sigma_pi, or eta.
- Threshold condition (Eq. 8) not empirically instantiated; quantitative
  instability boundary uncalibrated.
- Missing PCGrad/GradNorm baselines in this single-auxiliary setting.
- Frozen-policy = argmax of iteration-0 is a strong structural change; soft
  snapshot probabilities or multiple snapshots would give a continuum.
- Lambda sweep coarse and non-monotonic; more points / per-iteration gating /
  adaptive schedules would sharpen theory link.

Questions (numbered):
1. Directly estimate Sigma_pi (consecutive-policy KLs, as in trust-region
   decomposition) and correlate with Final50 degradation and cosine-std across
   seeds and tasks.
2. Soft frozen targets (probability distributions from early snapshots) or
   multiple frozen snapshots to interpolate random -> co-learning.
3. Measure/approximate J_pi by perturbing target-policy logits on logged
   trajectories, observing changes in g_aux.
4. Robustness to different auxiliary definitions (teammate latent states,
   returns, contrastive representations)?
5. PCGrad/GradNorm or direction-aware schedulers in the Full setting: do they
   reduce variance despite I_mag ~= 0?
6. Drift-gated aux controller (binary stop-gradient on cosine-std threshold),
   with window/threshold sensitivity.
7. SMAX: non-zero E[cos] late in training (violating zero-mean assumption) or
   bias in eps explaining the mean gap?
8. Sensitivity to schedules beyond linear annealing (cosine,
   inverse-temperature, adaptive keyed to Sigma_pi proxies)?

Limitations noted: first-order analysis only (no higher-order/cumulative
effects); cooperative-MARL-only scope; interventions are symptomatic
(gradient-path engineering) not root-cause (aux-target design); cosine-std
detects but cannot quantitatively attribute drift contributions.
