---
tags: [neurips2026, rebuttal, synthesis, ready-to-post, final]
status: active
related: [VERIFIED_NUMBERS, rebuttal_evidence_audit, neurips2026_reviews]
---

# NeurIPS 2026 Submission 29511: Final Responses (2026-07-27)

Final rewrite integrating all round-2 experiments (J_pi measurement, auxiliary
task variants, random-gate control, schedules, representation rank, EMA-
distilled targets). Written for first-time readers: no internal codenames;
every experimental setup is described in the sentence that uses it. Every
number traces to VERIFIED_NUMBERS.md. Labels follow the pre-registered rule:
"significant" only for 20-seed bootstrap-CI results; 5-seed results are
"directional." Zero em-dashes.

Posting order: the three public reviewer responses first, then the
confidential comment to the AC.

---

# Response to Reviewer yGKw

Thank you for a review that functioned as an experimental program. We ran new
experiments addressing every question, including Q3's measurement and two
auxiliary-task variants for Q4, and your limitation about root cause versus
symptom led us to build and validate a target-design intervention. Where a
requested analysis remained incomplete or inconclusive, we narrow the
corresponding claim explicitly below. Each answer starts from the reasoning
we believe motivated the question.

**What the new experiments establish.** The paper's central comparison is the
Full configuration (attention plus a teammate-action-prediction auxiliary
loss at weight lambda = 0.05) against No-Aux (identical network and training,
auxiliary loss disabled) on Overcooked Asymmetric Advantages. We extended
this comparison from 5 to 20 seeds per arm and ran it twice: once in the
submission's exact software environment, and once in an independently rebuilt
environment with newer numerical libraries, to check that the result is not
an artifact of one software stack. The deficit replicates in both: 1.92
points, bootstrap 95% CI [0.17, 3.66], p = 0.044 (significant but marginal)
in the submission environment, and 3.47 points, CI [1.68, 5.37], p = 0.00098
(significant) in the rebuilt one. A further rerun using the exact submitted
code revision and the original random seeds gives a 2.56-point gap
(directional, n = 5). Across more than 230 full-length runs completed during
this discussion period, no run of any configuration that severs the auxiliary
gradient pathway or holds targets stationary ever finished below 460, while
4 of 55 plain Full runs did (the 460 threshold was selected after seeing the
data, so these counts are reported descriptively, never with a p-value).

**W1 and Q1, Q3 (the theory's quantities were unmeasured).** The concern, as
we read it: a first-order model whose central quantities are never measured
is unfalsifiable decoration. We agree, and this is where the discussion
period moved the paper most. Three of the four quantities are now measured.
(1) Target-policy drift: consecutive-policy KL logged at every iteration;
late-training drift is 2.42 to 2.58 x 10^-3 nats across all conditions,
confirming that policy movement persists in the regime where the deficit is
measured. Two honest caveats: this measures the live policy's drift, which
equals target drift only in the Full condition, so it validates a premise
rather than attributing the effect; and we have not yet completed the
cross-seed and cross-task correlation you requested between this KL, the
cosine-variability measure, and Final50 degradation, so we treat the KL
measurement as evidence that drift persists, not as quantitative validation
of the predicted scaling. (2) Pathway sensitivity, your Q3, with
pre-registered criteria: we perturb the auxiliary targets (each target
flipped to a random action with probability eps) and measure the relative
change in the auxiliary gradient. The response scales linearly with eps
(ratio between eps = 0.2 and eps = 0.05 responses: 3.95, inside the 3 to 5
band we pre-registered as supporting a first-order sensitivity model), and
severing the encoder pathway with stop-gradient reduces the response by
roughly 40 percent (Full 2.47 versus stop-gradient 1.52 at eps = 0.1), with
non-overlapping per-seed ranges at n = 5. The pathway the model posits is
real, linear in the measured range, and carried substantially by the shared
encoder. (3) The zero-mean noise assumption: tested on SMAX, late-training
mean gradient cosine is +0.0062, so we detect no substantial directional
bias in this measurement (your Q7); a point estimate does not establish the
assumption, and the revision reports it with its uncertainty. The learning-rate coupling eta remains unmeasured; that is now
the only unmeasured quantity, and the revision says so.

**W2 (Eq. 8 is uncalibrated).** Correct. Eq. 8 is restated as an ordering
prediction under explicit local assumptions, not a calibrated boundary. The
new per-iteration drift and gradient logs make calibration feasible and it is
scheduled for the revision.

**W3 and Q5 (PCGrad and GradNorm).** The reasoning we believe motivated this:
if the interference were ordinary persistent task conflict, standard gradient
surgery should repair it. It does not: PCGrad reaches 461.0 +/- 5.0
(directional, n = 5), no better than un-gated Full. Because projection
activity was not instrumented, we do not claim this as positive evidence for
our account, only as a failed repair by the conflict-targeting method.
GradNorm, which adaptively raised the auxiliary weight to roughly 19 times
our lambda in pursuit of gradient-norm parity, degrades all five seeds
(446.2 +/- 15.8): damage from auxiliary magnitude, not a discriminating
result. To separate magnitude from drift at that weight, we ran a
weight-by-stationarity control at lambda = 0.95: with drifting targets,
440.5 +/- 14.6 with a learn-then-hold curve; with frozen targets, 337.5 +/-
67.4, impaired from the first fifth of training because predicting a frozen
random-policy snapshot at dominant weight impedes learning from the start.
So at high weight, damage does not require drift, and we scope the
directional account to the low-weight regime the paper actually studies. At
that operating point the stationarity contrast is the operative one:
drifting targets show the replicated deficit and all four sub-460 plain-Full
runs, while frozen or gradient-shielded configurations show neither in 80
runs.

**W4 and Q2 (the frozen-argmax control changes too much).** We ran the
soft-snapshot continuum you specified: all arms predict a snapshot policy's
full action distribution on the current state, differing only in how often
the snapshot refreshes (every 1, 25, or 100 iterations, or never). Returns:
468.0 +/- 1.8, 466.0 +/- 3.3, 466.8 +/- 5.6, and 470.5 +/- 2.3 respectively
(all directional, n = 5). The stationary endpoint is best and its auxiliary
gradient direction is far more stable between iterations (self-cosine 0.974
against 0.79 to 0.86 for the drifting arms), but the interior does not order
with refresh rate, and predictability remains confounded with drift. We
present this as endpoint-consistent evidence, not as isolation of
non-stationarity.

**Q4 (other auxiliary definitions).** We tested two new variants with a
separate prediction head: predicting each teammate's next belief vector (a
structured, drifting target) and reconstructing the agent's own observation
(a structured, stationary target). Neither shows a deficit at n = 5 (471.85
+/- 1.43 and 472.95 +/- 1.26, both at or above the No-Aux level). This
result went against our pre-registered prediction for the drifting variant,
and we report it as such. Its interpretation is limited by a caveat we
stated before running: these variants use mean-squared-error losses whose
gradient scale was not matched to the action-prediction loss, so the test
cannot distinguish "the deficit is specific to the action-prediction
formulation" from "these auxiliary gradients were too small to matter." The
demonstrated scope of the phenomenon is therefore next-action prediction
through the shared encoder, and the revision states exactly that.

**Q6 (drift-gated controller).** Implemented to your specification:
stop-gradient on the auxiliary pathway whenever the rolling standard
deviation of the gradient cosine exceeds a threshold. At tau = 0.10 (gate
active about 70 percent of iterations) it reaches 471.90 +/- 1.93, the
strongest auxiliary-on result in the study, though not significantly above
No-Aux (p = 0.097). You would reasonably ask whether that is just duty
cycle. We ran the matched control: gating randomly with the same 70 percent
probability yields only 466.72 +/- 2.52, at un-gated Full level. Gating at
the moments your signal selects outperforms gating randomly at matched duty
cycle (directional, n = 5). Scope disclosure: we evaluated two thresholds
(tau = 0.10 and tau = 0.15) with a single rolling-window size of 10
iterations, plus the matched-duty random control; threshold and window-size
sensitivity beyond these points remains untested, so we present this as a
proof-of-concept controller rather than a robustness result. It also
sharpens an honest distinction: the between-condition version of the cosine
diagnostic failed and is withdrawn (details in the corrections paragraph),
but the within-run timing information in the same signal is demonstrably
useful, which is precisely what your controller exploits.

**Q8 and W5 (schedules beyond linear; non-monotonicity).** Three additional
schedules, five seeds each: cosine annealing 471.65 +/- 1.54, exponential
decay 472.99 +/- 3.47 (the best arm in this round), and an adaptive
controller keyed to the measured drift as you suggested, 468.38 +/- 1.77.
No run of any schedule finished below 460. The mitigation family is robust
to schedule choice. The lambda non-monotonicity you flagged remains an open
observation and is listed as such.

**Your limitation on root cause versus symptom.** You distinguished removing
drift at the target source from blocking its gradient path, and asked for a
target design scheme. We built one: EMA-distilled targets, where the
auxiliary head predicts the soft actions of an exponential-moving-average
copy of the policy rather than the live co-learning policy, the
target-network idea applied to auxiliary learning. With slow averaging
(decay 0.995) it reaches 469.88 +/- 3.43, No-Aux-level performance while
retaining the auxiliary task; a faster average (decay 0.99) gives 466.69 +/-
3.71, at the un-gated Full level; both arms have zero sub-460 runs
(directional, n = 5). The designed family now spans live targets (full
drift), EMA (slowed), periodic snapshots (piecewise), and frozen (zero
drift); its low-drift end consistently avoids the deficit, and the
directional difference between the two tested averaging rates is consistent
with rate dependence. The averaging rate inherits the
drift-versus-relevance tradeoff the paper analyzes, which is exactly why the
analysis matters to the design.

**Corrections we made proactively.** Larger samples falsified three of the
submission's claims and we removed them: the variance asymmetry between Full
and No-Aux (fails at 20 seeds in both environments), the bimodality framing,
and the between-condition interpretation of the cosine-variability
diagnostic (direct measurement reversed it). The submitted 10.24-point gap
was inflated by two rare low-outcome runs among five seeds; the calibrated
effect is 2 to 3.5 points, and a rerun under the exact submitted code and
seeds reproduces non-collapsed seeds while neither low run recurs,
demonstrating those were rare stochastic events rather than typical
behavior. SMAX moves to limitations: its small gap is unstable across
batches and shows no directional-bias signature. These corrections came from
our own falsification tests, most of them suggested by you, and what
survived them is stated below.

**What the revised paper claims.** In cooperative MARL with
teammate-action-prediction auxiliaries through a shared actor encoder, a
constant-weight auxiliary loss produces a small but replicated
final-performance deficit and a rare low-outcome tail that never appears
when the pathway is severed or the targets are stationary; the auxiliary
gradient's sensitivity to target changes is measurable, linear, and carried
by the shared encoder; and the deficit is removed by weight scheduling, by
gating on a measured timing signal, or at the target level by distillation
through a slow moving average. We believe this is a narrower and
substantially better-supported contribution than the submission, and that
the majority of the improvement traces directly to your review.

---

# Response to Reviewer fXvf

Your review identified the two weaknesses that mattered most, statistical
power and the confounded target-source comparison, and the discussion period
was organized around them. Each answer below begins from the concern we
understood behind the question.

**W2 and Q6 (five seeds cannot support these conclusions).** You were right,
and we answered with scale rather than argument. The headline comparison
(Full: attention plus teammate-action-prediction auxiliary at lambda = 0.05;
No-Aux: identical training without the auxiliary term; Overcooked Asymmetric
Advantages) now stands at 20 seeds per arm, run twice: in the submission's
exact software environment, gap 1.92 points, bootstrap 95% CI [0.17, 3.66],
p = 0.044, Cohen's d = 0.66 (significant but marginal); and in an
independently rebuilt environment with newer numerical libraries, gap 3.47,
CI [1.68, 5.37], p = 0.00098, d = 1.14 (significant). A rerun under the
exact submitted code revision and original seeds gives 2.56 (directional,
n = 5). The evaluation window you asked us to justify does not drive the
result: measured over the final 25, 50, 100, and 200 iterations the gap is
1.78, 1.92, 2.45, and 3.70 in the submission environment and 3.61, 3.47,
3.05, and 3.29 in the rebuilt one, so the reported window is conservative.
Peak-to-final drops are equal between Full and No-Aux in the submission
environment (4.77 +/- 2.47 versus 4.82 +/- 1.83), which we report as a null.
Your statistical requests are adopted wholesale: bootstrap intervals, Welch
tests, and pooled-standard-deviation effect sizes on every 20-seed contrast,
per-seed curves in the revision, and 5-seed mechanism results labeled
directional throughout.

The larger samples also performed the falsification five seeds could not:
the submitted 10.24-point gap came from two rare low-outcome seeds (the
calibrated effect is 2 to 3.5 points), the variance asymmetry fails at 20
seeds in both environments and is withdrawn along with the bimodality
framing, and under a threshold chosen after seeing the data (and therefore
reported descriptively, never with a p-value), 4 of 55 plain Full runs
finished below 460 against 0 of 80 across every configuration with the
pathway severed or targets stationary.

**W1 and Q5 (the target-source experiment changes several things at once).**
The concern: our frozen-target control altered label entropy and
predictability along with drift, so non-stationarity was not isolated. We
ran the matched control you specified: every arm predicts a snapshot
policy's soft action distribution, and only the snapshot refresh interval
varies. The stationary endpoint performs best (470.5 +/- 2.3 versus 468.0
+/- 1.8 for per-iteration refresh; directional, n = 5) and its auxiliary
gradient direction is markedly more stable (self-cosine 0.974 versus 0.79 to
0.86), but the drifting interior does not order with refresh rate and
predictability remains confounded with drift. We therefore narrow the claim
exactly as your question implies we must: the endpoints are consistent with
a stationary-versus-drifting contrast; non-stationarity alone is not
isolated. Your requested baselines are also run: PCGrad does not repair the
deficit (461.0 +/- 5.0), and GradNorm, which raised the auxiliary weight
about 19-fold toward norm parity, degrades all seeds (446.2 +/- 15.8), a
magnitude effect we do not count as evidence for directional interference.

**W3 and Q4 (theoretical quantities are proxied, assumptions strong).**
Three of the four central quantities are now measured directly rather than
proxied: target-policy drift via consecutive-policy KL at every iteration
(2.42 to 2.58 x 10^-3 nats late in training across conditions); pathway
sensitivity via a pre-registered target-perturbation experiment showing a
linear auxiliary-gradient response (linearity ratio 3.95 against a
pre-stated 3 to 5 band) that drops by about 40 percent when the encoder
pathway is severed; and the zero-mean noise assumption, tested on SMAX,
where we detect no substantial directional bias (late mean cosine +0.0062;
a point estimate, reported with its uncertainty in the revision). The between-condition cosine
diagnostic reversed under this direct measurement and is withdrawn.
Proposition 1 is restated as an approximation with explicit
local-quadraticity and noise assumptions. The theory's role in the revision
is to organize testable predictions, several of which the new measurements
now pass, and one of which they falsified.

**W5 (fixes inconsistent across environments).** Two updates. First, the
mitigation family is broader, and all tested mitigation arms avoided
sub-460 outcomes in these five-seed experiments: linear, cosine,
exponential, and drift-adaptive schedules, stop-gradient, a drift-gated
controller (471.90 +/- 1.93 at its stronger setting, and gating on the
measured signal beats random gating at matched duty cycle, 466.72 +/- 2.52),
and an EMA-distilled target design (469.88 +/- 3.43) all avoid the
low-outcome tail entirely, zero sub-460 runs among them. Second, SMAX: its
fresh gap is small, unstable across batches (it reversed sign between two
5-seed batches), and shows no directional-bias signature, so SMAX and the
critic-side variance observation move to limitations, scoped as you asked.

**Q2, first half (related work).** The revision adds direct comparisons with
moving-target and dynamic-teacher methods, adaptive auxiliary-task
weighting, PCGrad, and GradNorm, distinguishing target stabilization, loss
balancing, and persistent-conflict correction from the temporal directional
interference studied here; the PCGrad and GradNorm rows of that comparison
are now backed by the experiments above rather than citation alone.

**Q3 (implementation details).** Agents share parameters and use learned
identity embeddings. The auxiliary head predicts each visible teammate's
next action from the attended belief representation, with per-teammate
losses averaged; visibility masks gate both attention and the auxiliary
loss, so unobserved teammates contribute no gradient. This paragraph is
added verbatim to Section 3.

**W6 and Q2 (the text in the PDF layer).** On 2026-07-24 we extracted the
full text layer of our LaTeX source and of both PDFs we uploaded and found
no hidden text of any kind; we can supply the extraction command, its
output, and SHA-256 checksums of our uploaded files. The only occurrence of
the word "prompt" anywhere in our source is visible text in Appendix L,
drafting meta-commentary about how the paper reads. It is not hidden and not
an instruction, but it should not have been in a submission and will be
removed. We make no claim about the provenance of the text you observed and
have asked the program chairs to compare the review-system copy against our
uploaded file. Thank you for raising it; integrity flags deserve engagement.

**What the revised paper claims.** A replicated, correctly calibrated
final-performance deficit from co-adaptive action-prediction auxiliaries in
cooperative MARL, with the pathway measured, the rare failure tail
characterized, simple alternative explanations ruled out by controls you and
your fellow reviewers specified, and a family of mitigations, including a
target-level design, that eliminate it. The claims are scoped to what the
evidence supports, and the evidence is now roughly four times the
submission's.

---

# Response to Reviewer PYCT

Thank you for engaging with the identification logic of the paper; both of
your questions anticipated where the new measurements ended up mattering
most.

**Stronger empirical foundation.** The central comparison (Full: attention
plus teammate-action-prediction auxiliary at lambda = 0.05; No-Aux:
identical without the auxiliary; Overcooked Asymmetric Advantages) now
stands at 20 seeds per arm in two software environments: gap 1.92, 95% CI
[0.17, 3.66], p = 0.044 in the submission's environment, and 3.47, CI
[1.68, 5.37], p = 0.00098 in an independently rebuilt one, plus a 2.56-point
gap in a rerun under the exact submitted code and seeds (directional,
n = 5). We recalibrated the headline effect from the submitted 10.24 points
(inflated by two rare low-outcome seeds) to 2 to 3.5 points, and withdrew
the variance-asymmetry and bimodality claims that 20 seeds falsified.

**W1 (the paper validates its own reconstruction, not the published
systems).** We agree and the revision adopts your distinction. BEPAL and
Dynamic Belief instantiate the design pattern the paper studies,
teammate-prediction auxiliaries coupled to shared actor representations, but
we do not claim to have established the mechanism inside them. VABL is the
controlled testbed in which target source and gradient path can be
manipulated independently; the Zhai et al. Static-Belief comparison is
external corroboration, not replication; a replication attempt on a
published system is planned for the camera-ready, and if it cannot be
completed to the evidential standard of this response, the claim scope
remains exactly as stated here.

**W2 (the linearization is tight where the effect is mildest).** Accepted
and now stated in Section 4. The first-order model is scoped to late
training, where the final-performance claims live. Two additions partially
compensate: the drift measurement (consecutive-policy KL, logged every
iteration) is model-free and covers all of training, and the model's
late-training sensitivity structure now has direct support: perturbing the
auxiliary targets produces a linear gradient response (pre-registered
linearity band hit) that drops by about 40 percent when the encoder pathway
is severed.

**Q1 (does a secondary pathology such as representation collapse coexist on
SMAX?).** We took your hypothesis seriously enough to measure it in both
environments. Two results. First, we detect no substantial directional bias
where you asked: late-training mean gradient cosine on SMAX is +0.0062, a
point estimate that gives the directional-bias account no support without
our claiming the assumption is established. Second, representation collapse
is disfavored:
the effective rank of the belief representation is higher, not lower, under
the auxiliary loss (SMAX: 35.2 +/- 2.7 versus 30.4 +/- 2.6 for No-Aux;
Overcooked shows the same direction, 21.7 versus 16.1). The auxiliary task
enriches the representation while degrading the policy, which sharpens the
puzzle your question pointed at rather than closing it. We also note the
SMAX gap itself is unstable: it reversed sign between two independent 5-seed
batches (10.25 versus 10.56, then 10.57 versus 10.49), so SMAX moves to
limitations as an environment where no stable effect is detectable, and your
instinct that something other than our mechanism governs it stands.

**Q2 (would a stronger Hanabi agent show the same severity?).** The Hanabi
experiment isolates ordering with a deliberately minimal agent (effect size
d = +2.81 at n = 5, labeled preliminary), and its absolute scores are not
competitive with specialized agents. For a stronger agent the prediction is
genuinely open, and your question now has quantitative footing: severity
should scale with the product of target drift, which stronger agents
plausibly reduce late in training, and pathway sensitivity, which we can now
measure directly (the perturbation-response method above) and which richer
belief architectures plausibly increase. We state this as a testable
prediction rather than a claimed implication.

**What the revised paper claims.** A replicated failure pattern in a
controlled setting that instantiates a common MARL design; direct
measurements replacing what were previously proxies, including the pathway
sensitivity your Q2 turns on; the rival explanation your Q1 proposed tested
and disfavored, with the residual SMAX puzzle honestly retained; and
practical mitigations, including a target-level design, that eliminate the
failure tail. The corrections in the revision (effect size, variance claims,
one diagnostic) were made proactively when larger samples falsified them,
which we believe is how the discussion period is supposed to work.

---

# Author to AC confidential comment (y2yo)

We write briefly and confidentially on three matters. The full evidence is
in our public responses to the three reviewers, each self-contained.

**1. The discussion-period record.** We understand the metareview
necessarily reflects the initial reviews. Since it was written we have
completed more than 230 full-length runs covering every experiment the
reviewers requested, with new experiments addressing all eight of reviewer
yGKw's questions (incomplete analyses are flagged as such in the responses).
The headline comparison now stands at 20 seeds per arm in two software
environments plus a rerun under the exact submitted code and seeds:

| Evaluation | n/arm | Full | No-Aux | No-Aux minus Full |
|---|---|---|---|---|
| Submitted five-seed experiments | 5 | 463.21 +/- 8.55 | 473.45 +/- 3.62 | 10.24 |
| 20 seeds, submission's environment | 20 | 467.99 +/- 2.96 | 469.91 +/- 2.86 | 1.92 [0.17, 3.66], p=0.044 |
| 20 seeds, rebuilt environment | 20 | 468.14 +/- 3.49 | 471.62 +/- 2.54 | 3.47 [1.68, 5.37], p=0.00098 |
| Exact submitted code and seeds | 5 | 468.50 +/- 1.79 | 471.06 +/- 1.29 | 2.56, directional |

The direction holds everywhere; the calibrated effect is 2 to 3.5 points
rather than the submitted 10.24, which two rare low-outcome runs inflated
(the exact-code rerun reproduces normal seeds closely while neither low run
recurs). We withdrew what larger samples did not support: the
variance-asymmetry claim, the bimodality framing, one diagnostic, and SMAX
as supporting evidence. Alongside the corrections, the discussion period
added affirmative results: the model's pathway sensitivity is now measured
directly (linear response, pre-registered criteria met, reduced ~40 percent
when the pathway is severed), the reviewer-proposed gating controller beats
a matched-duty random control, four weight schedules and an EMA-distilled
target design all eliminate the failure tail, and the representation-
collapse alternative was tested and disfavored in both environments. No new
positive claim enters the revision that was not tested this period; every
change to reviewed claims is a calibration or narrowing.

**2. Category and scope.** We agree Theory was the wrong category (all three
reviewers marked General) and that the submitted abstract overclaimed. The
revision presents the work as an empirical characterization organized by a
first-order model, scoped to structured, co-adaptively generated
teammate-prediction targets whose gradients propagate through a shared actor
encoder.

**3. The PDF text layer (confidential; submission integrity).** Reviewer
fXvf reported a hidden prompt-injection instruction in the PDF text layer.
On 2026-07-24 we extracted the full text layer of our LaTeX source and of
both PDFs we uploaded and found no hidden text of any kind; we can provide
the extraction command, its output, and SHA-256 checksums of our uploads.
The only occurrence of the word "prompt" in our source is visible text in
Appendix L, drafting meta-commentary being removed. We ask that the program
chairs compare the review-system copy of our submission against our upload.
We make no claim about the provenance of whatever the reviewer observed, and
our public reply is factual and without speculation.

We respectfully ask that the final recommendation weigh the post-response
record, including reviewer engagement with the new experiments, rather than
the initial-review snapshot. The calibrated paper (a replicated failure
pattern in a common MARL design, its pathway directly measured, rival
explanations tested, and validated mitigations including a target-level
design) is, we believe, a sound NeurIPS contribution, and a more credible
one for reporting the smaller effect.

---

# Before posting (checklist)

1. Atiq reads all four pieces plus VERIFIED_NUMBERS.md sections 0 and 3e-3i.
2. PDF-provenance stance is verified-facts-only. If an official NeurIPS
   announcement of a reviewer-copy experiment exists, cite it; otherwise do
   not assert provenance.
3. Number sweep against VERIFIED_NUMBERS.md after any edit.
4. Em-dash scan (currently zero).
5. Post reviewer responses first, confidential comment last.
6. Run count "more than 230" is the verified floor (239 by per-batch file
   count); do not round up to 250.
