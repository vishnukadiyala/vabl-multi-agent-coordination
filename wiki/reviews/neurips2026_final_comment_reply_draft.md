---
tags: [neurips2026, rebuttal, yGKw, draft]
status: archived
related: [neurips2026_final_comment_yGKw, r15_r16_final_comment, VERIFIED_NUMBERS]
---

# Final Reply to yGKw — POSTED to OpenReview 2026-08-03 (v5, corollary version)

Posted text = v5: impact-first opening, R16 representation-drift paragraph,
MPE Sigma_pi withdrawal + Corollary (product-form boundary), J_pi described
as in progress (R17 seed variance made the n=2 numbers unquotable: per-seed
jpi_rel_eps0.1 = 1.26 / 4.12). No R17 numbers were posted.

Strategy: defend Proposition 1 (three-factor product), concede exactly two
textual items (appendix "Sigma_eps = 0" sentence; intro "slow symmetric
drift" clause). R16 framed as a passed adversarial test, not a concession.
R15's Sigma_pi falsification scoped to the intro shorthand, not the theory.
R17 (J_pi on MPE) running; upgrade the marked sentence if it lands before
posting. All numbers verified (VERIFIED_NUMBERS sections 3g, 5, 10).

---

Thank you for the close reading. We ran new measurements against both
points. The first identifies a term our model does not carry; measurement
shows it is real but common-mode and non-causal. The second led us to
measure the quantity behind our boundary sentence; the measurement shows
the sentence attributed the boundary to the wrong factor, which we correct
below.

**On representation drift under fixed labels.** The observation is correct:
fixing the auxiliary labels does not make the auxiliary gradient
stationary, because the shared encoder continues to update under the RL
loss. In the model, Sigma_eps = J_pi Sigma_pi J_pi^T is by construction the
target-drift component of auxiliary-gradient variability, and the
experimental contrasts are designed to difference the remaining terms out:
representation drift is present in every arm, including No-Aux and frozen.
We have now measured it rather than assuming it. Rerunning Full,
frozen-target, and No-Aux (5 seeds each, 10M steps, Overcooked AA) while
tracking the aux head's input representation on a fixed probe batch of 1024
states, late-training feature self-cosine is 0.9908 +/- 0.0021 (Full),
0.9851 +/- 0.0012 (frozen), and 0.9915 +/- 0.0009 (No-Aux). Representation
drift is common-mode, and the frozen arm, which drifts slightly more in
representation space than Full, reproduces neither the deficit nor the
low-outcome tail (frozen 470.6 +/- 1.1 vs Full 466.8 +/- 6.0 in these
reruns). Label drift, by contrast, does separate the arms: under matched
soft labels the auxiliary gradient's iteration-to-iteration self-cosine is
0.974 with frozen targets against 0.79 to 0.86 with drifting targets. So
the term you identify exists, is now measured, and is shared by conditions
with and without the pathology, which is exactly what licenses treating the
frozen condition as the Sigma_pi = 0 contrast. The camera-ready will make
the decomposition explicit (a target-drift term, modeled, plus a
representation-drift term, measured and common-mode) and will correct the
appendix sentence "Sigma_eps = 0" to "the target-drift component of
Sigma_eps vanishes."

**On the MPE simple_spread boundary.** Here your comment exposed a real
error, and we want to be precise about what the error is. Proposition 1's
instability condition is a three-factor product, alpha^2 tr(Sigma_eps) /
lambda_min(H) with Sigma_eps = J_pi Sigma_pi J_pi^T; the introduction's
boundary sentence compressed this to one factor ("slow symmetric drift,"
i.e., small Sigma_pi), with no derivation, as you note. We have now
measured that factor. Instrumenting MPE simple_spread with the same
consecutive-policy KL measurement used on Overcooked (5 seeds per arm;
reruns reproduce the original returns), late-training per-update drift is
5.5e-3 +/- 0.9e-3 nats (Full) and 6.7e-3 +/- 1.0e-3 (No-Aux), roughly 2.2
to 2.7 times higher than the Overcooked reference band (2.42 to 2.58e-3),
not lower. (One caveat: MPE's per-update batch is 1600 steps versus 25600
on Overcooked, so some of the additional movement may be batch-noise
driven; either way, Sigma_pi there is not small.) The intro sentence is
therefore withdrawn: Sigma_pi does not carry the MPE boundary. What this
measurement does not contradict is the proposition itself, which bounds the
product rather than Sigma_pi alone; the MPE null is consistent with the
theory if the pathway sensitivity J_pi or the curvature lambda_min(H)
compensates there. [UPGRADE SLOT: J_pi sentence from R17 when available.]
The camera-ready will (1) rescope the introduction and prediction (a) so
the boundary claim is stated on the product, keeping the
supervised-stationary clause that CIFAR-100 tests and removing the
"symmetric slow drift" clause, (2) report the MPE Sigma_pi measurement, and
(3) report the corresponding J_pi measurement on MPE (the target-
perturbation estimator from your Q3), or, if that does not resolve the
boundary, report MPE as an open boundary case. Your comment converted an
unexamined attribution into a measurement, and the boundary section will be
more accurate for it.

---

## Notes for Vishnu

- If R17 lands before posting: replace the UPGRADE SLOT with the measured
  comparison (AA references: J_pi response 2.47 at eps=0.1; aux/policy
  norm ratio 0.145 +/- 0.018) and, if favorable, tighten item (3) to a
  report of the number. If unfavorable, delete the slot and keep item (3)'s
  open-case fallback; do NOT claim the product is small without the number.
- Numbers: R16 (VERIFIED_NUMBERS section 10), R15 (section 10), R3
  soft-target self-cosines (section 5), R8 J_pi AA reference (section 3g).
- Do not quote R16 aux self-cos (hard-target ~0, section 8 caveat).
