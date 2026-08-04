---
tags: [neurips2026, rebuttal, yGKw, draft]
status: active
related: [neurips2026_final_comment_yGKw, r15_r16_final_comment, VERIFIED_NUMBERS]
---

# Draft Reply to yGKw Final Comment (for OpenReview)

REVISED 2026-08-03 after R15 + R16 completed. All numbers verified against
`scripts/analyze_R15_R16.py` output on `results/R15_mpe_kl/` and
`results/R16_repdrift/`, and VERIFIED_NUMBERS.md section 5 (R3 soft-target
self-cosines). No em-dashes. Ready to paste after Vishnu's review.

---

Thank you for the close reading. Both points are correct. We ran two new
experiments against them; one measurement supports the framing you pushed us
toward, and the other falsified our own wording, which we report and will
correct.

**On representation drift under fixed labels.** You are right: fixing the
auxiliary labels does not make the auxiliary gradient stationary, because
the shared encoder keeps updating under the RL loss, and our causal chain
never named this term. The quantity Sigma_eps = J_pi Sigma_pi J_pi^T in our
model is the target-drift component of auxiliary-gradient variability, not
the total, and the appendix sentence stating that Sigma_pi = 0 implies
Sigma_eps = 0 overstates this. We have now measured the omitted term. We
reran Full, frozen-target, and No-Aux (5 seeds each, 10M steps, Overcooked
AA) while tracking the aux head's input representation on a fixed probe
batch of 1024 states. Late-training representation drift is present and
common-mode across all three arms: consecutive-measurement feature
self-cosine 0.9908 +/- 0.0021 (Full), 0.9851 +/- 0.0012 (frozen), 0.9915
+/- 0.0009 (No-Aux). The frozen arm drifts slightly more in representation
space than Full, yet reproduces neither the deficit nor the low-outcome
tail (frozen 470.6 +/- 1.1 vs Full 466.8 +/- 6.0 in these reruns). So
representation drift is real, but it is shared by conditions that do and do
not exhibit the pathology, while label drift separates them; under matched
soft labels the auxiliary gradient's iteration-to-iteration self-cosine is
0.974 with frozen targets against 0.79 to 0.86 with drifting targets. The
camera-ready will state the decomposition explicitly: a target-drift term
(modeled) plus a representation-drift term (unmodeled, now measured, common
to all conditions), and will correct the "Sigma_eps = 0" sentence to "the
target-drift component of Sigma_eps vanishes."

**On the MPE simple_spread boundary.** Correct, and the measurement you
asked for made this concession stronger than you may have expected. There
is no derivation behind "symmetric slow drift"; it was a hypothesis, and we
had never measured Sigma_pi on MPE. We have now instrumented MPE
simple_spread with the same consecutive-policy KL measurement used on
Overcooked and rerun the original ablation arms (5 seeds each; reruns
reproduce the original returns). The result falsifies our attribution:
late-training per-update drift on MPE is 5.5e-3 +/- 0.9e-3 nats (Full) and
6.7e-3 +/- 1.0e-3 (No-Aux), roughly 2.2 to 2.7 times HIGHER than the
Overcooked reference band (2.42 to 2.58e-3), and it rises over training
rather than decaying. (One comparability caveat: MPE's per-update batch is
1600 steps versus 25600 on Overcooked, so some of the additional movement
may be batch-noise-driven; either way, per-update drift there is not
small.) MPE's policy is not slowly drifting, yet the pathology is absent.
The camera-ready will therefore withdraw the "slow symmetric drift"
explanation entirely and report the MPE null as an open boundary case: the
model's noise term Sigma_eps = J_pi Sigma_pi J_pi^T leaves a small pathway
sensitivity J_pi or a well-conditioned landscape (large lambda_min(H)) as
candidate explanations, and we have not measured either on MPE. The
prediction (a) statement and the introduction's boundary sentence will be
rescoped accordingly, with no generality claim over symmetric cooperative
games. This correction came from running the measurement your review asked
for, and we think the paper is more accurate for it.

---

## Notes for Vishnu (not part of the reply)

- Numbers: R16 feature self-cos and finals from analyze_R15_R16.py on
  results/R16_repdrift; R15 KL from results/R15_mpe_kl; 0.974 / 0.79-0.86
  from R3 (VERIFIED_NUMBERS section 5, soft targets).
- Do NOT cite R16's aux self-cosine (~0 in both arms under hard targets;
  known soft-target-only diagnostic, VERIFIED_NUMBERS section 8).
- The MPE result changes the PAPER, not just the reply: intro line ~155
  ("as the principle predicts"), prediction (a) line ~462 ("symmetric slow
  drift"), Table row commentary ~735, and the Fig 8e "MPE analog" framing
  all need the camera-ready rescope.
- Full/frozen/no_aux finals in R16 reproduce the deficit pattern at n=5
  (directional). Frozen-drifts-more-than-Full is a nice touch: it forecloses
  "the frozen arm just moves less" as an alternative reading.
