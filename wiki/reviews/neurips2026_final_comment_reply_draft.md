---
tags: [neurips2026, rebuttal, yGKw, draft]
status: active
related: [neurips2026_final_comment_yGKw, VERIFIED_NUMBERS]
---

# Draft Reply to yGKw Final Comment (for OpenReview)

Numbers verified against VERIFIED_NUMBERS.md section 5 and
neurips_submission.tex lines 1240-1256. No em-dashes. Ready to paste after
Vishnu's review.

---

Thank you for the close reading. Both points are correct, and we accept them.

**On representation drift under fixed labels.** You are right that fixing the
auxiliary labels does not make the auxiliary gradient stationary: the shared
encoder continues to update under the RL loss, so the features feeding the
auxiliary head drift in every condition, and our causal chain does not name
this term. The quantity Sigma_eps = J_pi Sigma_pi J_pi^T in our model is the
target-drift component of auxiliary-gradient variability, not the total, and
the sentence in the appendix stating that Sigma_pi = 0 implies Sigma_eps = 0
is an idealization that overstates this. Two measurements bound the size of
the omitted term in our setting. First, the frozen-target condition retains
full representation drift by construction (its encoder trains on the RL loss
throughout), yet it shows neither the deficit nor the low-outcome tail, so
representation drift alone does not produce the pathology we study. Second,
under matched soft labels the auxiliary gradient's iteration-to-iteration
self-cosine is 0.974 with frozen targets against 0.79 to 0.86 with drifting
targets (n = 5 per arm); if representation drift were a first-order
contributor to auxiliary-gradient direction variability in this regime, the
frozen arm could not sit at 0.974. The small nonzero residual we already
report on CIFAR-100 (the "weak boundary residual" in the appendix) is
plausibly exactly the term you identify, and your comment gives it the right
name. The camera-ready will state the decomposition explicitly: a
target-drift term (modeled) plus a representation-drift residual (unmodeled,
measured small, and common to all conditions), and will correct the
"Sigma_eps = 0" sentence to "the target-drift component of Sigma_eps
vanishes."

**On the MPE simple_spread boundary.** Correct, and we will not defend the
current wording. There is no derivation that symmetric slow drift drives the
effect to zero; the MPE null is an empirical boundary observation, and the
small-Sigma_pi attribution is a hypothesis we did not measure in that
environment (the synthetic "MPE analog" in the appendix illustrates the
model's low-Sigma_eps regime and is not a derivation for the environment).
The camera-ready will make two changes. The claim will be rescoped from "as
the principle predicts" to "consistent with the principle," stated for the
tested environment only, with no generality claim over symmetric cooperative
games. And we will run the consecutive-policy KL instrumentation built for
this rebuttal on MPE simple_spread, so that the small-drift attribution is
measured rather than asserted; if the measured drift there is not small, we
will report the MPE null as an open boundary case instead.

---

## Notes for Vishnu (not part of the reply)

- The 0.974 vs 0.79-0.86 self-cosine is the load-bearing number; it is from
  the soft-snapshot continuum (matched soft labels), section 5 of
  VERIFIED_NUMBERS.md. Do not swap in the between-condition cosine-std
  (0.160/0.226), which does not separate the arms and was withdrawn.
- Deliberately short. The reviewer is doing a final consistency pass, not
  asking for new in-window experiments.
- The MPE KL run is a camera-ready commitment, honest either way it lands.
