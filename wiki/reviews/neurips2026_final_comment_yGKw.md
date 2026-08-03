---
tags: [neurips2026, rebuttal, reviews, yGKw, representation-drift, mpe-boundary]
status: active
related: [neurips2026_rebuttal_plan, VERIFIED_NUMBERS, mpe_ablation, cifar_5seed]
---

# yGKw Final Comment (2026-08-02) — Two Theory-Scope Objections

Reviewer yGKw (rating 3, conf 4) posted a follow-up comment on 2026-08-02
after our 2026-07-27 rebuttal. Two objections, both aimed at the theory's
framing rather than the experiments. Neither asks for new in-window runs.

## Objection 1: Representation drift under fixed labels

**Claim:** Even with frozen labels, the shared encoder keeps updating, so the
features feeding the aux head drift and aux gradients couple with the evolving
feature space. The paper equates fixed labels with Sigma_pi = 0 and never puts
representation drift in the causal chain. "Theoretically inconsistent."

**Where it lands in the paper:** `neurips_submission.tex:1252-1256` (CIFAR
appendix): "with stationary targets ... Sigma_pi = 0, so
Sigma_eps = J_pi Sigma_pi J_pi^T = 0 and the steady-state variance ...
vanishes." Also the prediction (a) statement at line 462.

**Assessment: the reviewer is right about the text, and we already hold the
counter-evidence.** Sigma_eps as written is the *target-drift component* of
aux-gradient variability, not the total; setting Sigma_pi = 0 zeroes that
component, not the whole. The "= 0" sentence is an idealization. Two facts
bound the omitted representation-drift term:

1. **The frozen condition retains full representation drift** (encoder trains
   on the RL loss throughout) yet shows no deficit and near-stationary aux
   gradients. So representation drift alone is insufficient for the pathology.
2. **Measured:** under matched soft labels, aux-gradient self-cosine is 0.974
   (frozen) vs 0.79-0.86 (drifting), n=5 per arm (VERIFIED_NUMBERS section 5).
   If representation drift were first-order in aux-gradient direction
   variability, the frozen arm could not sit at 0.974.
3. The nonzero CIFAR residual the paper already reports as "weak boundary
   residual" (`tex:1240-1250`) is plausibly exactly this term.

**Camera-ready commitment:** state the two-term decomposition explicitly
(target-drift term, modeled; representation-drift residual, unmodeled,
measured small); correct "Sigma_eps = 0" to "the target-drift component of
Sigma_eps vanishes."

## Objection 2: MPE simple_spread boundary is not derived

**Claim:** "Symmetric slow-drift effect drops to zero" on MPE has no formal
derivation; it is empirically attributed to small Sigma_pi and cannot be
generalized to other symmetric cooperative environments.

**Where it lands:** `tex:155-156` ("as the principle predicts"), `tex:462-463`
(prediction (a) includes "symmetric slow drift" under Sigma_pi ~ 0),
`tex:729` (Table row, ~0 effect). Fig 8e "MPE analog" is a synthetic
illustration, not a derivation.

**Assessment: fully correct; concede.** Sigma_pi (consecutive-policy KL) was
never measured on MPE (mpe_ablation ran on the April stack, before the July
KL instrumentation). The small-Sigma_pi attribution is a hypothesis.

**Camera-ready commitment:** (1) reword "as the principle predicts" to
"consistent with"; scope the boundary to the tested environment, no
universality claim over symmetric cooperative games. (2) Run the
consecutive-policy KL instrumentation on MPE simple_spread (cheap: horizon
25, 100K episodes) so the attribution is measured. If drift there is not
small, report the MPE null as an open boundary case instead.

## Strategy

Both points are concession-plus-measurement replies. The reviewer responded
well to the rebuttal's direct-concession style; repeat it, keep it short
(two paragraphs), lead each with "you are right." Do not dispute the word
"inconsistent"; concede the substance as an unmodeled term. Respond fast;
discussion window is closing (comment dated Aug 2, today Aug 3).

Draft reply lives in this page's companion:
see `neurips2026_final_comment_reply_draft.md`.
