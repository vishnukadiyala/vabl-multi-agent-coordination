---
tags: [experiments, neurips2026, yGKw, representation-drift, mpe, sigma-pi, complete]
status: complete
related: [neurips2026_final_comment_yGKw, mpe_ablation, gradient_diagnostics]
---

# R15 + R16: yGKw Final-Comment Measurements (2026-08-03)

Two experiments answering the reviewer's 2026-08-02 post-rebuttal comment.
Data: `results/R15_mpe_kl/` (local), `results/R16_repdrift/` (Celestia).
Analysis: `scripts/analyze_R15_R16.py`. Late window = last 20% of logged
entries. All arms n = 5 seeds.

## R15: Sigma_pi on MPE simple_spread — FALSIFIES the paper's attribution

Consecutive-policy KL (nats/update), same measurement semantics as R1 on AA.

| Arm | Final (last 3200 ep) | Late policy KL | vs AA band 2.42-2.58e-3 |
|---|---|---|---|
| full | -18.63 +/- 0.39 | 5.46e-3 +/- 0.93e-3 | 2.18x HIGHER |
| no_aux | -19.67 +/- 0.98 | 6.73e-3 +/- 1.04e-3 | 2.69x HIGHER |

- Reruns reproduce canonical April finals (full -18.82, no_aux -19.65), so
  the measurement rides on faithful replications.
- MPE per-update drift RISES over training (3.2e-3 early to 5.4e-3 late).
- **The paper's "slow symmetric drift => small Sigma_pi => no pathology"
  attribution is measured-wrong: MPE drifts 2.2-2.7x faster per update than
  AA, yet shows no pathology.** The MPE null is real but its mechanism is
  unresolved; Sigma_eps = J_pi Sigma_pi J_pi^T leaves small J_pi or large
  lambda_min(H) as unmeasured candidates.
- Caveat: MPE's per-update batch is 1600 steps vs 25600 on AA, so part of
  the higher KL may be batch-noise-driven movement rather than
  co-adaptation. This does not rescue the "slow drift" claim: Sigma_pi
  enters the theory per update, whatever its source.

## R16: Representation-drift probe on AA — SUPPORTS the common-mode reply

Encoder representation (new_belief, the aux head's input) drift on a fixed
1024-sample probe batch, fixed vae key, logged every 5 iterations. 10M
steps, exact R1 configuration plus `--log-feature-drift`.

| Arm | Final | Late feat self-cos | Late feat rel-L2 | Late policy KL |
|---|---|---|---|---|
| full | 466.84 +/- 5.95 | 0.9908 +/- 0.0021 | 0.131 +/- 0.015 | 2.37e-3 |
| frozen | 470.56 +/- 1.06 | 0.9851 +/- 0.0012 | 0.169 +/- 0.007 | 2.70e-3 |
| no_aux | 470.54 +/- 1.84 | 0.9915 +/- 0.0009 | 0.127 +/- 0.006 | 2.56e-3 |

- **Representation drift is real, ongoing, and common-mode**: all three arms
  sit at 0.985-0.991 self-cosine (0.13-0.17 relative L2 per 5-iteration
  window). The reviewer is right that it exists under fixed labels.
- **The frozen arm drifts slightly MORE in representation space than Full**
  (0.9851 vs 0.9908) yet shows no deficit and no low-outcome tail. So
  representation drift does not produce the pathology; label drift is the
  arm-separating factor.
- Deficit pattern reproduces: full 466.8 +/- 6.0 vs frozen/no_aux ~470.5
  with tight spreads (directional, n = 5).
- Policy-KL cross-check matches the R1 band in all arms.
- NOTE: R16's late aux-gradient self-cosine is ~0 in BOTH full and frozen
  arms (hard targets). This is the known limitation recorded in
  VERIFIED_NUMBERS section 8: the per-task self-cosine separates arms only
  under soft targets (R3: 0.974 frozen vs 0.79-0.86 drifting). Quote the R3
  numbers for the gradient-stability contrast, never R16's.

## What goes in the reply

1. Objection 1 (representation drift): concede the term exists, then give
   the R16 measurement: common-mode across arms, slightly larger in frozen,
   no deficit there; combined with R3 soft-target self-cosines, the
   decomposition is measured.
2. Objection 2 (MPE boundary): concede no derivation AND report that the
   requested measurement falsified our own small-Sigma_pi attribution;
   withdraw the "slow symmetric drift" language; MPE becomes an open
   boundary case in the camera-ready.
