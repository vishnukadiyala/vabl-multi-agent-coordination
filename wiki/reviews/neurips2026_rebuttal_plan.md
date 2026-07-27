---
tags: [neurips2026, rebuttal, reviews, strategy]
status: active
related: [neurips2026_reviews, vabl_neurips_pivot, gradient_diagnostics, mechanism_identification, theory_parallels]
---

# NeurIPS 2026 Rebuttal Plan — Submission29511

Reviews received 2026-07-24. Raw text: `raw/reviews/neurips2026_reviews.md`.

## Score situation

| Reviewer | Rating | Confidence | Stance |
|----------|--------|------------|--------|
| fXvf | 4 (borderline accept) | 3 | Positive, long concrete ask list |
| PYCT | 4 (borderline accept) | 3 | Most positive; 2 sharp questions |
| yGKw | 3 (borderline reject) | 4 | Engaged skeptic; Sig/Orig = 2; wrote an 8-item to-do list |
| AC y2yo | meta leaning reject | — | Category mismatch, abstract, "overclaims general DL" |

Path to accept: (1) flip yGKw 3 -> 4 by delivering their measurable asks
(they are highest-confidence, so their score anchors the AC); (2) correct the
AC's two factual misreadings; (3) keep fXvf/PYCT at 4+ with direct answers.
All three reviewers rate Quality/Clarity 3 — nobody disputes the mechanism;
the fight is over identification rigor, measurement directness, and framing.

## The three cross-cutting asks (multiple reviewers)

1. **Measure the theory quantities directly** (fXvf Q4/Q6, yGKw Q1/Q3, PYCT
   linearization weakness). Estimate Sigma_pi via consecutive-policy KL,
   J_pi via logit perturbation on logged trajectories; correlate with
   Final50 degradation and cosine-std. THE highest-leverage item.
2. **PCGrad/GradNorm baselines** (fXvf Q5, yGKw W3/Q5). Paper predicts they
   do NOT fix directional interference (I_mag ~= 0) — running them is a
   falsifiable prediction test that strengthens the core distinction either way.
3. **Isolate temporal drift in the target-source test** (fXvf W1/Q5, yGKw
   W4/Q2). Soft-snapshot continuum: targets = policy snapshot at lag k
   (k = 0 (co-learning), small, large, infinite (frozen)), matched label
   distribution via soft probabilities. One experiment answers both reviewers.

## Per-reviewer strategy

### AC y2yo (top-level author comment — write FIRST, this is the real audience)
- **Category**: reviewers all marked the paper "General", not "Theory". If the
  submission form also says General (VERIFY on OpenReview), politely correct
  the factual premise of the meta-review. If it says Theory, own it and note
  the contribution-type field will be corrected; the paper self-describes as a
  mechanistic/empirical characterization.
- **"Overclaims broad applicability while main effects fail to transfer
  outside cooperative MARL"**: this inverts the paper's logic. The CIFAR-100
  null is a PREDICTED boundary (stationary targets -> no pathology), i.e.
  confirmatory evidence for the mechanism, not a failed generalization.
  State this in two sentences, cite the pre-registered prediction in §Theory.
- **Abstract**: concede density; post the revised 4-sentence abstract verbatim
  in the rebuttal (problem / gap / mechanism / evidence spine per writing
  guide). Cheap, visible concession.

### yGKw (rating 3, conf 4 — the kingmaker)
Deliver, in priority order:
- Q1: Sigma_pi via consecutive-policy KL from saved checkpoints; scatter vs
  cosine-std and Final50 drop across seeds/envs. (Check checkpoint cadence
  in results/ — if per-eval checkpoints exist this is reanalysis, not new runs.)
- Q3: J_pi logit-perturbation estimate on logged trajectories (reanalysis).
- Q5: PCGrad + GradNorm in Full setting on AA (new runs, 2 configs x 5 seeds).
  Frame as prediction: variance should NOT reduce because I_mag ~= 0.
- Q2: soft-snapshot continuum (new runs, ~4 conditions x 5 seeds on AA).
- Q6: drift-gated controller prototype (stop-gradient when cosine-std > tau).
  Already have the DG-PG Thm 4.2 framing; even a 1-env demo converts "future
  work" into evidence. `stop_gradient_belief` knob + diagnostic logging exist.
- Q7: SMAX E[cos] late-training bias — reanalysis of existing gradient logs;
  directly addresses the mean-gap question and the zero-mean assumption.
- Q4/Q8: answer honestly as scope/future work; do not promise runs.
- Sig/Orig = 2 counter: the identification methodology (target-source test +
  directional-vs-magnitude signature) is the contribution; PYCT independently
  calls it "an original identification method" — quote nothing, but make the
  same argument.

### fXvf (rating 4, conf 3)
- Q1/Q7: commit to exact claim-narrowing edits; give before/after wording for
  the worst two sentences ("harmless" -> "not resolved at n=5" is already the
  paper's softened style; extend to abstract/intro).
- Prompt-injection allegation: state plainly that the authors' source and
  submitted PDF contain no such text; the hidden text was inserted by the
  NeurIPS organizers' post-submission LLM-review-detection experiment, not by
  the authors. Factual, no accusation of the reviewer. Flag to AC in the
  top-level comment since it was cited as an integrity concern.
- Q3: parameter-sharing / multi-teammate / identity-embedding details — write
  the §3 clarification paragraph in the rebuttal (zero compute).
- Q5/Q6: covered by cross-cutting items 2/3 + seeds below.
- Seeds: extend the headline AA Phase-2 comparison and target-source test
  5 -> 10 seeds if the window allows; otherwise commit for camera-ready and
  report widened bootstrap CIs honestly.

### PYCT (rating 4, conf 3 — keep warm, aim for 5)
- Q1 (SMAX secondary pathology): hypothesis + diagnostic: check feature-rank /
  representation-collapse metrics on SMAX runs (No Representation No Trust
  cite is already in theory/); tie to E[cos] bias reanalysis (yGKw Q7 overlap).
- Q2 (Hanabi absolute scores): own it: VABL-on-Hanabi is a minimal testbed,
  not SOTA; the claim is the ordering, and d=+2.81 severity; note stronger
  baselines with aux heads are exactly the population the mechanism targets;
  offer R2D2-style baseline as camera-ready if feasible.
- BEPAL/Dynamic Belief external validity (their W1): concede directly; the
  honest answer is VABL is a minimal reconstruction by design (isolation);
  cite the Zhai Static-Belief point as corroboration not proof.

## Feasibility / compute triage

- **Zero-compute (write in rebuttal now)**: AC comment, abstract rewrite,
  claim-narrowing edits, §3 clarification, injection response, Hanabi answer,
  Appendix L deletion promise (fXvf may have misread it; delete it
  regardless, it is drafting meta-commentary that should not be in the paper).
- **Reanalysis of existing artifacts (days)**: Sigma_pi KL (needs checkpoint
  cadence check), J_pi perturbation, SMAX E[cos] bias, per-seed curves +
  peak-to-final drops (fXvf Q6), Final50 window justification.
- **New runs (needs GPU scoping vs rebuttal deadline)**: PCGrad/GradNorm
  (10 runs), soft-snapshot continuum (~20 runs), +5 seeds on headline configs
  (~10-20 runs), drift-gated controller (5 runs). JAX pipeline ~101 ep/s;
  scope against Celestia availability. Priority order as listed.

## Open items

- [x] Rebuttal deadline: **2026-07-27** (confirmed by Vishnu 2026-07-24)
- [x] Contribution type: submitted under **Theory** (confirmed; own it in the
      AC response, do not dispute the premise). Reviewers still marked General.
- [ ] Draft written: `neurips2026_rebuttal_draft.md` (placeholders pending runs
      R1-R7; schedule in draft)
- [ ] Check checkpoint cadence in results/ for Sigma_pi KL reanalysis
- [ ] Decide which PDF was submitted (`neurips_submission.pdf` vs `...1.pdf`)
      and record it; both are locally clean of injected text (verified
      2026-07-24 via pypdf text-layer scan)
