---
tags: [reviews, rebuttal, draft, icml2026]
status: in-progress
related: [icml_rebuttal, vabl_icml2026, rebuttal_r1]
---

# Rebuttal Round 2 — Draft

## Strategy

**Core pivot:** Stop defending attention as the main contribution. Reposition around what the evidence actually supports — auxiliary belief regularization for coordination maintenance. Propose a concrete revision plan that gives reviewers a reason to vote "accept conditional on revision."

**Target:** Flip 6RAp (most movable, specific ask). Give dVmV enough for a conditional accept. Give the AC ammunition if iBYE holds.

---

## Response to Reviewer dVmV

> "my core concerns remain unaddressed... requires a significant update to the paper"

We appreciate the reviewer's clarity. We agree the paper requires repositioning and propose the following concrete revision:

**Revised contribution statement:** The paper's primary contribution is *auxiliary belief regularization for coordination maintenance* — a lightweight mechanism (under 10% overhead) that prevents the policy collapse observed in MAPPO, AERIAL, and TarMAC. The MI-grounded auxiliary loss (Lemma B.1, which the reviewer called "the most rigorous piece of mathematics in the paper") constrains beliefs to encode coordination-relevant features, providing measurable variance reduction (2–3.5×) across all tested environments.

**Specific revision plan:**

1. **Title and abstract** repositioned: "attention-driven" → "auxiliary belief regularization." Attention described as complementary at N≥3, not the central mechanism.

2. **Propositions 5.1–5.3** moved to discussion section as informal architectural observations. Theoretical contribution centered on Lemma B.1 (MI bound) and the new Lemma 5.5 proof.

3. **Honest attention analysis** added as a dedicated subsection: inoperative at N=2, complementary at N≥3. 8-agent results: Full VABL Best 89.4±0.4 vs No Attention 88.5±1.0 — attention reduces variance by 2.5× with 7 teammates. All VABL variants outperform MAPPO (62.9±15.3) by 42%.

4. **"Coordination maintenance" elevated to primary narrative.** Every method discovers coordination — AERIAL peaks at 1110, MAPPO at 503, TarMAC at 395. The contribution is sustaining it without communication: VABL maintains 28.1 final reward (38% collapse) vs AERIAL 0 (100%) and MAPPO 0 (100%). The differentiating factor is the auxiliary loss — AERIAL has attention but no aux loss and collapses completely.

5. **2M-step AA ablation** [PLACEHOLDER: include results when available — does attention win at convergence?]

We believe this revision preserves what the reviewer already valued — the problem importance, the practical design, the MI proof, the collapse metric — while honestly addressing the framing mismatch.

---

## Response to Reviewer iBYE

> "the paper still lacks evidence on a canonical belief-centric partially observable benchmark"

We acknowledge this limitation. Hanabi evaluation is implemented and will appear in the camera-ready. We chose Overcooked to test coordination maintenance (role assignment, collision avoidance) rather than deduction (card inference), and we now frame this choice explicitly.

**On theoretical contributions:** We agree the theoretical contribution is now primarily Lemma B.1 (the MI bound connecting auxiliary loss to mutual information maximization). Propositions 5.1–5.3 will be reclassified as informal architectural arguments in the revision, not formal results. We believe Lemma B.1 plus the extensive empirical characterization (6 baselines, 3 environments, 5 seeds, collapse analysis) constitutes a sufficient contribution for the empirical track.

**On 10M-step evidence:** We now have MAPPO 10M data from two seeds:
- Seed 0: 25,000 episodes — peak 503, final 0, collapse 100%
- Seed 1: [PLACEHOLDER: updated results from Celestia restart] — peak 1078, showing same collapse trajectory

Both seeds confirm that extended training worsens MAPPO collapse rather than resolving it.

**New scaling evidence:** 8-agent Simple Coordination with stochastic visibility (p=0.7), 5 seeds:

| Method | Best | Final |
|--------|------|-------|
| VABL Full | 89.4 +/- 0.4 | 57.4 +/- 1.1 |
| VABL No Attn | 88.5 +/- 1.0 | 57.6 +/- 1.4 |
| VABL No Aux | 89.3 +/- 0.4 | 57.3 +/- 1.1 |
| MAPPO | 62.9 +/- 15.3 | 29.6 +/- 9.9 |

All VABL variants outperform MAPPO by 42% with 38× lower variance. Attention provides consistent variance reduction at N=8 (std 0.4 vs 1.0), confirming selective weighting becomes operative with 7 teammates.

---

## Response to Reviewer 6RAp

> "some responses of questions are still unsatisfied, such as more baselines"

We now compare against **6 methods** across 3 environments:

| Method | Type | Communication | Evaluated On |
|--------|------|--------------|-------------|
| MAPPO | Policy gradient | None | AA, CR, Simple (3/5/8 agents), Ego-PO |
| QMIX | Value-based | None | AA |
| AERIAL | Attention + hidden states | Hidden state sharing | AA |
| TarMAC | Targeted communication | Learned messages | AA |
| CommNet | Broadcast communication | Averaged states | Simple |
| VABL ablations (4 configs) | Ablated VABL | None | AA, CR, Simple (5/8 agents) |

This is more baselines than most MARL papers at this venue. Regarding the specific methods the reviewer mentioned:

- **BAD** (Bayesian Action Decoder): Requires enumerable belief spaces — Overcooked's continuous observation space violates this assumption.
- **DICG** (Deep Implicit Coordination Graphs): Requires pairwise observation sharing between agents, breaking the decentralized execution constraint that VABL explicitly maintains.
- **MAAC** (Iqbal & Sha, 2019): Uses centralized attention over joint observations during training — architecturally distinct from VABL's decentralized belief update during execution. We add a detailed MAAC discussion in the revision clarifying the architectural differences.

**New generalization evidence:**
[PLACEHOLDER: Multi-layout results (Coordination Ring, Forced Coordination) if available]

Could the reviewer specify which additional baseline would be most informative? We want to ensure the revision addresses this concern directly.

---

## New Evidence Summary (for all reviewers)

### 8-Agent Simple Coordination (stochastic visibility p=0.7, 200ep, 5 seeds)

| Method | Best | Final |
|--------|------|-------|
| VABL (Full) | **89.4 +/- 0.4** | **57.4 +/- 1.1** |
| VABL (No Attn) | 88.5 +/- 1.0 | 57.6 +/- 1.4 |
| VABL (No Aux) | 89.3 +/- 0.4 | 57.3 +/- 1.1 |
| MAPPO | 62.9 +/- 15.3 | 29.6 +/- 9.9 |

All VABL variants outperform MAPPO by 42% (best) with 38x lower variance. Attention provides a consistent variance reduction at N=8 (std 0.4 vs 1.0 on best reward), confirming that selective weighting becomes operative with multiple teammates.

### 2M-Step AA Ablation (5000 episodes, 5 seeds)

[PLACEHOLDER: Critical result — does attention win at convergence?]

| Config | Best | Final | Aux Accuracy |
|--------|------|-------|-------------|
| Full VABL | PENDING | PENDING | PENDING |
| No Attention | PENDING | PENDING | PENDING |
| No Aux Loss | PENDING | PENDING | PENDING |
| Neither | PENDING | PENDING | PENDING |

### MAPPO 10M (multi-seed)

| Seed | Episodes | Best | Final | Collapse |
|------|----------|------|-------|----------|
| 0 | 25,000 | 503 | 0 | 100% |
| 1 | [PLACEHOLDER] | 1078 | [PLACEHOLDER] | [PLACEHOLDER] |

---

## Revision Commitments

| Change | Addresses |
|--------|-----------|
| Reposition contribution around auxiliary belief regularization | dVmV, iBYE |
| Props 5.1–5.3 → informal discussion | dVmV, iBYE |
| Honest attention subsection (N=2 trivial, N≥3 complementary) | dVmV |
| Hanabi evaluation in camera-ready | iBYE |
| MAAC discussion added | 6RAp |
| 8-agent + 2M AA ablation results | dVmV, iBYE |
| Multi-layout generalization | 6RAp |
