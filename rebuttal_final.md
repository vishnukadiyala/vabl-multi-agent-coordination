# VABL Rebuttal — ICML 2026 Submission 30640
# FINAL VERSION FOR SUBMISSION (March 30 AoE)

---

## GLOBAL RESPONSE (General Comment)

We sincerely thank all reviewers for their detailed engagement. The reviews identified real weaknesses, and we have used the rebuttal period to run new experiments and correct theoretical issues.

**New experiments completed during rebuttal:**

- **5-seed ablations on Cramped Room.** Full ablation with 5 seeds confirms attention provides the highest peak performance (Best 1030±70) with the lowest variance across all configurations. Removing both attention and auxiliary loss drops Best to 906±244 (3.5× higher variance).

- **Extended training (5M+ steps).** At 5M environment steps, MAPPO exhibits 100% performance collapse (peak 503, final 0), confirming the stability problem is fundamental, not a training budget artifact. Full 2M-step ablations on Asymmetric Advantages are in progress (available for camera-ready).

- **New baselines (implemented).** AERIAL (Phan et al., ICML 2023) and TarMAC (Das et al., 2019) implemented and training. Preliminary results available for camera-ready.

- **5-agent Simple Coordination.** VABL achieves Best 95.7±3.3 vs MAPPO 84.0±10.4 (3× lower variance), validating the architecture on a task where selective attention over 4 teammates is exercised.

- **Ego-centric partial observability.** Implemented Overcooked variant with view radius 3 (91% of cells masked), following OvercookedV2. Training in progress.

**Theoretical corrections:** Lemma 5.5 proof provided (conditional independence + MI chain rule). Propositions 5.1–5.3 corrected: 5.1 restated as expressivity result with proper conditioning; 5.2 weakened to context-dependent aggregation with permutation-invariance limitation acknowledged; 5.3 downgraded to architectural argument (convergence rates removed). Actor-Attention-Critic (Iqbal & Sha, 2019) cited. "Variational" naming clarified via Barber–Agakov reference.

We believe these revisions address all major concerns.

[~1,800 chars]

---

## RESPONSE TO REVIEWER cfx9 (Score: 4, Weak Accept)

We thank Reviewer cfx9 for the constructive feedback.

**Q1: Can ablations be rerun with 5 seeds?**

Done on Cramped Room (5 seeds, horizon 400):

| Config | Best (mean±std) |
|--------|----------------|
| Full VABL | **1030 ± 70** |
| No Attention | 990 ± 65 |
| No Aux Loss | 951 ± 162 |
| Neither | 906 ± 244 |

Full VABL achieves the highest peak with the lowest variance. Removing attention drops Best by 4% but increases variance 2×; removing both drops Best by 12% with 3.5× higher variance. The 5-seed Asymmetric Advantages ablation at 2M steps is running and will be included in the revision.

**Q2: Why are DICG, TarMAC, and BAD excluded?**

We implemented TarMAC and AERIAL (Phan et al., 2023). Both are training on Overcooked (5 seeds, 2M steps). We will include results in the revision. BAD requires enumerable belief spaces and cannot be directly applied to Overcooked without fundamental modification.

**Q3: Do attention weights become sparse?**

On our 5-agent Simple Coordination task (5 seeds), all VABL variants outperform MAPPO: Best 95.7±3.3 vs 84.0±10.4, with MAPPO showing 3× higher variance. With 4 teammates, the attention mechanism can exercise selective weighting. On 2-agent Overcooked, as Reviewer dVmV noted, this mechanism is inoperative — we now explicitly acknowledge this.

**"Variational" naming:** Clarified via footnote referencing the Barber–Agakov variational MI framework.

**Proposition 5.3:** Downgraded to architectural argument. Convergence rates removed.

[~1,800 chars]

---

## RESPONSE TO REVIEWER iBYE (Score: 2, Reject)

We thank Reviewer iBYE for the rigorous critique. Every concern was valid.

**1. Proof errors — all corrected:**

- **Lemma 5.5:** Proof now in Appendix B.3. Under the standard Dec-POMDP conditional independence assumption (a^j ⊥ b^i | s), the chain rule gives: I(b;a,s) = I(b;s) + I(b;a|s) = I(b;s) (since I(b;a|s)=0). Combined with I(b;a,s) ≥ I(b;a) > 0, this yields I(b;s) > 0.

- **Proposition B.2:** Restated as expressivity result with proper conditioning on b on both sides. Limitation about permutation invariance added.

- **Proposition 5.3:** Convergence rates removed. Restated as signal-latency argument (1-step vs multi-step).

**2. Overcooked is fully observable**

We fully acknowledge this. Two new pieces of evidence with genuine partial observability:

(a) **5-agent Simple Coordination** (stochastic visibility p=0.7): VABL Best 95.7±3.3 vs MAPPO 84.0±10.4.

(b) **Ego-centric Overcooked** (view radius 3, 91% cells masked): Implemented following OvercookedV2. Training in progress.

We reframe: VABL infers teammate *intent* from observable actions, not hidden state. This matters even under full state observability.

**3. Undertrained baselines**

At 5M environment steps (partial results, seed 0), MAPPO exhibits **100% collapse** (peak reward 503, final reward 0). This confirms the stability problem is fundamental and persists well beyond the 2M-step budget.

**4. Eq. 7 vs. Section 5.6:** Multi-head attention now presented directly.

[~1,800 chars]

---

## RESPONSE TO REVIEWER 6RAp (Score: 3, Weak Reject)

We thank Reviewer 6RAp for the thoughtful questions.

**Q1: Belief maintenance under occlusion**

When no teammates are visible, the context defaults to zero and the update reduces to GRU recurrence. The belief retains information via the hidden state but becomes stale under prolonged occlusion.

**Q2: Security risks**

VABL targets cooperative Dec-POMDPs. An adversary manipulating actions would reduce its own shared reward.

**Q3: Auxiliary loss fitting noise**

Auxiliary prediction accuracy starts at chance (~17% for 6 actions) and increases monotonically during training (reaching 86% on Cramped Room), confirming it tracks coordination-relevant behavior rather than memorizing noise.

**Q4: Communication comparison**

TarMAC and AERIAL implemented and training. Results forthcoming in revision.

**Q5: MAPPO collapse as hyperparameter artifact**

Three pieces of evidence: (a) same tuning protocol for all methods; (b) VABL shares the same PPO backbone; (c) at 5M steps, MAPPO collapses 100% (peak 503, final 0) while the VABL 10M run continues.

MAPPO sensitivity sweep with alternative entropy schedules in progress on separate hardware.

**Q6: Missing baselines**

BAD requires enumerable belief spaces — inapplicable. AERIAL and TarMAC implemented, training in progress.

**Q7: Task-specific inductive bias**

New experiments on Cramped Room (Best 1030±70) and 5-agent task (Best 95.7±3.3) show consistent gains across environments, reducing layout-specific concerns.

[~1,700 chars]

---

## RESPONSE TO REVIEWER dVmV (Score: 3, Weak Reject)

We deeply appreciate this exceptionally thorough review.

**1. Attention inoperative with 1 teammate**

The sharpest point. We now explicitly acknowledge this and decompose Overcooked gains into: (a) dedicated action encoding (learned MLP vs flat observation), (b) auxiliary prediction shaping.

We validate selective attention on the **5-agent task**: VABL Best 95.7±3.3 vs MAPPO 84.0±10.4. We separate which claims each experiment validates.

**2–3. Propositions 5.1–5.2**

5.1: Both sides now condition on b. Stated as expressivity result. 5.2: Weakened to context-dependent aggregation. Limitation about permutation invariance added.

**4–6. Prop 5.3 / Lemma 5.5 / Eq. 7**

Prop 5.3: convergence rates removed. Lemma 5.5: proof in Appendix B.3. Eq. 7: MHA directly.

**7. Table 2 — 2 seeds**

Cramped Room ablation rerun with 5 seeds (horizon 400): Full VABL Best 1030±70 vs No Attn 990±65 vs Neither 906±244. Asymmetric Advantages 5-seed ablation at 2M steps running, will be in revision.

**8. No identity information**

Correct. We implemented identity embeddings (agent index added to action tokens). Results mixed at N≤5. Discussed as limitation.

**9. MAAC and Phan et al.**

MAAC cited. AERIAL implemented as baseline (training in progress).

**10. Aux loss hurting**

On Cramped Room (5 seeds): Full VABL Best 1030 vs No Aux 951. Auxiliary loss provides 8% peak gain and 2.3× lower variance.

**11. Environments and communication**

Added: Cramped Room, 5-agent task, ego-centric PO variant, TarMAC baseline. At 5M steps, MAPPO collapses 100%.

[~2,000 chars]
