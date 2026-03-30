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

- **Lemma 5.5 (missing proof):** We provide the full argument here. Assume the standard Dec-POMDP conditional independence: agent j's action is independent of agent i's belief given the state, i.e., a^j_{t+1} ⊥ b^i_t | s_{t+1}. This holds because without communication, agents' private information is conditionally independent given the state. By the MI chain rule applied two ways:

  I(b; a, s) = I(b; s) + I(b; a|s) = I(b; s) + 0 = I(b; s)
  I(b; a, s) = I(b; a) + I(b; s|a) ≥ I(b; a) > 0

  Combining: I(b; s) ≥ I(b; a) > 0. Therefore any belief predictive of teammate actions must encode state-relevant information. QED.

- **Proposition B.2 Step 1:** The reviewer correctly noted that supremum over the attention family does not imply anything about learned weights. We now state this explicitly: "This is a representational capacity result. It does not guarantee that a specific learned θ achieves the supremum." The revised proposition compares sup_{θ∈Θ_attn} I(A; c_attn | b) ≥ sup_{θ∈Θ_mean} I(A; c_mean), conditioning on fixed b on both sides so the comparison is between deterministic functions of A.

- **Proposition B.2 Step 2:** The reviewer identified an unjustified switch from joint action A to subset A_S and from conditional to unconditional MI. We now explicitly state the relevance structure assumption: I(a^k; s | A_S) = 0 for k ∉ S, under which the argument holds. We also add a limitation: since attention computes a weighted sum, it cannot distinguish permutations of actions among equally-weighted teammates.

- **Proposition 5.3:** O(γ^k) convergence rates removed entirely. The proposition is now labeled "Informal" and stated as an architectural observation: VABL receives teammate action changes at 1-step latency via the attention context, while RNN must infer changes through multi-step observation effects. We explicitly state: "We do not claim formal convergence rates, as deriving these for nonlinear GRU dynamics would require spectral analysis under specific regularity conditions."

**2. Overcooked is fully observable**

We fully acknowledge this. We reframe: VABL infers teammate *intent* from observable actions, not hidden state. New evidence with genuine partial observability:

(a) **5-agent Simple Coordination** (stochastic visibility p=0.7): VABL Best 95.7±3.3 vs MAPPO 84.0±10.4.

(b) **Ego-centric Overcooked** (view radius 3, 91% cells masked): Implemented following OvercookedV2. Training in progress.

**3. Undertrained baselines**

At 10M environment steps (seed 0, 25,000 episodes), MAPPO exhibits **100% collapse** (peak reward 503, final reward 0). This confirms the stability problem is fundamental, not a training budget artifact.

**4. Eq. 7 vs. Section 5.6:** Equation 7 now presents multi-head attention directly with h=4 heads and scaled dot-product, matching the implementation. The scalar compatibility function g_θ is replaced with the full MHA formulation.

[~3,200 chars]

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

The sharpest point in all reviews. With one teammate, α = 1.0 trivially — no selective weighting occurs. We now explicitly acknowledge this and decompose Overcooked gains into: (a) dedicated action encoding — VABL processes teammate actions through a learned MLP (ψ_θ) concatenated with observation embeddings before the GRU, providing a stronger inductive bias than MAPPO's flat observation input; (b) auxiliary prediction shaping beliefs toward coordination-relevant features.

We validate selective attention on the **5-agent task** (4 teammates): VABL Best 95.7±3.3 vs MAPPO 84.0±10.4, with MAPPO showing 3× higher variance. We now clearly separate: Overcooked validates action-encoding and auxiliary components; the 5-agent task validates selective attention.

**2. Proposition 5.1 — MI comparison**

The reviewer's counterexample (mean pooling over two independent binary actions preserving more joint info than concentrated attention) is valid for specific weight settings. We now state the corrected version: sup_{θ∈Θ_attn} I(A; c_attn | b_{t-1}^i = b) ≥ sup_{θ∈Θ_mean} I(A; c_mean), conditioning on fixed b on both sides. This is explicitly labeled a "representational capacity result" — the attention family contains mean pooling as a special case (setting g_θ = const), so the supremum over the larger set cannot decrease. We state: "It does not guarantee that a specific learned θ achieves the supremum."

**3. Proposition 5.2 — weighted sum loses identity**

Weakened from "sufficient statistic" to "context-dependent aggregation." We add: "As a weighted sum, attention cannot distinguish permutations of actions within the relevant set: if two teammates take the same action, their key-value tokens are identical. When individual teammate identities are decision-relevant, additional input features (e.g., position encodings) are needed." We implemented identity embeddings (agent index added to action tokens); results mixed at N≤5, suggesting identity tracking matters primarily at larger scales.

**4. Proposition 5.3** — O(γ^k) rates removed entirely. Now labeled "(Informal)" and stated as: "VABL receives teammate action changes at 1-step latency via the attention context, while RNN must infer changes through multi-step observation effects." We explicitly disclaim: "We do not claim formal convergence rates."

**5. Lemma 5.5** — Full proof provided (see iBYE response #1 for the complete argument using conditional independence and MI chain rule).

**6. Eq. 7 vs Section 5.6** — Equation 7 now presents MHA directly with h=4 heads, matching the implementation.

**7. Table 2 — 2 seeds, 50 episodes**

Cramped Room ablation rerun with 5 seeds (horizon 400):

| Config | Best (mean±std) |
|--------|----------------|
| Full VABL | **1030 ± 70** |
| No Attention | 990 ± 65 |
| No Aux Loss | 951 ± 162 |
| Neither | 906 ± 244 |

Full VABL achieves highest peak with lowest variance (3.5× lower than the baseline).

**10. Aux loss hurting in original Table 2**

With 5 seeds on Cramped Room: Full VABL Best 1030 vs No Aux 951. Auxiliary loss provides 8% peak gain and 2.3× lower variance (±70 vs ±162). The primary benefit is representation shaping and cross-seed reliability, not peak performance alone.

**11. Environments and baselines**

Added: Cramped Room (Best 1030±70), 5-agent task (Best 95.7±3.3), ego-centric PO variant (training in progress), TarMAC and AERIAL baselines (training in progress). At 10M steps, MAPPO collapses 100% (peak 503, final 0).

[~3,500 chars]
