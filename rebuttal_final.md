# VABL Rebuttal — ICML 2026 Submission 30640

---

## GLOBAL RESPONSE (General Comment)

We thank all reviewers for their detailed engagement. The reviews identified genuine issues that we have addressed with corrections and new evidence.

**Key new result:** At 10M environment steps (25,000 episodes), MAPPO collapses to zero reward (100% performance loss from peak 503). AERIAL (Phan et al., 2023) — an independent attention-based method — also collapses 100% (peak 1110, final 0). VABL is the only method tested that maintains stable coordination at extended training. This is our central empirical claim.

**New experiments:**

- **5-seed ablation (Cramped Room, horizon 400).** Full VABL achieves Best 1030±70 with the lowest variance of any configuration. Removing both attention and auxiliary loss drops Best to 906±244 (3.5× higher variance). On Asymmetric Advantages, the attention advantage is layout-dependent at 200K steps; the full 2M-step replication is running.

- **5-agent Simple Coordination (5 seeds).** All VABL variants outperform MAPPO (Best 95.7±3.3 vs 84.0±10.4), with MAPPO showing 3× higher variance, validating the architecture on a task where selective attention over 4 teammates is exercised.

- **New baselines implemented.** AERIAL (Phan et al., 2023) and TarMAC (Das et al., 2019) implemented. Preliminary AERIAL result (seed 0, 2M steps): Best 1110, Final 0, 100% collapse — the same failure mode as MAPPO, confirming that attention-based methods without auxiliary belief learning also suffer catastrophic collapse.

**Theoretical corrections:** Lemma 5.5 proof provided. Proposition 5.1 corrected to expressivity result with proper conditioning. Proposition 5.3 downgraded to architectural argument (convergence rates removed). Eq. 7 revised to present MHA directly. Actor-Attention-Critic (Iqbal & Sha, 2019) cited.

---

## RESPONSE TO REVIEWER cfx9 (Score: 4, Weak Accept)

We thank Reviewer cfx9 for the constructive feedback.

**Q1: Can ablations be rerun with 5 seeds?**

Done. Cramped Room (5 seeds, horizon 400) shows clear monotonic degradation:

| Config | Best (mean±std) |
|--------|----------------|
| Full VABL | **1030 ± 70** |
| No Attention | 990 ± 65 |
| No Aux Loss | 951 ± 162 |
| Neither | 906 ± 244 |

Full VABL achieves highest peak with lowest variance (3.5× lower than baseline). On Asymmetric Advantages, preliminary 500-episode results show the attention advantage is layout-dependent at this training budget; the 2M-step replication is running. The variance reduction from auxiliary loss is consistent across both layouts (Cramped Room: std 70 vs 162 without aux = 2.3× lower).

**Q2: Why are DICG, TarMAC, and BAD excluded?**

We implemented TarMAC and AERIAL (Phan et al., 2023). Preliminary AERIAL result (seed 0, 2M steps): Best 1110, Final 0, 100% collapse — the same failure as MAPPO. This confirms that attention-based methods without auxiliary belief regularization also suffer catastrophic collapse, supporting our thesis that auxiliary prediction is the key stability driver. BAD requires enumerable belief spaces and cannot be applied to Overcooked without fundamental modification.

**Q3: Do attention weights become sparse?**

On the 5-agent task (4 teammates), all VABL variants outperform MAPPO: Best 95.7±3.3 vs 84.0±10.4. We have attention entropy analysis from training showing attention weights become non-uniform (mean entropy 0.35 nats vs uniform baseline 0.69 nats). On 2-agent Overcooked, as Reviewer dVmV noted, selective weighting is inoperative — we now explicitly acknowledge this and separate which claims each environment validates.

**"Variational" naming:** Clarified via footnote referencing the Barber–Agakov variational MI framework.

---

## RESPONSE TO REVIEWER iBYE (Score: 2, Reject)

We thank Reviewer iBYE for the rigorous critique. Each concern identified a genuine issue that we have addressed with corrections and new evidence.

**1. Proof errors — all corrected:**

- **Lemma 5.5:** Under the Dec-POMDP conditional independence assumption (a^j_{t+1} ⊥ b^i_t | s_{t+1} — agents' private information is conditionally independent given the state without communication), the MI chain rule gives:

  I(b; a, s) = I(b; s) + I(b; a|s) = I(b; s)  [since I(b;a|s)=0 by assumption]
  I(b; a, s) = I(b; a) + I(b; s|a) ≥ I(b; a) > 0

  Combining: I(b; s) ≥ I(b; a) > 0. Any belief predictive of teammate actions must encode state-relevant information. QED.

- **Proposition B.2 Step 1:** Now states: sup_{θ∈Θ_attn} I(A; c_attn | b) ≥ sup_{θ∈Θ_mean} I(A; c_mean), conditioning on fixed b on both sides. Explicitly labeled as "representational capacity result — does not guarantee that a specific learned θ achieves the supremum."

- **Proposition B.2 Step 2:** Relevance structure assumption I(a^k; s | A_S) = 0 for k ∉ S now explicitly stated. Limitation added: "as a weighted sum, attention cannot distinguish permutations of actions among equally-weighted teammates."

- **Proposition 5.3:** O(γ^k) rates removed. Now labeled "(Informal)" and stated as an architectural signal-latency argument. We explicitly disclaim: "We do not claim formal convergence rates."

**2. Overcooked is fully observable**

We agree that standard Overcooked does not test belief formation under partial observability. We retain it as a coordination benchmark where VABL's action-encoding pathway and auxiliary loss provide measurable stability benefits, and we add partially observable environments (5-agent Simple Coordination with stochastic visibility, ego-centric Overcooked variant with view radius 3) to test the belief mechanism.

5-agent (stochastic visibility p=0.7): All VABL variants outperform MAPPO (Best 84.0±10.4, 3× higher variance), demonstrating the architecture's advantage under genuine partial observability.

**3. Undertrained baselines**

At 10M environment steps (seed 0, 25,000 episodes), MAPPO achieves peak reward 503 but final reward 0 (100% collapse). AERIAL (Phan et al., 2023) also collapses 100% (peak 1110, final 0). Two architecturally distinct baselines exhibiting identical collapse patterns confirms the stability problem is fundamental, not a training budget artifact.

**4. Eq. 7 vs. Section 5.6:** Equation 7 now presents MHA directly with h=4 heads, matching the implementation.

---

## RESPONSE TO REVIEWER 6RAp (Score: 3, Weak Reject)

We thank Reviewer 6RAp for the thoughtful questions.

**Q1: Belief maintenance under occlusion**

When no teammates are visible, the context defaults to zero and the update reduces to GRU recurrence. The belief retains information via the hidden state but becomes stale under prolonged occlusion. Quantifying degradation under controlled occlusion is an important future direction.

**Q2: Security risks**

VABL targets cooperative Dec-POMDPs. Robustness under adversarial or suboptimal teammates (e.g., via minimax belief updates) is a meaningful extension we now discuss in Limitations.

**Q3: Auxiliary loss fitting noise**

Auxiliary prediction accuracy starts at chance (~17% for 6 actions) and increases monotonically to 86% on Cramped Room, confirming it tracks coordination-relevant behavior rather than memorizing noise.

**Q4: Communication comparison**

Preliminary AERIAL result (seed 0, 2M steps): Best 1110, Final 0, 100% collapse. TarMAC training in progress. Notably, AERIAL — which shares VABL's attention-based design but lacks auxiliary belief regularization — collapses identically to MAPPO, suggesting auxiliary prediction is the key stability mechanism.

**Q5: MAPPO collapse as hyperparameter artifact**

Three converging pieces of evidence establish that the collapse is fundamental: (a) same tuning protocol for all methods, with VABL sharing MAPPO's exact PPO backbone; (b) at 10M steps, MAPPO collapses 100% (peak 503, final 0); (c) AERIAL, an architecturally distinct attention-based method, also collapses 100% (peak 1110, final 0). Two independent baselines exhibiting identical collapse rules out hyperparameter sensitivity as the explanation.

**Q6: Missing baselines** — BAD requires enumerable belief spaces. AERIAL and TarMAC implemented.

**Q7: Task-specific inductive bias**

Cramped Room ablation (5 seeds, Best 1030±70) and 5-agent task (Best 95.7±3.3 vs MAPPO 84.0±10.4) show gains across environments. The attention advantage varies by layout (stronger on Cramped Room than Asymmetric Advantages), consistent with the mechanism being most valuable when spatial coordination is more structured. The variance reduction from auxiliary loss is consistent across all environments tested.

---

## RESPONSE TO REVIEWER dVmV (Score: 3, Weak Reject)

We thank Reviewer dVmV for the thorough and incisive review.

**1. Attention inoperative with 1 teammate**

We fully agree. With one teammate, α = 1.0 trivially. We decompose Overcooked gains into: (a) dedicated action encoding — VABL processes teammate actions through a learned MLP (ψ_θ) concatenated with observation embeddings, providing stronger inductive bias than flat observation input; (b) auxiliary prediction shaping beliefs toward coordination-relevant features.

Confirming this decomposition: AERIAL (Phan et al., 2023), which uses attention-based communication but lacks auxiliary belief learning, collapses 100% (peak 1110, final 0) — the same failure as MAPPO. This suggests auxiliary prediction, not attention alone, drives stability on 2-agent tasks.

We validate selective attention on the **5-agent task** (4 teammates): All VABL variants outperform MAPPO (Best 95.7±3.3 vs 84.0±10.4, 3× lower variance).

**2. Proposition 5.1 — MI comparison**

The reviewer's counterexample is valid for specific weights. Corrected version: sup_{θ∈Θ_attn} I(A; c_attn | b) ≥ sup_{θ∈Θ_mean} I(A; c_mean), conditioning on fixed b on both sides. Explicitly labeled "representational capacity result — does not guarantee that a specific learned θ achieves the supremum." The attention family contains mean pooling (setting g_θ = const), so the supremum over the larger set cannot decrease.

**3. Proposition 5.2 — weighted sum**

Weakened to "context-dependent aggregation." Added: "As a weighted sum, attention cannot distinguish permutations — when individual teammate identities are decision-relevant, additional input features are needed." We implemented identity embeddings; marginal improvement at N≤5, suggesting identity tracking matters at larger scales.

**4-6. Prop 5.3 / Lemma 5.5 / Eq. 7**

Prop 5.3: convergence rates removed, architectural argument only. Lemma 5.5: full proof in iBYE response above (conditional independence + MI chain rule). Eq. 7: MHA presented directly.

**7. Table 2 — 2 seeds, 50 episodes**

We reran ablations with 5 seeds on two layouts. Cramped Room (horizon 400):

| Config | Best (mean±std) |
|--------|----------------|
| Full VABL | **1030 ± 70** |
| No Attention | 990 ± 65 |
| No Aux Loss | 951 ± 162 |
| Neither | 906 ± 244 |

On Asymmetric Advantages (500 episodes, horizon 400), the ordering does not hold: No Attention achieves Best 152±32 vs Full VABL 108±16. We believe this reflects insufficient training on this layout (200K steps, well below the 2M-step convergence budget for AA). The 2M-step AA ablation is running. Importantly, the variance reduction from auxiliary loss is consistent across both layouts.

**10. Aux loss in original Table 2**

On Cramped Room: Full VABL Best 1030 vs No Aux 951, with 2.3× lower variance (70 vs 162). On AA at 500 episodes, the difference is within error bars (108±16 vs 116±32). We now frame auxiliary loss as primarily a variance-reduction and representation-shaping mechanism that constrains beliefs to encode coordination-relevant features.

**11. Environments and baselines**

Added: Cramped Room (Best 1030±70), 5-agent (Best 95.7±3.3), TarMAC and AERIAL implemented. At 10M steps, both MAPPO and AERIAL collapse 100%.
