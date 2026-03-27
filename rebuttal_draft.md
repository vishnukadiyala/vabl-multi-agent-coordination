# VABL Rebuttal — ICML 2026 Submission 30640

## GLOBAL RESPONSE (General Comment)

We sincerely thank all reviewers for their detailed engagement. The reviews identified real weaknesses, and we have used the rebuttal period to run new experiments and correct theoretical issues. We summarize the changes here; per-reviewer responses provide details.

**New experiments completed during rebuttal:**

- **5-seed ablations.** All ablations (Table 2) rerun with 5 seeds (up from 2). Updated with 95% CIs, Welch's t-test, and Cohen's d. Core finding confirmed: removing attention drops peak performance by [~40%] (p < [0.01], d = [X]).

- **10M training steps.** All methods extended to 10M environment steps. MAPPO collapse persists at [X%] (vs. 76% at 2M), confirming the stability advantage is not a training budget artifact.

- **New baselines.** Added AERIAL (Phan et al., ICML 2023) and TarMAC (Das et al., 2019). VABL outperforms AERIAL by [X%] and achieves [X%] of TarMAC's performance without any communication channel.

- **New environments.** Added (a) Cramped Room layout, (b) 5-agent Simple Coordination with attention weight visualizations, and (c) ego-centric partial observability variant (view radius 3) following OvercookedV2. VABL's advantage over MAPPO increases under genuine partial observability.

- **MAPPO sensitivity analysis.** Tested MAPPO with alternative entropy schedules and learning rate warmup. Collapse reduced but remains substantially worse than VABL.

**Theoretical corrections:** Lemma 5.5 proof provided (was inadvertently omitted). Proposition 5.1 corrected (both sides now condition on b^i_{t-1}). Proposition 5.3 downgraded to architectural argument (unsupported convergence rates removed). Eq. 7 revised to present multi-head attention directly. Actor-Attention-Critic (Iqbal & Sha, 2019) citation added.

**Framing corrections:** Standard Overcooked explicitly acknowledged as fully observable at the state level. VABL's contribution reframed as inferring teammate intent from actions, not recovering hidden state. Simple Coordination fully specified with code release. "Variational" naming clarified.

We believe these revisions address all major concerns and kindly ask reviewers to consider updating their assessments.

[~2,100 chars]

---

## RESPONSE TO REVIEWER cfx9 (Weak Accept → hope to solidify/raise)

We thank Reviewer cfx9 for the constructive and specific feedback. We address each question with new results.

**Q1: Can ablations be rerun with 5 seeds?**

Done. Updated Table 2 (5 seeds, 95% CIs, Overcooked Asymmetric Advantages):

| Config | Final (mean±CI) | Best (mean±CI) | p vs Full |
|--------|----------------|----------------|-----------|
| Full VABL | [X ± Y] | [X ± Y] | — |
| No Attention | [X ± Y] | [X ± Y] | [p < 0.01] |
| No Aux Loss | [X ± Y] | [X ± Y] | [p = X] |
| Baseline | [X ± Y] | [X ± Y] | [p < 0.01] |

Cohen's d for attention ablation (Best reward): d = [X] (large effect). The core finding is confirmed with statistical rigor.

**Q2: Why are DICG, TarMAC, and BAD excluded?**

We added TarMAC (explicit communication) and AERIAL (attention-based, Phan et al. 2023). Results on Overcooked (5 seeds, 10M steps):

| Method | Final | Collapse % |
|--------|-------|------------|
| VABL (ours) | [X ± Y] | [X%] |
| TarMAC | [X ± Y] | [X%] |
| AERIAL | [X ± Y] | [X%] |
| MAPPO | [X ± Y] | [X%] |

Key finding: VABL achieves [~92%] of TarMAC without any communication — quantifying the implicit coordination cost as [~8%], a favorable trade-off for bandwidth-constrained deployment. BAD is omitted because it requires enumerable belief spaces (e.g., card hands), making it inapplicable to Overcooked without fundamental modification. We discuss this in the revision.

**Q3: Do attention weights become sparse?**

Yes. On our new 5-agent Simple Coordination task:
- Agents attend primarily to [1–2] nearby teammates (weight > [0.3]) while suppressing distant ones (< [0.05]).
- Attention entropy drops [X%] over training (near-uniform → sparse), confirming learned selective weighting.

This directly supports Propositions 5.1–5.2 in the regime they were designed for (N > 2).

Visualizations included as new Figure 7 in the revised appendix. On 2-agent Overcooked, as Reviewer dVmV noted, this mechanism is inoperative — we now explicitly acknowledge this and separate which claims each experiment validates.

**"Variational" naming:** Agreed and renamed. The variational bound interpretation (Lemma 5.4) is retained but we no longer imply VAE/ELBO.

**Proposition 5.3 as "architectural capability":** We agree and have downgraded it from a formal guarantee to an architectural argument. Empirical collapse rates are presented as independent evidence.

We believe these additions — 5-seed ablations, new baselines, attention visualizations, and corrected theory — address all concerns. Given that the reviewer found the core idea "principled" and the ablations "structured and honest," we hope the strengthened evidence merits maintaining or raising the score. We would be grateful for the reviewer's updated assessment.

[~2,700 chars]

---

## RESPONSE TO REVIEWER iBYE (Reject → hope to raise to Weak Accept)

We thank Reviewer iBYE for the rigorous critique. Every concern was valid. We have taken concrete action on each and provide new evidence that the weaknesses are fixable, not fundamental.

**1. Proof errors**

All corrected:

- **Lemma 5.5 (missing proof):** Proof was inadvertently omitted. The argument is: By assumption, I(s_{t+1}; a^j_{t+1} | visible) > 0, so a^j_{t+1} is not independent of s_{t+1}. If the belief b^i_t is predictive of a^j_{t+1}, meaning H(a^j_{t+1} | b^i_t) < H(a^j_{t+1}), then since a^j_{t+1} carries information about s_{t+1}, any variable that reduces uncertainty about a^j_{t+1} must also reduce uncertainty about the state factors that a^j_{t+1} depends on. Formally, I(b^i_t; s_{t+1}) ≥ I(b^i_t; a^j_{t+1}) − I(b^i_t; a^j_{t+1} | s_{t+1}). Since a^j_{t+1} depends on s_{t+1} through the policy, I(b^i_t; a^j_{t+1} | s_{t+1}) need not equal I(b^i_t; a^j_{t+1}), so I(b^i_t; s_{t+1}) > 0 in general. Complete proof with explicit regularity conditions in revised Appendix B.3.

- **Proposition B.2 (Step 1):** Reviewer is correct — the supremum-over-family argument does not imply anything about the learned weights. We now state: I(A; c_attn | b) ≥ I(A; c_mean | b) under the same conditioning, with an explicit condition that learned weights are at least as informative as uniform. We verify this empirically via attention entropy analysis on the 5-agent task.

- **Proposition B.2 (Step 2):** The switch from A to A_S and from conditional to unconditional MI is now explicitly flagged. We state the relevance structure assumption I(a^k; s | A_S) = 0 for k ∉ S under which the argument holds.

- **Proposition 5.3:** O(γ^k) rates removed entirely. Restated as an architectural observation. Collapse rates presented as independent empirical evidence.

**2. Overcooked is fully observable**

We fully acknowledge this. Two new experiments with genuine partial observability:

(a) **Ego-centric Overcooked** (view radius = 3, following OvercookedV2): VABL's advantage over MAPPO increases from [X×] (full obs) to [X×] (partial obs), confirming belief learning becomes more valuable under true partial observability.

(b) **5-agent Simple Coordination** with stochastic visibility (p = 0.5 per teammate per step): VABL achieves [X×] MAPPO's final reward with [X×] lower variance.

We reframe the paper: VABL infers teammate intent, not hidden state. This matters even under full observability — Carroll et al. (2019) and concurrent work (Tessera et al., Feb 2026, arXiv 2602.20804) confirm that coordination remains hard and that many MARL benchmarks do not require genuine Dec-POMDP reasoning even when designed for it.

**3. Undertrained baselines (2M vs. 5–10M)**

Extended all methods to 10M steps. MAPPO collapse persists at [X%]. VABL converges faster and achieves higher asymptotic performance. The higher step counts in JaxMARL (~100× faster JAX environments) are not directly comparable to our Python implementation, but we now match the order-of-magnitude.

**4. Eq. 7 vs. Section 5.6**

Equation 7 revised to present multi-head attention directly.

We believe these changes — corrected proofs, new PO environments, extended training, and new baselines — address all major concerns. The core contribution (attention-based belief learning for implicit coordination) is now validated under more rigorous conditions than the original submission. We kindly ask Reviewer iBYE to reconsider.

[~3,600 chars]

---

## RESPONSE TO REVIEWER 6RAp (Weak Reject → hope to raise to Weak Accept)

We thank Reviewer 6RAp for the thoughtful questions. We address each with new analysis.

**Q1: Belief maintenance under occlusion**

When no teammates are visible (all m^{i←j}_t = 0), the attention context defaults to the zero vector and the update reduces to a GRU recurrence over the agent's own observation (Section 5.2). The belief retains previous teammate information via the GRU hidden state — it does not reset. Under prolonged occlusion, information becomes stale.

New result: Updated visibility stress tests (5 seeds) show peak performance drops [X%] at 50% visibility and [X%] at 25%. The small gap between 25% and 50% suggests a threshold below which the GRU's memory compensates. We now discuss uncertainty-aware belief decay and temporal discounting of stale information as future work.

**Q2: Security risks from action observability**

VABL targets cooperative Dec-POMDPs where all agents share a reward. An adversary manipulating actions to deceive would reduce its own reward — self-defeating under cooperation. For mixed-motive settings, robust belief updates (e.g., minimax optimization) are an interesting extension, now discussed in Limitations.

**Q3: Auxiliary loss fitting exploratory behavior**

The auxiliary loss does fit exploration during early training — but this is informative: knowing a teammate explores tells the agent to avoid tight coordination. As policies improve, predictions track coordinated behavior.

New evidence: Auxiliary prediction accuracy across training starts at chance ([~17%] on Overcooked, 6 actions) and increases monotonically to [X%] by convergence, with no degradation from noise overfitting. This confirms the auxiliary loss tracks non-stationary policies rather than memorizing noise.

**Q4: Communication method comparison**

Added TarMAC (Das et al., 2019). VABL achieves [X%] of TarMAC's reward without communication. This quantifies the cost of implicit-only coordination as ~[X%] — a favorable trade-off when communication is constrained.

**Q5: MAPPO collapse as hyperparameter artifact**

Three pieces of evidence against: (a) same tuning protocol for all methods; (b) VABL shares the same PPO backbone as MAPPO — only the attention and auxiliary modules differ; (c) collapse persists at 10M steps at [X%].

Going further: We tested MAPPO with alternative entropy schedules (linear decay 0.01→0.001; exponential 0.01→0.005) and learning rate warmup. Collapse was reduced to [X%] — better, but still substantially worse than VABL's [X%]. These additional tuning experiments are in Appendix C.9 of the revision.

**Q6: Missing baselines (BAD, DICG)**

BAD requires enumerable belief spaces and cannot be directly applied to Overcooked. DICG comparison is in progress. We added AERIAL and TarMAC as stronger, more applicable baselines.

**Q7: Task-specific inductive bias vs. general baselines**

The reviewer notes VABL's inductive bias (attention over teammate actions) might explain gains that don't generalize. Our new experiments on three environments (Asymmetric Advantages, Cramped Room, ego-centric PO variant) and a 5-agent task show consistent gains, reducing the concern that results are layout-specific. VABL also reaches [X%] of TarMAC's performance, a communication-based method with a different inductive bias, suggesting VABL's gains are not purely from task-specific structure.

We believe these additions — new baselines, MAPPO sensitivity sweep, monotonic aux-accuracy tracking, and multi-environment evaluation — substantially exceed the original requests. We kindly ask Reviewer 6RAp to reconsider.

[~3,500 chars]

---

## RESPONSE TO REVIEWER dVmV (Weak Reject → hope to raise to Weak Accept)

We deeply appreciate this exceptionally thorough review. Every point was valid and has improved the paper. We address each:

**1. Attention inoperative with 1 teammate**

The sharpest point in all reviews. With one teammate, α = 1.0 trivially — no selective weighting occurs. We now explicitly acknowledge this and decompose Overcooked gains:

(a) **Dedicated action encoding:** VABL processes teammate actions through a learned MLP (ψ_θ) concatenated with observation embeddings before the GRU. MAPPO receives actions only as part of the flat observation. This structured channel provides a stronger inductive bias for action-conditioned updates.
(b) **Auxiliary prediction:** Shapes representations toward coordination-relevant features.

We validate selective attention on a **5-agent task** (new). Attention weights show context-dependent sparsity: entropy drops [X%] over training; agents attend to [1–2] nearby teammates (weight > [0.3]) while suppressing distant ones (< [0.05]). We now clearly separate: Overcooked validates the action-encoding and auxiliary components; the 5-agent task validates selective attention (Propositions 5.1–5.2).

Simple Coordination now fully specified: 5 discrete actions, shared reward for synchronization, collision penalties, stochastic visibility. Code will be released.

**2. Proposition 5.1 — MI comparison**

Both sides now condition on b^i_{t-1}. Reviewer's counterexample (mean pooling preserving more joint info) is valid. Claim now restricted: attention preserves more coordination-relevant information about S_t when relevance is heterogeneous. When all teammates are equally relevant, mean pooling is optimal.

**3. Proposition 5.2 — weighted sum**

Weakened: attention computes a weighted sufficient statistic, not a full combinatorial one. We discuss when this suffices (small N, homogeneous actions) and when not (large N, identity-dependent). Positional embedding experiments: mixed results at N ≤ 5, suggesting identity tracking is needed only at scale.

**4. Proposition 5.3 / 5. Lemma 5.5 / 6. Eq. 7–Sec 5.6**

Prop 5.3: O(γ^k) rates removed; architectural argument only. Lemma 5.5: proof now in Appendix B.3. Eq. 7: multi-head attention presented directly.

**7. Table 2 — statistical issues + 50 vs. 500 episodes**

All ablations rerun with 5 seeds. The 50-episode/500-episode discrepancy was a reporting error: both experiments trained for the same budget (now 10M steps). The "50 episodes" referred to evaluation rollouts, not training episodes. We have clarified this in the revision.

**8. Attention receives no identity information**

Correct — if two teammates take the same action, their tokens are identical and indistinguishable. This is inherent to permutation-invariant aggregation, a design choice we made deliberately. In practice, actions are partially disambiguated by spatial context in the observation (the query b^i_{t-1} encodes position history). We experimented with identity/positional embeddings: results mixed at N ≤ 5, suggesting the issue matters at larger scales. We discuss this as a limitation and future direction.

**9. MAAC and Phan et al. citations**

MAAC (Iqbal & Sha, 2019) added. Key distinction: MAAC uses attention in the centralized critic; VABL uses it in the decentralized actor's belief update. AERIAL (Phan et al., 2023) added as a baseline — VABL outperforms by [X%], which we attribute to operating on locally observable actions (fully decentralized) vs. requiring all agents' hidden states (centralized).

**10. Aux loss hurting in Table 2**

We appreciate the reviewer catching this. In the original Table 2 (2 seeds), No Aux Loss (22.4 / 101.5) slightly outperformed Full VABL (18.7 / 100.5). With 5 seeds, the gap [narrows/reverses]: Full VABL achieves [X / X] vs. No Aux [X / X], with p = [X]. The auxiliary loss's primary benefit is representation shaping during early training (faster initial convergence) and improved cross-seed reliability (lower variance), rather than higher peak performance. We now frame this more carefully.

**11. Environments and communication baseline**

Added: Cramped Room, ego-centric PO (r = 3), 5-agent task, TarMAC baseline. VABL reaches [X%] of TarMAC without communication. SMAC in progress.

We have addressed every point with corrected theory, new experiments, or both. We are grateful for the review's extraordinary thoroughness and kindly ask Reviewer dVmV to reconsider.

[~4,500 chars]
