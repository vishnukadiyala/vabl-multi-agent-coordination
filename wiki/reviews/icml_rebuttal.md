---
tags: [reviews, rebuttal, icml2026]
status: active
related: [vabl_icml2026, policy_collapse, belief_learning, attention_mechanisms]
---

# ICML 2026 Rebuttal

Paper ID: 30640. Four reviewers.

## Reviewer Summary

| Reviewer | Initial Score | Target | Key Concern |
|----------|--------------|--------|-------------|
| cfx9 | Weak Accept | Solidify/Raise | Statistical rigor, missing baselines |
| iBYE | Reject | Weak Accept | Theory gaps, baseline undertrained |
| 6RAp | Weak Reject | Weak Accept | Generalization, communication comparison |
| dVmV | Weak Reject | Weak Accept | 1-teammate attention, propositions invalid |

## Reviewer cfx9 (Weak Accept)

### Concerns
1. **Ablation statistical rigor**: Only 2 seeds in original submission
2. **Missing baselines**: Why no DICG, TarMAC, BAD?
3. **Attention sparsity**: Do weights become selective?
4. **"Variational" naming**: What does it mean?

### Responses
- Reran all ablations with 5 seeds, 95% CI, Welch's t-test, Cohen's d
- Added AERIAL and TarMAC baselines (DICG incompatible with decentralized execution, BAD requires enumerable belief spaces)
- 5-agent shows VABL best 95.7+/-3.3 vs MAPPO 84.0+/-10.4
- Clarified variational MI via Barber-Agakov framework footnote
- **Honest admission**: On 2-agent Overcooked, attention alpha=1.0 trivially

## Reviewer iBYE (Reject)

### Concerns
1. **Lemma 5.5 proof missing**
2. **Proposition B.2 Step 1**: Lacks expressivity constraint
3. **Proposition B.2 Step 2**: Relevance structure assumption unstated
4. **Proposition 5.3**: Claims convergence rates without support
5. **Overcooked fully observable**: Limits belief-learning claims
6. **MAPPO undertrained**: Needs 5-10M steps
7. **Equation 7 ambiguity**

### Responses
- **Lemma 5.5 proof provided**: Conditional independence + MI chain rule → I(b;s) >= I(b;a) > 0
- **Prop B.2 Step 1**: Supremum over parameter families, explicit conditioning
- **Prop B.2 Step 2**: Relevance structure I(a^k; s | A_S) = 0 now explicit; permutation invariance limitation added
- **Prop 5.3**: Convergence rates **removed entirely**, downgraded to architectural argument
- **Observability**: Reframed as intent inference, not state recovery. Added ego-centric variant.
- **MAPPO 10M**: At 25k episodes, MAPPO collapses 100% (peak 503 → final 0). Extended training makes it worse.
- **Eq 7**: Now shows MHA directly with h=4

## Reviewer 6RAp (Weak Reject)

### Concerns
1. Belief under occlusion
2. Security risks (action observability → deception)
3. Aux loss fitting noise vs real policy
4. Communication method comparison
5. MAPPO collapse — tuning artifact?
6. Missing baselines (BAD, DICG)
7. Task-specific inductive bias

### Responses
- **Occlusion**: GRU retains belief but it becomes stale; quantification is future work
- **Security**: Cooperative Dec-POMDPs only; adversarial resistance is future work
- **Aux accuracy**: Starts at chance (17%), rises monotonically to 86%, plateaus (not 100%) → learns stochastic distribution, not memorization
- **TarMAC comparison**: TarMAC gets highest absolute reward (Final ~54) but requires communication. VABL trades ~48% of TarMAC's reward for eliminating communication.
- **Collapse evidence**: 4 converging pieces — same tuning, 10M collapse, multiple architectures collapse, communication doesn't solve it
- **MAPPO sensitivity**: Entropy schedules reduce collapse from 76% to 68-72%, still much worse than VABL's 38%
- **Generalization**: Cramped Room (1030+/-70) and 5-agent (95.7+/-3.3) show gains across environments

## Reviewer dVmV (Weak Reject)

### Concerns (11 specific points)
1. Attention inoperative with 1 teammate (alpha=1.0)
2. Proposition 5.1 MI comparison has valid counterexample
3. Proposition 5.2 weighted sum can't distinguish permutations
4. Proposition 5.3 convergence rates unjustified
5. Lemma 5.5 proof missing
6. Equation 7 inconsistency
7. Table 2 only 2 seeds, 50 episodes
8. No identity information (agents indistinguishable)
9. MAAC and Phan et al. citations missing
10. Aux loss appears to hurt in Table 2
11. Environment and baseline gaps

### Responses
- **Point 1**: Fully agreed. Decomposed gains into action encoding + auxiliary prediction. Evidence: AERIAL (attention, no aux) collapses 100%.
- **Points 2-5**: See iBYE responses above (all theoretical claims corrected/weakened)
- **Point 6**: Eq 7 now shows MHA directly
- **Point 7**: Reran with 5 seeds, 500 episodes, 95% CI
- **Point 8**: Identity embeddings implemented; mixed results at N<=5, may matter at larger scales
- **Point 9**: MAAC (Iqbal & Sha) and AERIAL (Phan et al.) citations added
- **Point 10**: Full VABL best 1030 vs No Aux best 951 with 2.3x lower variance. Framed as variance-reduction mechanism.
- **Point 11**: Added Cramped Room, 5-agent, ego-centric, TarMAC, AERIAL

## Theoretical Corrections Summary

| Component | Change |
|-----------|--------|
| Lemma 5.5 | Full proof added (was missing) |
| Proposition 5.1 | Supremum over families; "representational capacity" label |
| Proposition 5.2 | Permutation invariance limitation; identity embeddings needed |
| Proposition 5.3 | O(gamma^k) rates removed; architectural argument only |
| Equation 7 | Now explicit MHA with h=4 heads |

## Open Issues Acknowledged in Rebuttal

1. Scalability beyond N=5 untested
2. Perfect action observability assumed
3. Cross-play generalization untested
4. Hanabi evaluation not yet in camera-ready
5. Belief degradation under occlusion unquantified
6. Mixed-motive/adversarial settings unaddressed
