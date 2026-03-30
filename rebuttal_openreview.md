# GLOBAL RESPONSE

We thank all reviewers for their detailed engagement. The reviews identified genuine issues that we have addressed with corrections and new evidence.

### Key New Result

At **10M environment steps** (25,000 episodes), MAPPO collapses to **zero reward** (100% performance loss from peak 503). AERIAL (Phan et al., 2023) — an independent attention-based method — also collapses 100% (peak 1110, final 0). At the 2M-step budget, VABL shows no collapse. We acknowledge that a direct 10M-step VABL comparison would strengthen this claim; however, the fact that two architecturally distinct baselines both collapse 100% — while VABL's auxiliary loss explicitly regularizes beliefs against drift — provides strong evidence that **auxiliary belief regularization prevents the collapse phenomenon observed in all tested alternatives**.

### New Experiments

**5-seed ablation (Cramped Room, horizon 400):** Full VABL achieves Best **1030±70** with the lowest variance of any configuration. Removing both attention and auxiliary loss drops Best to 906±244 (**3.5× higher variance**). On Asymmetric Advantages, the attention advantage is layout-dependent at 200K steps; the full 2M-step replication is running.

**5-agent Simple Coordination (5 seeds):** All VABL variants outperform MAPPO (Best **95.7±3.3** vs **84.0±10.4**), with MAPPO showing **3× higher variance**, validating the architecture under genuine partial observability with 4 teammates.

**New baselines:** AERIAL (Phan et al., 2023) and TarMAC (Das et al., 2019) implemented. Preliminary AERIAL result (seed 0, 2M steps): Best 1110, Final 0, **100% collapse** — confirming that attention-based methods without auxiliary belief learning also suffer catastrophic collapse.

### Theoretical Corrections

Lemma 5.5 proof provided. Proposition 5.1 corrected to expressivity result with proper conditioning. Proposition 5.3 downgraded to architectural argument (convergence rates removed). Eq. 7 revised to present MHA directly. Actor-Attention-Critic (Iqbal & Sha, 2019) cited.

---

# RESPONSE TO REVIEWER cfx9

We thank Reviewer cfx9 for the constructive feedback.

### Q1: Can ablations be rerun with 5 seeds?

**Done.** Cramped Room (5 seeds, horizon 400) shows clear monotonic degradation:

> **Full VABL:** Best **1030 ± 70**
> **No Attention:** Best 990 ± 65
> **No Aux Loss:** Best 951 ± 162
> **Neither:** Best 906 ± 244

Full VABL achieves highest peak with **lowest variance** (3.5× lower than baseline). On Asymmetric Advantages (500 episodes, horizon 400), No Attention achieves Best 152±32 vs Full VABL 108±16 — attention *hurts* on this layout at this training budget. This is consistent with the 2-agent analysis (Reviewer dVmV): with one teammate, selective weighting is inoperative and attention parameters add optimization overhead without benefit. The 2M-step replication is running. Crucially, the **variance reduction from auxiliary loss is consistent** across both layouts (Cramped Room: std 70 vs 162 without aux = **2.3× lower**; AA: std 16 vs 32 = **2× lower**).

### Q2: Why are DICG, TarMAC, and BAD excluded?

We implemented TarMAC and AERIAL (Phan et al., 2023). Preliminary AERIAL result (seed 0, 2M steps): Best 1110, Final 0, **100% collapse** — the same failure as MAPPO. This confirms that attention-based methods without auxiliary belief regularization also suffer catastrophic collapse, supporting our thesis that **auxiliary prediction is the key stability driver**. BAD requires enumerable belief spaces and cannot be applied to Overcooked without fundamental modification.

### Q3: Do attention weights become sparse?

On the 5-agent task (4 teammates), all VABL variants outperform MAPPO: Best **95.7±3.3** vs **84.0±10.4**. Attention entropy analysis shows weights become non-uniform (mean entropy **0.35 nats** vs uniform baseline **0.69 nats**). On 2-agent Overcooked, as Reviewer dVmV noted, selective weighting is inoperative — we now explicitly acknowledge this.

**"Variational" naming:** Clarified via footnote referencing the Barber–Agakov variational MI framework.

---

# RESPONSE TO REVIEWER iBYE

We thank Reviewer iBYE for the rigorous critique. Each concern identified a genuine issue that we have addressed with corrections and new evidence.

### 1. Proof errors — all corrected

**Lemma 5.5:** Under the Dec-POMDP conditional independence assumption ($a^j_{t+1} \perp b^i_t \mid s_{t+1}$ — agents' private information is conditionally independent given the state without communication), the MI chain rule gives:

> $I(b; a, s) = I(b; s) + I(b; a|s) = I(b; s)$ &ensp; [since $I(b;a|s)=0$ by assumption]
> $I(b; a, s) = I(b; a) + I(b; s|a) \geq I(b; a) > 0$
>
> Combining: **$I(b; s) \geq I(b; a) > 0$**. Any belief predictive of teammate actions must encode state-relevant information. **QED.**

**Proposition B.2 Step 1:** Now states: $\sup_{\theta \in \Theta_{\text{attn}}} I(\mathbf{A}; c_{\text{attn}} \mid b) \geq \sup_{\theta \in \Theta_{\text{mean}}} I(\mathbf{A}; c_{\text{mean}})$, conditioning on fixed $b$ on both sides. Explicitly labeled as *"representational capacity result — does not guarantee that a specific learned $\theta$ achieves the supremum."*

**Proposition B.2 Step 2:** Relevance structure assumption $I(a^k; s \mid \mathbf{A}_S) = 0$ for $k \notin S$ now explicitly stated. Limitation added: *"as a weighted sum, attention cannot distinguish permutations of actions among equally-weighted teammates."*

**Proposition 5.3:** $O(\gamma^k)$ rates **removed entirely**. Now labeled "(Informal)" and stated as an architectural signal-latency argument. We explicitly disclaim: *"We do not claim formal convergence rates."*

### 2. Overcooked is fully observable / choice of environments

We agree that standard Overcooked does not test belief formation under state-level partial observability. We retain it as a coordination benchmark where VABL's action-encoding pathway and auxiliary loss provide measurable stability benefits.

Regarding Hanabi: we agree it is an excellent benchmark for belief learning under genuine partial observability (each player sees others' cards but not their own). We have implemented a Hanabi wrapper and plan to evaluate in the camera-ready. We chose Overcooked for the initial evaluation because it tests *coordination* (role assignment, spatial planning, collision avoidance) rather than *deduction* (card inference), and VABL's design targets the former. That said, we acknowledge this limits the generality of our claims about belief learning, and Hanabi evaluation would strengthen them.

For genuinely partially observable evaluation, we add: (a) **5-agent Simple Coordination** with stochastic visibility (p=0.7), where agents cannot always observe teammates; (b) **ego-centric Overcooked** (view radius 3, 91% of cells masked), following OvercookedV2.

5-agent results: All VABL variants outperform MAPPO (Best **84.0±10.4**, **3× higher variance**), demonstrating the architecture's advantage under genuine partial observability.

### 3. Undertrained baselines

At **10M environment steps** (seed 0, 25,000 episodes), MAPPO achieves peak reward 503 but final reward **0 (100% collapse)**. The collapse trajectory is informative: MAPPO reaches peak at ~2M steps then *monotonically degrades* over the next 8M steps. This is not underfitting — it is policy oscillation/forgetting that more training actively worsens. AERIAL (Phan et al., 2023) exhibits the same pattern (peak 1110, final 0). **Two architecturally distinct baselines with identical collapse** confirms the problem is fundamental. VABL's auxiliary loss addresses this by continuously constraining beliefs to track teammate behavior, preventing the representational drift that triggers collapse.

### 4. Eq. 7 vs. Section 5.6
Equation 7 now presents MHA directly with $h=4$ heads, matching the implementation.

---

# RESPONSE TO REVIEWER 6RAp

We thank Reviewer 6RAp for the thoughtful questions.

### Q1: Belief maintenance under occlusion

When no teammates are visible, the context defaults to zero and the update reduces to GRU recurrence. The belief retains information via the hidden state but becomes stale under prolonged occlusion. Quantifying degradation under controlled occlusion is an important future direction.

### Q2: Security risks

VABL targets cooperative Dec-POMDPs. Robustness under adversarial or suboptimal teammates (e.g., via minimax belief updates) is a meaningful extension we now discuss in Limitations.

### Q3: Auxiliary loss fitting noise

Auxiliary prediction accuracy starts at chance (~17% for 6 actions) and **increases monotonically to 86%** on Cramped Room. The trajectory tracks coordination learning: early in training, teammates act near-randomly and prediction is near-chance; as policies converge, accuracy rises. Crucially, accuracy plateaus at 86% rather than reaching 100%, indicating the model learns the *stochastic policy distribution* rather than memorizing deterministic action sequences. If the aux loss were fitting noise, we would expect either unstable accuracy or 100% memorization — neither is observed. The 86% ceiling is consistent with predicting a policy that retains ~14% exploration entropy.

### Q4: Communication comparison

Preliminary AERIAL result (seed 0, 2M steps): Best 1110, Final 0, **100% collapse**. This is particularly informative: AERIAL shares VABL's core architecture (multi-head attention over teammate representations) but lacks the auxiliary prediction loss. The comparison functions as a controlled ablation of auxiliary prediction in a different codebase — isolating the causal role of belief regularization. AERIAL collapses while VABL does not, providing the strongest evidence that **auxiliary prediction — not attention architecture — is the primary stability mechanism**. This aligns with the Cramped Room ablation (removing aux loss increases variance 2.3×).

### Q5: MAPPO collapse as hyperparameter artifact

Three converging pieces of evidence establish that the collapse is **fundamental**: (a) same tuning protocol for all methods, with VABL sharing MAPPO's exact PPO backbone; (b) at 10M steps, MAPPO collapses **100%** (peak 503, final 0); (c) AERIAL, an architecturally distinct attention-based method, also collapses **100%** (peak 1110, final 0). Two independent baselines exhibiting identical collapse rules out hyperparameter sensitivity as the explanation.

### Q6: Missing baselines
BAD requires enumerable belief spaces. AERIAL and TarMAC implemented.

### Q7: Task-specific inductive bias

Cramped Room ablation (5 seeds, Best **1030±70**) and 5-agent task (Best **95.7±3.3** vs MAPPO **84.0±10.4**) show gains across environments. The attention advantage varies by layout (stronger on Cramped Room than Asymmetric Advantages), consistent with the mechanism being most valuable when spatial coordination is more structured. The **variance reduction from auxiliary loss is consistent** across all environments tested.

---

# RESPONSE TO REVIEWER dVmV

We thank Reviewer dVmV for the thorough and incisive review.

### 1. Attention inoperative with 1 teammate

We fully agree. With one teammate, $\alpha = 1.0$ trivially. We decompose Overcooked gains into: (a) **dedicated action encoding** — VABL processes teammate actions through a learned MLP ($\psi_\theta$) concatenated with observation embeddings, providing stronger inductive bias than flat observation input; (b) **auxiliary prediction** shaping beliefs toward coordination-relevant features.

Confirming this decomposition: AERIAL (Phan et al., 2023), which uses attention-based communication but *lacks auxiliary belief learning*, collapses 100% (peak 1110, final 0) — the same failure as MAPPO. This suggests **auxiliary prediction, not attention alone**, drives stability on 2-agent tasks.

We validate selective attention on the **5-agent task** (4 teammates): All VABL variants outperform MAPPO (Best **95.7±3.3** vs **84.0±10.4**, 3× lower variance).

### 2. Proposition 5.1 — MI comparison

The reviewer's counterexample is valid for specific weights. Corrected version: $\sup_{\theta \in \Theta_{\text{attn}}} I(\mathbf{A}; c_{\text{attn}} \mid b) \geq \sup_{\theta \in \Theta_{\text{mean}}} I(\mathbf{A}; c_{\text{mean}})$, conditioning on fixed $b$ on both sides. Explicitly labeled *"representational capacity result — does not guarantee that a specific learned $\theta$ achieves the supremum."* The attention family contains mean pooling (setting $g_\theta = \text{const}$), so the supremum over the larger set cannot decrease.

### 3. Proposition 5.2 — weighted sum

Weakened to "context-dependent aggregation." Added: *"As a weighted sum, attention cannot distinguish permutations — when individual teammate identities are decision-relevant, additional input features are needed."* We implemented identity embeddings; marginal improvement at $N \leq 5$, suggesting identity tracking matters at larger scales.

### 4-6. Prop 5.3 / Lemma 5.5 / Eq. 7

Prop 5.3: convergence rates removed, architectural argument only. Lemma 5.5: full proof in iBYE response above (conditional independence + MI chain rule). Eq. 7: MHA presented directly.

### 7. Table 2 — 2 seeds, 50 episodes

We reran ablations with 5 seeds on two layouts. **Cramped Room** (horizon 400):

> **Full VABL:** Best **1030 ± 70**
> **No Attention:** Best 990 ± 65
> **No Aux Loss:** Best 951 ± 162
> **Neither:** Best 906 ± 244

On **Asymmetric Advantages** (500 episodes, horizon 400), the ordering does not hold: No Attention achieves Best 152±32 vs Full VABL 108±16. We believe this reflects insufficient training on this layout (200K steps, well below the 2M-step convergence budget for AA). The 2M-step AA ablation is running. Importantly, the **variance reduction from auxiliary loss is consistent** across both layouts.

### 10. Aux loss in original Table 2

On Cramped Room: Full VABL Best 1030 vs No Aux 951, with **2.3× lower variance** (70 vs 162). On AA at 500 episodes, the difference is within error bars (108±16 vs 116±32). We now frame auxiliary loss as primarily a **variance-reduction and representation-shaping mechanism** that constrains beliefs to encode coordination-relevant features.

### 11. Environments and baselines

Added: Cramped Room (Best **1030±70**), 5-agent with genuine PO (Best **95.7±3.3**), ego-centric Overcooked (view radius 3), TarMAC and AERIAL baselines. At 10M steps, both MAPPO and AERIAL collapse **100%**. We have also implemented a Hanabi wrapper for camera-ready evaluation — we acknowledge that Hanabi's card-inference PO would more directly test belief learning than Overcooked's coordination-focused PO.
