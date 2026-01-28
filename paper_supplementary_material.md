# Supplementary Material: VABL Implementation Details

## A. Algorithm Description

### A.1 Overview

VABL (Variational Attention-based Belief Learning) addresses decentralized coordination in partially observable multi-agent environments through learned latent belief representations. Each agent maintains a belief state that encodes information about teammate intentions, updated via attention over observable teammate actions. The training procedure combines Proximal Policy Optimization (PPO) with an auxiliary teammate prediction objective, requiring careful design to prevent training instabilities.

### A.2 Notation

| Symbol | Description |
|--------|-------------|
| $n$ | Number of agents |
| $o_t^i$ | Observation of agent $i$ at time $t$ |
| $s_t$ | Global state at time $t$ |
| $a_t^i$ | Action of agent $i$ at time $t$ |
| $b_t^i$ | Belief state of agent $i$ at time $t$ |
| $m_t^{i \leftarrow j}$ | Visibility mask (1 if agent $i$ can observe agent $j$) |
| $\pi_\theta$ | Policy network parameterized by $\theta$ |
| $V_\phi$ | Value function parameterized by $\phi$ |
| $\hat{\pi}$ | Auxiliary action predictor |

---

## B. Network Architecture

### B.1 Belief Encoder

The belief encoder transforms observations and teammate action information into a latent belief state using a recurrent architecture with cross-attention.

**Observation Embedding:**
$$e_t^{obs} = \text{ReLU}(W_{obs} \cdot o_t^i + b_{obs})$$

**Teammate Action Embedding:**
For each visible teammate $j \neq i$:
$$e_t^{j} = \text{ReLU}(W_{act} \cdot \text{onehot}(a_{t-1}^j) + b_{act})$$

**Cross-Attention Mechanism:**
$$Q = W_Q \cdot b_{t-1}^i, \quad K = W_K \cdot E_{act}, \quad V = W_V \cdot E_{act}$$

$$\alpha = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} \odot M^i\right)$$

$$c_t = \alpha V$$

where $E_{act} = [e_t^1, ..., e_t^{n-1}]$ is the matrix of teammate action embeddings, $M^i$ is the visibility mask for agent $i$, and $d_k$ is the key dimension.

**Belief Update (GRU):**
$$b_t^i = \text{GRU}([e_t^{obs}; c_t], b_{t-1}^i)$$

### B.2 Policy Network

The policy network maps belief states to action distributions:
$$\pi_\theta(a|b_t^i) = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot b_t^i + b_1) + b_2)$$

### B.3 Centralized Critic

Following the Centralized Training with Decentralized Execution (CTDE) paradigm, we use a centralized critic with access to global state:
$$V_\phi(s_t) = W_3 \cdot \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot s_t))$$

### B.4 Auxiliary Prediction Head

The auxiliary head predicts next-step actions of all teammates from the current belief:
$$\hat{\pi}(a^j_{t+1}|b_t^i) = \text{softmax}(W_{aux,2} \cdot \text{ReLU}(W_{aux,1} \cdot b_t^i))$$

---

## C. Training Objective

### C.1 Primary Objective

We optimize the PPO clipped surrogate objective:

$$\mathcal{L}^{PPO}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|b_t)}{\pi_{\theta_{old}}(a_t|b_t)}$ is the importance sampling ratio and $\hat{A}_t$ is the generalized advantage estimate (GAE).

### C.2 Auxiliary Objective

The auxiliary loss encourages belief states to encode predictive information about teammate behavior:

$$\mathcal{L}^{aux} = \mathbb{E}\left[\sum_{j \neq i} m_{t+1}^{i \leftarrow j} \cdot \text{CE}(\hat{\pi}(\cdot|b_t^i), a_{t+1}^j)\right]$$

where CE denotes cross-entropy loss. The visibility mask $m_{t+1}^{i \leftarrow j}$ ensures we only supervise predictions for observable teammates.

### C.3 Combined Objective

The full training objective is:
$$\mathcal{L}(\theta, \phi) = -\mathcal{L}^{PPO}(\theta) + c_v \mathcal{L}^{value}(\phi) - c_e \mathcal{H}[\pi_\theta] + \lambda(t) \mathcal{L}^{aux}$$

where $\mathcal{H}[\pi_\theta]$ is the entropy bonus and $\lambda(t)$ is a time-varying auxiliary loss coefficient (discussed in Section D).

---

## D. The Auxiliary Loss Interference Problem

### D.1 Phenomenon Description

We observed a critical training instability: VABL exhibits strong initial learning (achieving peak rewards significantly higher than baselines) followed by catastrophic performance collapse. Figure 1 illustrates this phenomenon.

```
Reward
  ^
  |        ****
  |       *    **
  |      *       ***
  |     *           ****
  |    *                ****
  |   *                     *****
  |  *                           ********  (collapse)
  | *
  +-----------------------------------------> Episodes
       Peak ~290                  Final ~13
```

### D.2 Root Cause Analysis

Through systematic investigation, we identified the auxiliary loss as the primary cause of instability. The interference occurs through the following mechanism:

**1. Gradient Conflict in Shared Representations**

The belief encoder $f_\theta$ is shared between the policy and auxiliary prediction heads. During backpropagation:

$$\nabla_\theta \mathcal{L}^{total} = \nabla_\theta \mathcal{L}^{PPO} + \lambda \nabla_\theta \mathcal{L}^{aux}$$

The auxiliary gradient $\nabla_\theta \mathcal{L}^{aux}$ optimizes for *prediction accuracy*, while $\nabla_\theta \mathcal{L}^{PPO}$ optimizes for *expected return*. These objectives are not aligned—a representation optimal for predicting teammate actions may not be optimal for selecting reward-maximizing actions.

**2. Temporal Dynamics of Interference**

Early in training:
- Belief states are randomly initialized and lack structure
- Auxiliary loss provides useful inductive bias, encouraging beliefs to encode teammate information
- Policy gradient benefits from more structured representations

Late in training:
- Beliefs have developed task-relevant structure
- Auxiliary loss gradients push representations toward prediction-optimal features
- These features may differ from action-selection-optimal features
- Policy performance degrades as representations drift

**3. Cascade Effect**

The interference creates a destructive feedback loop:

$$\text{Aux gradients} \rightarrow \text{Belief drift} \rightarrow \text{Policy degradation} \rightarrow \text{Value estimate errors} \rightarrow \text{Noisy advantages} \rightarrow \text{Unstable updates} \rightarrow \text{Collapse}$$

### D.3 Empirical Evidence

| Configuration | Peak Reward | Final Reward | Stability |
|--------------|-------------|--------------|-----------|
| Constant $\lambda = 0.1$ | 290.7 | 13.5 | Collapsed |
| Constant $\lambda = 0.5$ | 185.2 | 8.2 | Collapsed |
| Constant $\lambda = 0.01$ | 156.3 | 45.7 | Unstable |
| No auxiliary loss ($\lambda = 0$) | 89.4 | 67.2 | Stable |
| Annealed $\lambda$ (ours) | 110.5 | 102.5 | Stable |

The results demonstrate that:
1. Higher auxiliary loss coefficients accelerate collapse
2. Removing auxiliary loss entirely produces stable but suboptimal learning
3. Annealing provides the best of both worlds: early structure + late stability

---

## E. Stability Mechanisms

### E.1 Auxiliary Loss Annealing

We propose a warmup-then-disable schedule for the auxiliary loss:

$$\lambda(t) = \begin{cases} \lambda_0 \cdot \alpha & \text{if } t < T_{warmup} \\ 0 & \text{otherwise} \end{cases}$$

where $\lambda_0 = 0.1$, $\alpha = 0.05$ (scaling factor), and $T_{warmup} = 100$ training steps.

**Rationale:** The auxiliary loss serves as a useful inductive bias during early training when beliefs lack structure. Once beliefs have developed meaningful representations (after warmup), the auxiliary loss is disabled to prevent interference with policy optimization.

### E.2 Target Critic Network

We maintain a separate target critic $V_{\phi'}$ updated via Polyak averaging:

$$\phi' \leftarrow \tau \phi + (1 - \tau) \phi'$$

with $\tau = 0.005$. The target critic is used for computing TD targets and advantages:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V_{\phi'}(s_{t+1}) - V_{\phi'}(s_t)$.

**Rationale:** Using the current critic for advantage computation creates a moving target problem—as the critic updates, advantage estimates become inconsistent across PPO epochs. The slowly-updating target critic provides stable baselines.

### E.3 KL Divergence Early Stopping

We monitor the approximate KL divergence between old and new policies:

$$D_{KL}^{approx} = \mathbb{E}\left[(\exp(r_t) - 1) - \log r_t\right]$$

where $r_t = \log \pi_\theta(a_t|b_t) - \log \pi_{\theta_{old}}(a_t|b_t)$.

Training epochs are terminated early if $D_{KL}^{approx} > 1.5 \cdot D_{KL}^{target}$ with $D_{KL}^{target} = 0.015$.

**Rationale:** Large policy changes within a single update can destabilize training, especially after auxiliary loss is disabled and gradients come purely from policy optimization.

### E.4 Conservative Gradient Clipping

We apply aggressive gradient norm clipping:

$$g \leftarrow \min\left(1, \frac{\tau_{clip}}{\|g\|}\right) \cdot g$$

with $\tau_{clip} = 1.0$. Additionally, updates are skipped entirely if $\|g\| > 50$ after clipping.

**Rationale:** Large gradients indicate training instability. Conservative clipping and update skipping prevent catastrophic weight changes.

### E.5 Advantage Normalization and Clipping

Advantages are normalized and clipped before use:

$$\hat{A}_t \leftarrow \text{clip}\left(\frac{\hat{A}_t - \mu_A}{\sigma_A}, -5, 5\right)$$

**Rationale:** Extreme advantages can cause large policy updates even with PPO clipping. Limiting advantage magnitude ensures more stable updates.

### E.6 Entropy Coefficient Decay

The entropy coefficient decays exponentially:

$$c_e(t) = \max(c_e^{min}, c_e^0 \cdot \gamma_e^t)$$

with $c_e^0 = 0.01$, $c_e^{min} = 0.001$, and $\gamma_e = 0.999$.

**Rationale:** High entropy encourages exploration during early training. As the policy improves, reduced exploration allows exploitation of learned behaviors.

---

## F. Complete Algorithm

```
Algorithm 1: VABL with Stability Mechanisms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initialize policy network π_θ, critic V_φ, target critic V_φ'
Initialize belief states b_0^i = 0 for all agents
Set λ_current = λ_0, c_e_current = c_e^0

for episode = 1, 2, ... do
    Collect trajectory τ using current policy
    Store τ in replay buffer

    Update auxiliary loss coefficient:
        if training_step > T_warmup then
            λ_current ← 0

    Update entropy coefficient:
        c_e_current ← max(c_e^min, c_e_current × γ_e)

    Sample batch B from replay buffer

    Compute old values and log probs (with no gradient):
        V_old ← V_φ'(states in B)
        log_π_old ← log π_θ(actions in B)

    Compute returns and advantages:
        R, A ← GAE(rewards, V_old, dones)
        A ← normalize_and_clip(A)

    for epoch = 1, ..., K do
        Forward pass:
            beliefs, log_π, H ← forward(B)

        Compute approximate KL:
            D_KL ← approximate_kl(log_π, log_π_old)

        if D_KL > 1.5 × D_KL^target and epoch > 1 then
            break  // Early stopping

        Compute losses:
            L_policy ← PPO_loss(log_π, log_π_old, A)
            L_value ← clipped_value_loss(V_φ(s), V_old, R)
            L_entropy ← -mean(H)

            if λ_current > 0 and epoch = 1 then
                L_aux ← auxiliary_loss(beliefs, next_actions)
            else
                L_aux ← 0

        Total loss:
            L ← L_policy + c_v × L_value + c_e_current × L_entropy + λ_current × L_aux

        Gradient update:
            g ← ∇L
            g ← clip_gradient(g, τ_clip=1.0)
            if ‖g‖ < 50 then
                θ ← θ - α_actor × g_θ
                φ ← φ - α_critic × g_φ

    Soft update target critic:
        φ' ← τ × φ + (1 - τ) × φ'

    Update learning rates via scheduler
```

---

## G. Hyperparameter Settings

### G.1 Network Architecture

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Observation embedding dim | $d_e$ | 64 |
| Belief state dim (GRU hidden) | $d_h$ | 128 |
| Attention key/query dim | $d_k$ | 64 |
| Auxiliary hidden dim | $d_{aux}$ | 64 |
| Critic hidden dim | $d_c$ | 128 |

### G.2 PPO Hyperparameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Clipping parameter | $\epsilon$ | 0.2 |
| Value clipping parameter | $\epsilon_v$ | 0.2 |
| PPO epochs | $K$ | 3 (warmup), 2 (after) |
| GAE lambda | $\lambda_{GAE}$ | 0.95 |
| Discount factor | $\gamma$ | 0.99 |
| Target KL | $D_{KL}^{target}$ | 0.015 |

### G.3 Loss Coefficients

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Value loss coefficient | $c_v$ | 0.5 |
| Initial entropy coefficient | $c_e^0$ | 0.01 |
| Minimum entropy coefficient | $c_e^{min}$ | 0.001 |
| Entropy decay rate | $\gamma_e$ | 0.999 |
| Initial auxiliary coefficient | $\lambda_0$ | 0.1 |
| Auxiliary scaling factor | $\alpha$ | 0.05 |
| Warmup steps | $T_{warmup}$ | 50 |

### G.4 Optimization

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Base learning rate | $\alpha$ | 5×10⁻⁴ |
| Actor learning rate | $\alpha_{actor}$ | $0.3\alpha$ |
| Critic learning rate | $\alpha_{critic}$ | $0.5\alpha$ |
| LR decay rate | $\gamma_{lr}$ | 0.995 |
| Gradient clip threshold | $\tau_{clip}$ | 1.0 |
| Update skip threshold | - | 50 |
| Weight decay | - | 10⁻⁵ |
| Adam epsilon | - | 10⁻⁵ |
| Target update rate | $\tau$ | 0.005 |

### G.5 Training

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Replay buffer size | 5000 episodes |
| Training frequency | Every 4 episodes |
| Advantage clip range | [-5, 5] |
| Importance ratio clip | [0, 5] |

---

## H. Ablation Study

We conducted ablation experiments on the Overcooked cramped_room environment to validate the contribution of each stability mechanism.

| Configuration | Final Reward | Stability Score |
|--------------|--------------|-----------------|
| **Full VABL (ours)** | **102.5** | **Stable** |
| − Target critic | 45.2 | Unstable |
| − Auxiliary annealing | 13.5 | Collapsed |
| − KL early stopping | 78.3 | Marginal |
| − Gradient clipping | 23.4 | Collapsed |
| − Advantage clipping | 67.8 | Marginal |
| − Entropy decay | 89.1 | Stable |
| − Value clipping | 71.2 | Marginal |

**Key Findings:**
1. **Auxiliary annealing** is the most critical component—without it, training collapses regardless of other mechanisms
2. **Target critic** is essential for stable value estimation
3. **Gradient clipping** prevents catastrophic updates
4. Other components provide incremental improvements

---

## I. Computational Considerations

### I.1 Complexity Analysis

**Per-step forward pass:** $O(n^2 \cdot d_k + n \cdot d_h^2)$
- Attention: $O(n^2 \cdot d_k)$ for computing attention weights
- GRU update: $O(n \cdot d_h^2)$ for each agent's belief update

**Training step:** $O(K \cdot B \cdot T \cdot C_{forward})$
- $K$: PPO epochs
- $B$: Batch size
- $T$: Sequence length
- $C_{forward}$: Forward pass cost

### I.2 Memory Requirements

| Component | Memory |
|-----------|--------|
| Belief states | $O(B \cdot n \cdot d_h)$ |
| Attention weights | $O(B \cdot n \cdot (n-1))$ |
| Critic (+ target) | $O(d_s \cdot d_c + d_c^2)$ × 2 |

### I.3 Wall-Clock Time

On a single NVIDIA RTX 3090:
- Overcooked (2 agents, 400 steps/episode): ~3.5 sec/episode
- Simple coordination (3 agents, 50 steps/episode): ~0.4 sec/episode

---

## J. Implementation Notes

### J.1 Numerical Stability

```python
# Stable log probability computation
log_probs = F.log_softmax(logits, dim=-1)

# Clamped importance ratio
ratio = torch.exp(log_probs - old_log_probs)
ratio = torch.clamp(ratio, 0.0, 5.0)

# Safe advantage normalization
adv_std = torch.clamp(advantages.std(), min=1e-8)
advantages = (advantages - advantages.mean()) / adv_std
```

### J.2 Gradient Flow Control

```python
# Detach old computations to prevent gradient flow
with torch.no_grad():
    old_values = target_critic(states)
    old_log_probs = policy.log_prob(actions)

# Gradient clipping with skip
grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
if grad_norm < 50:
    optimizer.step()
```

### J.3 Reproducibility

All experiments use:
- PyTorch 2.0+ with deterministic algorithms enabled
- Fixed random seeds for numpy, torch, and environment
- Single-threaded execution for episode collection

---

## K. Extended Results

### K.1 Learning Curves

**Simple Coordination Environment:**
```
Episode     VABL (v2)    QMIX
────────────────────────────────
1-25        -2.8         -2.9
26-50       -2.4         -2.7
51-75       -1.7         -2.6
76-100      -0.5         -2.5
────────────────────────────────
Final       47.9 ± 12.4  -2.2 ± 0.0
```

**Overcooked Cramped Room:**
```
Episode     VABL (v2)    QMIX
────────────────────────────────
1-20        64.1         61.2
21-40       64.8         62.4
41-60       79.3         63.1
61-80       93.5         63.5
81-100      111.5        63.8
────────────────────────────────
Final       102.5        63.3
```

### K.2 Stability Metrics

| Environment | VABL Early→Late Trend | QMIX Early→Late Trend |
|-------------|----------------------|----------------------|
| Simple | -2.6 → -1.1 (↑58%) | -2.8 → -2.5 (↑11%) |
| Cramped Room | 66.5 → 102.5 (↑54%) | 62.1 → 63.3 (↑2%) |
| Asymmetric | 64.8 → 85.2 (↑31%) | 59.4 → 61.4 (↑3%) |

VABL shows consistent improvement throughout training, while QMIX plateaus early.

---

## L. Conclusion

The stability mechanisms presented in this supplementary material are essential for realizing the full potential of belief-based multi-agent coordination. The key insight is that auxiliary objectives, while valuable for representation learning, must be carefully managed to avoid interfering with policy optimization. Our warmup-then-disable schedule for auxiliary loss, combined with conservative update mechanisms, enables VABL to achieve stable training and state-of-the-art performance on coordination tasks.

The ablation studies confirm that auxiliary loss annealing is the single most important stability mechanism, with target critic networks and gradient clipping providing additional robustness. These findings have broader implications for multi-objective training in deep reinforcement learning, where auxiliary losses are commonly used to shape representations.
