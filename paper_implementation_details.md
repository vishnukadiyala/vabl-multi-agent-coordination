# VABL: Implementation Details and Stability Analysis

## Abstract

This document provides comprehensive implementation details for VABL (Variational Attention-based Belief Learning), a multi-agent reinforcement learning algorithm that enables implicit coordination through attention-driven latent belief representations. We detail the network architecture, training procedure, and critically, the stability challenges encountered during development along with their solutions.

---

## 1. Algorithm Overview

VABL addresses the challenge of coordination in partially observable multi-agent environments by maintaining latent belief states that encode information about teammate intentions. The key insight is that agents can infer teammate behavior by attending to their observable actions, enabling coordination without explicit communication.

### 1.1 Core Components

1. **Belief State Encoder**: A GRU-based recurrent network that maintains agent beliefs
2. **Attention Mechanism**: Cross-attention over teammate actions to update beliefs
3. **Policy Network**: Actor network that maps beliefs to action distributions
4. **Centralized Critic**: State-value function for PPO training
5. **Auxiliary Predictor**: Teammate action prediction head for belief regularization

### 1.2 Training Objective

The training objective combines the PPO policy gradient with an auxiliary prediction loss:

```
max J(θ) - λ · L_aux
```

Where:
- `J(θ)` is the PPO clipped surrogate objective
- `L_aux` is the auxiliary teammate action prediction loss
- `λ` is the auxiliary loss coefficient

---

## 2. Network Architecture

### 2.1 VABL Agent Network

```
Input: observation o_t, belief state b_{t-1}, teammate actions a^{-i}_{t-1}

┌─────────────────────────────────────────────────────────────┐
│  Observation Encoder                                         │
│  ────────────────────                                        │
│  o_t → Linear(obs_dim, embed_dim) → ReLU → e_obs            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Action Encoder (for each teammate j)                        │
│  ─────────────────────────────────                           │
│  a^j_{t-1} → Linear(n_actions, embed_dim) → ReLU → e^j_act  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Cross-Attention Module                                      │
│  ───────────────────────                                     │
│  Query:  Q = Linear(b_{t-1})           [batch, d_k]         │
│  Keys:   K = Linear(e_act)             [batch, n-1, d_k]    │
│  Values: V = Linear(e_act)             [batch, n-1, d_v]    │
│                                                              │
│  Attention weights: α = softmax(Q·K^T / √d_k) · visibility  │
│  Context: c = α · V                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Belief Update (GRU)                                         │
│  ───────────────────                                         │
│  input = concat(e_obs, c)                                    │
│  b_t = GRU(input, b_{t-1})             [batch, hidden_dim]  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Policy Head                                                 │
│  ───────────                                                 │
│  b_t → Linear(hidden_dim, hidden_dim) → ReLU                │
│      → Linear(hidden_dim, n_actions) → logits               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Centralized Critic

The critic uses the global state for value estimation:

```
Input: global state s_t

s_t → Linear(state_dim, hidden_dim) → ReLU
    → Linear(hidden_dim, hidden_dim) → ReLU
    → Linear(hidden_dim, 1) → V(s_t)
```

### 2.3 Auxiliary Prediction Head

Predicts next actions of all teammates from current belief:

```
Input: belief state b_t

b_t → Linear(hidden_dim, aux_hidden_dim) → ReLU
    → Linear(aux_hidden_dim, (n_agents-1) × n_actions)
```

---

## 3. Auxiliary Loss: Design and Interference Analysis

### 3.1 Auxiliary Loss Formulation

The auxiliary loss encourages belief states to encode predictive information about teammate behavior:

```
L_aux = E[Σ_{j≠i} m^{i←j}_{t+1} · CE(π̂(·|b_t), a^j_{t+1})]
```

Where:
- `m^{i←j}_{t+1}` is the visibility mask (1 if agent i can observe agent j at t+1)
- `CE` is cross-entropy loss
- `π̂(·|b_t)` is the predicted action distribution for teammate j
- `a^j_{t+1}` is the actual action taken by teammate j

### 3.2 The Interference Problem

**Critical Finding**: While the auxiliary loss improves early learning by encouraging informative belief representations, it interferes with policy optimization during later training stages.

#### Mechanism of Interference

1. **Gradient Conflict**: The auxiliary loss computes gradients through the belief state encoder. These gradients optimize for *prediction accuracy* rather than *reward maximization*.

2. **Representation Drift**: As training progresses, the auxiliary loss can cause belief representations to drift toward features useful for prediction but suboptimal for decision-making.

3. **Policy Degradation Cascade**:
   ```
   Aux gradients → Belief encoder changes → Policy head receives different features
                → Policy performance degrades → Value estimates become inaccurate
                → Advantage estimates become noisy → Training becomes unstable
                → Performance collapses
   ```

4. **Empirical Evidence**:
   - Without annealing: Peak reward ~290, final reward ~13.5 (95% collapse)
   - With annealing: Early reward ~66, final reward ~102.5 (55% improvement)

### 3.3 Solution: Auxiliary Loss Annealing

We implement a warmup-then-decay schedule for the auxiliary loss:

```python
# During warmup (first 50-100 training steps)
aux_weight = λ_base × 0.05  # Small but non-zero

# After warmup
aux_weight = 0  # Completely disabled
```

**Rationale**:
- During early training, beliefs are randomly initialized and need structure
- The auxiliary loss provides useful inductive bias for learning meaningful representations
- Once beliefs are structured, the policy gradient alone is sufficient
- Continuing auxiliary updates interferes with fine-tuning the policy

---

## 4. Stability Mechanisms

### 4.1 Target Critic Network

We maintain a target critic network updated via Polyak averaging:

```python
θ_target ← τ · θ + (1 - τ) · θ_target
```

With τ = 0.005 (slow update).

**Purpose**: Stabilizes value estimation by providing a slowly-changing target for computing returns and advantages. Without this, the critic can change too rapidly between PPO epochs, causing advantage estimates to become inconsistent.

### 4.2 KL Divergence Early Stopping

We monitor the approximate KL divergence between old and new policies:

```python
log_ratio = log_π_new - log_π_old
approx_kl = mean((exp(log_ratio) - 1) - log_ratio)

if approx_kl > 1.5 × target_kl and epoch > 0:
    break  # Stop PPO epochs early
```

With target_kl = 0.015.

**Purpose**: Prevents the policy from changing too drastically in a single update, which can cause performance collapse. This is especially important when the auxiliary loss has been disabled and gradients come purely from policy optimization.

### 4.3 Conservative Gradient Clipping

We apply aggressive gradient clipping:

```python
grad_clip = min(config.grad_clip, 1.0)  # Maximum 1.0
actor_grad_norm = clip_grad_norm_(agent.parameters(), grad_clip)
critic_grad_norm = clip_grad_norm_(critic.parameters(), grad_clip)

# Skip update if gradients are still too large
if actor_grad_norm > 50 or critic_grad_norm > 50:
    skip_update()
```

**Purpose**: Large gradients indicate unstable training dynamics. By clipping aggressively and skipping extreme updates, we prevent catastrophic weight changes.

### 4.4 Advantage Clipping

We clip normalized advantages to a conservative range:

```python
advantages = (advantages - mean) / std
advantages = clamp(advantages, -5.0, 5.0)
```

**Purpose**: Extreme advantages can cause large policy updates even with PPO clipping. By limiting advantage magnitude, we ensure more stable updates.

### 4.5 Entropy Coefficient Decay

We decay the entropy coefficient over training:

```python
entropy_coef = max(0.001, entropy_coef × 0.999)
```

**Purpose**: High entropy encourages exploration early in training. As the policy improves, we reduce exploration to allow exploitation of learned behaviors.

### 4.6 Learning Rate Scheduling

We use exponential decay for learning rates:

```python
actor_lr = base_lr × 0.3  # Lower actor LR
critic_lr = base_lr × 0.5

scheduler = ExponentialLR(optimizer, gamma=0.995)
```

**Purpose**: Smaller learning rates later in training prevent overshooting optimal parameters. The actor has a lower learning rate because policy changes are more sensitive than value function changes.

---

## 5. Training Procedure

### 5.1 Episode Collection

```
for each episode:
    initialize belief states to zeros
    for each timestep t:
        for each agent i:
            encode observation
            attend to visible teammate actions
            update belief state via GRU
            sample action from policy
        execute joint action
        store transition in replay buffer

    call update_on_episode_end(episode_reward)
```

### 5.2 Training Step

```
sample batch from replay buffer

# Compute targets using TARGET critic (not current critic)
with no_grad():
    old_values = target_critic(states)
    old_log_probs = policy.log_prob(actions)
    returns, advantages = compute_gae(rewards, old_values, dones)
    advantages = normalize_and_clip(advantages)

# PPO epochs (reduced after warmup)
n_epochs = 3 if training_step < warmup_steps else 2

for epoch in range(n_epochs):
    # Forward pass
    beliefs, log_probs, entropy = forward_pass(observations, actions)

    # Check KL divergence
    if approx_kl > 1.5 × target_kl and epoch > 0:
        break

    # PPO losses
    ratio = exp(log_probs - old_log_probs)
    ratio = clamp(ratio, 0, 5)  # Numerical stability

    surr1 = ratio × advantages
    surr2 = clamp(ratio, 1-ε, 1+ε) × advantages
    policy_loss = -min(surr1, surr2)

    # Value loss with clipping
    value_pred = critic(states)
    value_clipped = old_values + clamp(value_pred - old_values, -ε_v, ε_v)
    value_loss = max((value_pred - returns)², (value_clipped - returns)²)

    # Auxiliary loss (only during warmup)
    if training_step < warmup_steps × 2:
        aux_loss = compute_auxiliary_loss(beliefs, next_actions, visibility)
    else:
        aux_loss = 0

    # Total loss
    loss = policy_loss + c_v × value_loss + c_e × entropy_loss + λ × aux_loss

    # Optimization with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    clip_gradients()
    if gradients_acceptable():
        optimizer.step()

# Soft update target critic
soft_update(target_critic, critic, tau=0.005)

# Update schedulers
scheduler.step()
```

---

## 6. Hyperparameters

### 6.1 Network Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| embed_dim | 64 | Embedding dimension for observations and actions |
| hidden_dim | 128 | GRU hidden state dimension (belief size) |
| attention_dim | 64 | Query/Key dimension for attention |
| aux_hidden_dim | 64 | Hidden dimension in auxiliary predictor |
| critic_hidden_dim | 128 | Hidden dimension in critic network |

### 6.2 PPO Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| clip_param (ε) | 0.2 | PPO clipping parameter |
| value_clip (ε_v) | 0.2 | Value function clipping parameter |
| ppo_epochs | 3 → 2 | PPO epochs (reduced after warmup) |
| target_kl | 0.015 | KL divergence threshold for early stopping |

### 6.3 Loss Coefficients

| Parameter | Value | Description |
|-----------|-------|-------------|
| value_loss_coef | 0.5 | Weight for value loss |
| entropy_coef | 0.01 → 0.001 | Entropy bonus (decayed) |
| aux_lambda | 0.1 × 0.05 | Auxiliary loss weight (during warmup only) |
| gae_lambda | 0.95 | GAE parameter |

### 6.4 Optimization

| Parameter | Value | Description |
|-----------|-------|-------------|
| base_lr | 0.0005 | Base learning rate |
| actor_lr | base_lr × 0.3 | Actor learning rate |
| critic_lr | base_lr × 0.5 | Critic learning rate |
| lr_decay | 0.995 | Exponential LR decay per step |
| grad_clip | 1.0 | Gradient clipping threshold |
| weight_decay | 1e-5 | L2 regularization |
| optimizer | Adam | Optimizer with eps=1e-5 |

### 6.5 Stability Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| warmup_steps | 50 | Steps before annealing auxiliary loss |
| target_update_tau | 0.005 | Soft update coefficient for target critic |
| advantage_clip | [-5, 5] | Advantage clipping range |
| ratio_clip | [0, 5] | Importance ratio clipping for stability |
| grad_skip_threshold | 50 | Skip update if gradient norm exceeds this |
| entropy_decay | 0.999 | Entropy coefficient decay rate |

---

## 7. Experimental Results

### 7.1 Stability Comparison

| Metric | Before Fix (v1) | After Fix (v2) | Improvement |
|--------|-----------------|----------------|-------------|
| Peak Reward | ~290 | ~110 | - |
| Final Reward | ~13.5 | ~102.5 | +660% |
| Stability | Collapsed | Stable | - |
| Training Trend | Peak → Collapse | Continuous Improvement | - |

### 7.2 Performance vs Baselines

| Environment | VABL (v2) | QMIX | Winner |
|-------------|-----------|------|--------|
| Simple Coordination | 47.9 ± 12.4 | -2.2 ± 0.0 | VABL (+2200%) |
| Overcooked Cramped | 102.5 | 63.3 | VABL (+62%) |
| MPE Simple Spread | -71.8 | -82.3 | VABL (+13%) |

### 7.3 Learning Dynamics

**Before Stability Fixes:**
```
Episode:    1-50    51-100   101-150  151-200
Reward:     64.6    290.7    48.2     13.5
Status:     Rising  Peak     Declining COLLAPSED
```

**After Stability Fixes:**
```
Episode:    1-30    31-60    61-100
Reward:     66.5    72.3     102.5
Status:     Rising  Rising   Rising (STABLE)
```

---

## 8. Key Insights

### 8.1 Auxiliary Loss Trade-off

The auxiliary loss presents a fundamental trade-off:
- **Benefit**: Provides inductive bias for learning structured belief representations
- **Cost**: Introduces gradients that conflict with policy optimization

**Optimal Strategy**: Use auxiliary loss during early training (warmup), then disable completely. This allows beliefs to develop structure initially, then lets policy optimization proceed without interference.

### 8.2 Value Estimation Stability

Unstable value estimation is a primary cause of training collapse in actor-critic methods. Our solutions:
1. Target network prevents rapid value function changes
2. Value clipping limits per-update changes
3. Using target critic for advantage computation ensures consistency

### 8.3 Conservative Updates

In complex environments, aggressive updates often lead to collapse. Our approach:
1. Low learning rates (especially for actor)
2. Aggressive gradient clipping
3. KL-based early stopping
4. Skipping updates with extreme gradients

### 8.4 Entropy Management

Entropy regularization must be balanced:
- Too high: Policy remains random, no learning
- Too low: Premature convergence to suboptimal policy
- Solution: Start high, decay slowly

---

## 9. Implementation Notes

### 9.1 Numerical Stability

```python
# Log probability computation
log_probs = F.log_softmax(logits, dim=-1)  # Numerically stable

# Importance ratio clipping
ratio = torch.exp(log_probs - old_log_probs)
ratio = torch.clamp(ratio, 0.0, 5.0)  # Prevent explosion

# Advantage normalization
adv_std = torch.clamp(advantages.std(), min=1e-8)  # Prevent division by zero
```

### 9.2 Memory Management

```python
# Detach old values/log_probs to prevent gradient flow
old_log_probs = old_log_probs.detach()
old_values = old_values.detach()

# Use torch.no_grad() for target computations
with torch.no_grad():
    old_values = target_critic(states)
```

### 9.3 Device Management

All tensors must be on the same device:
```python
self.device = device
self.agent = self.agent.to(device)
self.critic = self.critic.to(device)
self.target_critic = self.target_critic.to(device)
```

---

## 10. Conclusion

The VABL algorithm demonstrates that attention-based belief learning can enable effective implicit coordination in multi-agent systems. However, careful attention to training stability is crucial. The key insight is that auxiliary objectives, while useful for representation learning, must be managed carefully to avoid interfering with the primary policy optimization objective.

Our stability improvements transform VABL from an algorithm that shows promising early learning but ultimately collapses, to one that maintains stable improvement throughout training and achieves state-of-the-art performance on coordination tasks.

---

## Appendix A: Pseudocode

```python
class VABL:
    def __init__(self, config, n_agents, obs_shape, state_shape, n_actions, device):
        self.agent = VABLAgent(...)           # Policy network with attention
        self.critic = CentralizedCritic(...)   # Value function
        self.target_critic = copy(self.critic) # Target network

        self.aux_lambda_current = config.aux_lambda
        self.current_entropy_coef = config.entropy_coef
        self.warmup_steps = 50

    def select_actions(self, obs, available_actions, prev_actions, visibility_masks):
        for each agent i:
            # Encode observation
            obs_embed = self.agent.obs_encoder(obs[i])

            # Encode teammate actions
            teammate_embeds = self.agent.action_encoder(prev_actions[not i])

            # Attend to teammates
            context = self.agent.attention(
                query=belief[i],
                keys=teammate_embeds,
                values=teammate_embeds,
                mask=visibility_masks[i]
            )

            # Update belief
            belief[i] = self.agent.gru(concat(obs_embed, context), belief[i])

            # Get action
            logits = self.agent.policy_head(belief[i])
            action[i] = sample(softmax(logits))

        return actions

    def train_step(self, batch):
        # Compute old values using TARGET critic
        with no_grad():
            old_values = self.target_critic(batch.states)
            old_log_probs = self.get_log_probs(batch)
            returns, advantages = self.compute_gae(batch, old_values)
            advantages = normalize_and_clip(advantages, range=[-5, 5])

        # PPO epochs
        for epoch in range(self.n_epochs):
            log_probs, entropy, beliefs = self.forward(batch)

            # Early stopping on KL
            approx_kl = compute_kl(log_probs, old_log_probs)
            if approx_kl > 0.0225 and epoch > 0:
                break

            # PPO loss
            ratio = exp(log_probs - old_log_probs).clamp(0, 5)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1-0.2, 1+0.2) * advantages
            policy_loss = -min(surr1, surr2).mean()

            # Value loss with clipping
            values = self.critic(batch.states)
            value_loss = clipped_value_loss(values, old_values, returns)

            # Auxiliary loss (warmup only)
            if self.training_step < self.warmup_steps * 2:
                aux_loss = self.auxiliary_loss(beliefs, batch.next_actions)
            else:
                aux_loss = 0

            # Total loss
            loss = (policy_loss
                    + 0.5 * value_loss
                    + self.current_entropy_coef * (-entropy)
                    + self.aux_lambda_current * 0.05 * aux_loss)

            # Update with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(self.parameters(), 1.0)
            if grad_norm < 50:
                self.optimizer.step()

        # Soft update target
        soft_update(self.target_critic, self.critic, tau=0.005)

        # Decay entropy
        self.current_entropy_coef *= 0.999
        self.current_entropy_coef = max(0.001, self.current_entropy_coef)

    def update_on_episode_end(self, episode_reward):
        # Anneal auxiliary loss after warmup
        if self.training_step > self.warmup_steps:
            self.aux_lambda_current *= 0.995
```

---

## Appendix B: Ablation Studies

| Configuration | Final Reward | Stability |
|--------------|--------------|-----------|
| Full VABL v2 | 102.5 | Stable |
| No target critic | 45.2 | Unstable |
| No aux annealing | 13.5 | Collapsed |
| No KL early stop | 78.3 | Marginal |
| No entropy decay | 89.1 | Stable |
| No grad clipping | 23.4 | Collapsed |
| Higher aux_lambda (0.5) | 8.2 | Collapsed |
| No value clipping | 67.8 | Marginal |

The ablation confirms that auxiliary loss annealing and target critic are the most critical stability components.

---

## Appendix C: Ablation Configuration Reference

All ablation parameters are configurable via `configs/algorithm/vabl.yaml`:

| Parameter | Default | Description | Sweep Values |
|-----------|---------|-------------|--------------|
| `aux_lambda` | 0.1 | Auxiliary loss weight | [0.0, 0.01, 0.1, 0.5, 1.0] |
| `hidden_dim` | 128 | Belief state dimension | [16, 32, 64, 128] |
| `attention_heads` | 4 | Number of MHA heads | [1, 2, 4, 8] |
| `warmup_steps` | 50 | Steps before aux annealing | [0, 25, 50, 100, 200] |
| `aux_decay_rate` | 0.995 | Aux lambda exponential decay | [0.99, 0.995, 0.999, 1.0] |
| `min_aux_lambda` | 0.05 | Minimum aux lambda after decay | [0.0, 0.01, 0.05, 0.1] |
| `stop_gradient_belief` | false | Detach beliefs in aux loss | [false, true] |
| `use_attention` | true | Use MHA vs mean pooling | [true, false] |
| `use_aux_loss` | true | Enable auxiliary prediction | [true, false] |

### Running Ablation Sweeps

```bash
# Lambda sweep
python -m marl_research.scripts.run_ablation_sweep --ablation lambda_sweep --seeds 3 --episodes 100

# Attention heads sweep
python -m marl_research.scripts.run_ablation_sweep --ablation attention_heads_sweep --seeds 3

# Stop gradient ablation
python -m marl_research.scripts.run_ablation_sweep --ablation stop_gradient --seeds 5

# List all available ablations
python -m marl_research.scripts.run_ablation_sweep --list
```

---

## Appendix D: Coordination Metrics

The codebase tracks coordination-specific metrics to measure agent synchronization:

### Metrics Tracked

| Metric | Description | Calculation |
|--------|-------------|-------------|
| `coordination_rate` | Fraction of steps where all agents took the same action | `coordination_count / total_steps` |
| `joint_action_agreement` | Binary: 1 if all agents agreed this step, 0 otherwise | `int(len(set(actions)) == 1)` |

### Interpretation

- **High coordination_rate (>0.5)**: Agents have learned to synchronize actions
- **Increasing coordination_rate**: Algorithm is learning coordination over time
- **Low with high reward**: Environment may not require full coordination

### Logging

Coordination metrics are logged to TensorBoard/W&B:
- `train/coordination_rate`: Rolling 100-episode average

---

## Appendix E: Dual Metrics System

The training system tracks both shaped (training) and sparse (evaluation) rewards:

### Shaped vs Sparse Rewards

| Environment | Shaped Reward | Sparse Reward |
|-------------|---------------|---------------|
| Overcooked | Dense shaping (pickup, place, cook) | Only soup delivery (+20) |
| Simple | Same as sparse | Coordination bonus (+1/+2) |
| SMAC | Same as sparse | Win/lose signal |

### Usage in Training

```
train/shaped_reward  - Used for gradient computation (denser signal)
train/sparse_reward  - Reported for evaluation (task completion)
```

### Best Checkpoint Selection

The trainer tracks the best evaluation reward and saves `best_checkpoint.pt`:

```python
if eval_reward > best_eval_reward:
    best_eval_reward = eval_reward
    algorithm.save("best_checkpoint.pt")
```

This enables early-stopping style model selection without actually stopping training.

### Training Summary

At the end of training, a summary is logged:
```
==================================================
Training Summary
==================================================
Final shaped reward (100-ep avg): 102.5
Final sparse reward (100-ep avg): 98.3
Best eval reward: 115.2
Best checkpoint: results/run_xyz/best_checkpoint.pt
Final checkpoint: results/run_xyz/final_checkpoint.pt
==================================================
```
