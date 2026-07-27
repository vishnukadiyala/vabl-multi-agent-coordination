# VABL Research Wiki — Index

LLM-maintained knowledge base for the VABL research project.
Read this first to navigate. See [log.md](log.md) for activity history.

Last updated: 2026-05-07

> **Status — 2026-05-06: Paper submitted (or imminently submitting).**
>
> **Title:** *When Auxiliary Losses Fail: Non-Stationary Targets Induce
> Directional Gradient Noise.* (Final title; successor to three earlier
> working titles: "Structured Non-Stationary Auxiliary Targets Induce
> Directional Gradient Noise Near Convergence", "When Auxiliary Losses
> Harm Belief Learning", and "Constant Auxiliary Losses Considered
> Harmful." The final framing is a learning-dynamics principle:
> *structured non-stationary aux targets inject directional gradient
> noise that dominates the parameter-variance contribution of vanishing
> policy gradients near convergence* in cooperative MARL.)
>
> NeurIPS 2026 submission. Main body locked at the 9-page limit; appendix
> trimmed during the final re-read by removing the Sample-Efficiency
> Curves appendix (Appendix~H + Figure~7 + Table~16) because its
> "pathology is transient" reading contradicted the Phase~2 d=+1.40
> permanence finding at 10M and was not load-bearing for the main
> argument. Total 32 pages (was 33 before App~H removal). Compile clean,
> 0 undefined refs. Run-count updated to 290+ (from 375+) to reflect
> the dropped appendix. Anonymous code archive built
> (`ICML_anonymous/when-aux-fails.zip`, 4 MB, 249 files including all
> 161 per-seed result JSONs).
>
> Key findings supporting the reframe:
> - Target-source distinguishing test (frozen / random / teammate) on AA
>   confirms the non-stationarity axis is the driver, not aux capacity.
> - Aux-capacity scaling (8x sweep) null — falsifies capacity consumption.
> - Gradient diagnostics at N=5 show std(cos) = 0.185, R^2 = 0.94 for the
>   cosine-temporal-std -> Final50 cross-seed std proxy.
> - CIFAR-100 5-seed stationarity null: d = -0.16 on Best, CI crosses 0.
> - Five new theory-parallel citations (COALA-PG, DG-PG, ROCKET, GAC,
>   Trust-Region Decomposition) integrated; DG-PG Assumption 3.1
>   exogeneity violation gives the formal "why" for the pathology.
> - Drift-gated aux controller (future work) now citable to DG-PG
>   Theorem 4.2 as the closed-form optimum it approximates.
>
> See [NeurIPS pivot page](papers/vabl_neurips_pivot.md) for full status
> table; [theory parallels page](concepts/theory_parallels.md) for the
> five-paper mapping; [cifar_5seed](experiments/cifar_5seed.md) for the
> supervised null.

---

## Sources (Ingested)

- [VABL Paper](sources/vabl_paper.md) — Main ICML submission: claims, evidence, theoretical status
- [Rebuttal Round 1](sources/rebuttal_r1.md) — 4 reviewers, scores, what worked/didn't
- [Experiment Results](sources/experiment_results_rebuttal.md) — 5 JSON files, cross-cutting findings

**Raw manifest:** [`raw/manifest.md`](../raw/manifest.md) — full catalog of all source files (84 JSONs, 57 figures, papers, docs)

## Algorithms

- [VABL](algorithms/vabl.md) — *(in-progress 2026-04-09)* Our method. Canonical implementation is now `marl_research/algorithms/jax/vabl_v2.py` with three fix-path knobs for the camera-ready empirical investigation.
- [MAPPO](algorithms/mappo.md) — Primary baseline: multi-agent PPO with centralized critic
- [QMIX](algorithms/qmix.md) — Value-based baseline: monotonic value factorization
- [TarMAC](algorithms/tarmac.md) — Communication baseline: targeted multi-agent communication
- [AERIAL](algorithms/aerial.md) — Attention baseline: hidden state sharing (Phan et al., ICML 2023)
- [CommNet](algorithms/commnet.md) — Simple communication baseline: broadcast averaging
- [ConfigurableAgent](algorithms/configurable_agent.md) — *(NEW 2026-04-11)* Swappable recurrence + attention for architecture sweep

## Environments

- [Overcooked](environments/overcooked.md) — Primary eval: 2-agent cooking (AA, Cramped Room, Ego-PO)
- [SMAC / SMAC v2](environments/smac.md) — StarCraft micromanagement (planned for PRAJNA)
- [Simple Coordination](environments/simple.md) — Fast test: N-agent with stochastic visibility

## Concepts

- [Policy Collapse](concepts/policy_collapse.md) — *(STALE 2026-04-08)* Central phenomenon: agents lose learned coordination — disappears at 10M
- [Belief Learning](concepts/belief_learning.md) — Core contribution: latent beliefs via auxiliary prediction
- [Attention Mechanisms](concepts/attention_mechanisms.md) — MHA for teammate action aggregation
- [Credit Assignment](concepts/credit_assignment.md) — How agents attribute team reward
- [Aux Loss Bug (Apr 2026)](concepts/aux_loss_bug.md) — *(NEW)* JAX silently disabled aux loss; fix revealed Full VABL is worst of its own ablations at 10M
- [Training Budget Artifact](concepts/training_budget_artifact.md) — *(NEW)* "Collapse" is largely an artifact of stopping training too early; methodological rule for the camera-ready
- [GPU Utilization Bottleneck](concepts/gpu_utilization_bottleneck.md) — *(RESOLVED 2026-04-11)* Was 15% GPU util; scan rollout fix now live (3x speedup, 101 ep/s)
- [Theory Parallels (5 new papers)](concepts/theory_parallels.md) — *(NEW 2026-04-22)* Maps I_dir, J_pi, Sigma_pi, Prop 1 onto COALA-PG, DG-PG, ROCKET, GAC, LN-DQN, Trust-Region Decomposition. DG-PG's Assumption 3.1 is what our setting violates.

## Experiments

- [Phase 2 Minimal Diagnostic](experiments/phase2_minimal.md) — *(COMPLETE)* 7-config ablation + fix paths × 5 seeds at 10M on AA
- [Mechanism Identification (ExpA + ExpD)](experiments/mechanism_identification.md) — *(NEW 2026-04-22)* Target-source distinguishing test + 8x aux-capacity sweep; falsifies capacity consumption, identifies structured non-stationarity as driver
- [Gradient Diagnostics (incl. ExpB)](experiments/gradient_diagnostics.md) — *(UPDATED 2026-04-22)* N=5 gradient decomposition across 4 conditions; cosine-std proxy R^2 = 0.94 for Final50 variance ordering
- [Hanabi (ExpY)](experiments/hanabi.md) — *(NEW 2026-04-22)* Turn-based partial-info cross-env; pathology reproduces with d=+2.81, attention no longer required (both aux-ON cells pathological)
- [VAE Belief (ExpX-A)](experiments/vae_belief.md) — *(DROPPED FROM PAPER 2026-04-23)* Dynamic-Belief-style encoder replication; underpowered at n=5 (CI crosses zero); appendix removed; raw data preserved as historical record
- [MPE Ablation](experiments/mpe_ablation.md) — *(COMPLETE 2026-04-10)* 4-config ablation on MPE simple_spread; pathology absent (environment-dependent threshold)
- [CIFAR-100 5-seed](experiments/cifar_5seed.md) — *(NEW 2026-04-22)* 4-config × 5 seeds supervised null; stationarity prediction confirmed (d = −0.16 on Best, CI crosses 0)
- [Sample-Efficiency Curves](experiments/sample_efficiency.md) — *(DROPPED FROM PAPER 2026-05-06)* 4 configs × 4 budgets × 5 seeds; transient-collapse reading (46% → 3% with budget) contradicted Phase 2's d=+1.40 permanence at 10M; Appendix H + Figure 7 + Table 16 removed; raw data preserved as historical record
- [Architecture Sweep](experiments/architecture_sweep.md) — *(COMPLETE)* 60-run component isolation: pathology generalizes across recurrence/attention, absent with mean pool, reversed on critic-aux
- [Rebuttal Runs](experiments/rebuttal_runs.md) — *(STALE)* Superseded by Phase 2
- [Ablation Studies](experiments/ablations.md) — Component-wise analysis of VABL
- [10M Scaling](experiments/10m_scaling.md) — *(STALE)* Superseded by Phase 2

## Papers

- [VABL — ICML 2026](papers/vabl_icml2026.md) — *(REJECTED 2026-04-09)* Original submission, claims now superseded by Phase 2 data
- [NeurIPS 2026 submission](papers/vabl_neurips_pivot.md) — *(SUBMITTED 2026-05-06)* "When Auxiliary Losses Fail: Non-Stationary Targets Induce Directional Gradient Noise" — 32 pages (9 main + appendix + checklist; was 33 before App H removal), 290+ runs (was 375+), anonymous code archive built (when-aux-fails.zip, 4 MB)
- [Citation audit](papers/citation_audit.md) — *(NEW 2026-04-13)* 10 missing canonical citations patched (PPO, GRU, MHA, etc.) + SMAX/SMAC misattribution fix
- [VABL framing concern](papers/vabl_framing_concern.md) — *(NEW 2026-04-13, UNRESOLVED)* "You built it and broke it" reviewer risk; pivot to AERIAL as published face?
- [Self-review feedback](papers/self_review_2026-04-13.md) — *(NEW 2026-04-13, UNRESOLVED)* 8 concrete concerns including hyperparameter sensitivity gap + Story A/B/C tangle; Batch 0/1/2 plan
- [PRAJNA](papers/prajna.md) — Next paper: predictive belief-space imagination

## Reviews

- [ICML Rebuttal](reviews/icml_rebuttal.md) — 4 reviewers, concerns, responses, evidence mapping
- [Rebuttal Round 2 Draft](reviews/rebuttal_round2_draft.md) — Draft response with placeholders for pending results
- [NeurIPS 2026 Rebuttal Plan](reviews/neurips2026_rebuttal_plan.md) — *(NEW 2026-07-24)* Scores 4/4/3 + AC meta leaning reject; per-reviewer strategy, cross-cutting asks, compute triage. Raw reviews: `raw/reviews/neurips2026_reviews.md`

---

*36 pages | 3 sources ingested | Last lint: never*

*Added 2026-04-08/09: aux_loss_bug, training_budget_artifact, phase2_minimal, vabl_neurips_pivot, gpu_utilization_bottleneck*
*Added 2026-04-10: mpe_ablation, sample_efficiency, gradient_diagnostics*
*Added 2026-04-11: configurable_agent, architecture_sweep; updated gpu_utilization_bottleneck (resolved)*
*Updated 2026-04-12: architecture_sweep COMPLETE, gradient_diagnostics N=5 COMPLETE, SMAX fix paths COMPLETE, paper submission-ready*
*Added 2026-04-22: theory_parallels (5 new theory papers), cifar_5seed (CIFAR 4x5 supervised null), mechanism_identification (ExpA+ExpD), hanabi (ExpY), vae_belief (ExpX-A)*
*Updated 2026-04-30: Atiq feedback applied (Conclusion rewrite, Prop 1 rename, R^2 softening, internal figure titles stripped); statistical-rigor wording softened ("statistically indistinguishable" -> "not resolved at n=5", "ruling out" -> "inconsistent with")*
*Updated 2026-05-04/05: PAT review polish round (~12 of 16 items applied: pathology-permanence reconciliation, critic-side reversal language, Cohen's d sign convention, std unification to population, isotropic-tightness fix, PCGrad rewording, CIFAR Final5 switch, hyperparameters expansion); two new figure scripts (plot_f7_sample_efficiency.py, plot_f8_synthetic.py)*
*Updated 2026-05-04: anonymous code archive built (ICML_anonymous on when-aux-fails branch) with 161 per-seed result JSONs; ZIP supplementary path adopted*
*Updated 2026-05-06: paper at submission state, title locked to "When Auxiliary Losses Fail"*
*Updated 2026-05-06: end-to-end re-read pass; dropped Appendix H (Sample-Efficiency Curves) + Figure 7 + Table 16 because the "transient pathology" reading contradicted the Phase 2 d=+1.40 permanence at 10M; fixed §4 title (Teammate -> Target-Policy Drift), §8 Limitations (critic-side "improves stability" -> "qualitatively different regime"), Table 14 std (sample -> population on Mean pool row), checklist run-count (375+ -> 290+, GPU-hours 400 -> 320). Final compile: 32 pages, 0 undefined refs.*
