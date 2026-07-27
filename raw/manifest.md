# Raw Sources Manifest

Immutable source documents. The LLM reads from these but never modifies them.
This manifest is auto-generated — update when new sources are added.

Last updated: 2026-04-07

## Papers

| Source | Path | Date | Notes |
|--------|------|------|-------|
| Main paper (LaTeX) | `paper/example_paper.tex` | Jan 2026 | Primary submission source |
| Revised paper | `paper/revised_paper.tex` | Mar 2026 | Post-rebuttal revision |
| Bibliography | `paper/example_paper.bib` | Jan 2026 | All citations |
| Submitted PDF | `icml2026___Implicit_Coordination_*.pdf` | Jan 2026 | As-submitted version |

## Rebuttal Documents

| Source | Path | Date | Notes |
|--------|------|------|-------|
| Rebuttal draft | `rebuttal_draft.md` | Mar 30 2026 | Working draft |
| Rebuttal final | `rebuttal_final.md` | Mar 31 2026 | Submitted version |
| OpenReview format | `rebuttal_openreview.md` | Mar 31 2026 | Formatted for OpenReview |

## Research Notes

| Source | Path | Date | Notes |
|--------|------|------|-------|
| Paper claims update | `PAPER_CLAIMS_UPDATE.md` | Mar 2026 | Tracks claim revisions |
| Findings | `findings.md` | Feb 2026 | Key experimental findings |
| Implementation details | `paper_implementation_details.md` | Feb 2026 | Reproducibility details |
| Supplementary material | `paper_supplementary_material.md` | Feb 2026 | Appendix content |
| Paper update log | `paper_update.md` | Mar 2026 | Revision history |
| Submission improvements | `submission_improvements.md` | Jan 2026 | Pre-submission changes |
| Implementation plan | `docs/IMPLEMENTATION_PLAN.md` | Jan 2026 | Original experiment plan |

## Experiment Results (Key Files)

| Source | Path | Date | Experiment |
|--------|------|------|------------|
| 5-agent comparison | `results/5agent_comparison_5agents.json` | Mar 26 | VABL vs MAPPO, 5 agents |
| AA ablation (strong) | `results/ablation_strong_overcooked.json` | Mar 26 | 5-seed VABL ablation on AA |
| CR ablation (strong) | `results/ablation_strong_cramped_room.json` | Mar 28 | 5-seed VABL ablation on CR |
| AERIAL baseline | `results/baseline_comparison_overcooked_asymmetric_advantages.json` | Apr 3 | AERIAL 500ep on AA |
| Ego-PO comparison | `results/baseline_comparison_overcooked_ego_asymmetric_advantages.json` | Apr 3 | Ego-PO VABL+MAPPO |
| 8-agent results | `results/large_n_8agent.json` | Apr 6 | 8-agent Simple (in progress) |
| 2M AA ablation | `results/ablation_2m_aa.json` | Apr 6 | 2M-step AA ablation (in progress) |

**On Celestia:**
| Source | Path | Date | Experiment |
|--------|------|------|------------|
| TarMAC 5000ep | `results/baseline_comparison_overcooked_asymmetric_advantages.json` | Mar 2026 | TarMAC on AA |
| MAPPO 10M | `results/mappo_10m_persistent.json` | Apr 2026 | 25k ep, seeds 0-2 |
| VABL 10M | `results/vabl_10m_persistent.json` | Apr 2026 | 25k ep, seed 0 partial |

## Figures (Key)

| Source | Path | Description |
|--------|------|-------------|
| AA comparison (updated) | `figures/comparison_overcooked_asymmetric_advantages_updated.png` | Main comparison figure |
| CR comparison (updated) | `figures/comparison_overcooked_cramped_room_updated.png` | Cramped Room comparison |
| Simple comparison (updated) | `figures/comparison_simple_updated.png` | Simple env comparison |
| Collapse analysis | `figures/collapse_analysis_overcooked.png` | Collapse visualization |
| CR ablation | `figures/ablation_cramped_room_strong.png` | 5-seed ablation figure |
| AA ablation | `figures/ablation_overcooked_strong.png` | 5-seed ablation figure |
| Attention weights (OC) | `figures/attention_weights_overcooked.png` | Attention visualization |
| Attention weights (Simple) | `figures/attention_weights_simple.png` | Attention visualization |
| Attention entropy | `figures/attention_entropy_comparison.png` | Entropy analysis |
| Aux accuracy | `figures/aux_accuracy.png` | Auxiliary prediction accuracy |
| Lambda sensitivity | `figures/lambda_sensitivity.png` | Hyperparameter sensitivity |
| Sample efficiency | `figures/sample_efficiency.png` | Sample efficiency plot |
| Variance comparison | `figures/variance_comparison.png` | Variance across methods |

## Poster

| Source | Path | Notes |
|--------|------|-------|
| Final poster | `poster/VABL_Poster_FINAL_v3.pptx` | Latest version |
| Architecture diagram | `poster/fig_architecture_v2.png` | VABL architecture |
| Discovery vs maintenance | `poster/fig_discovery_vs_maintenance.png` | Key framing figure |
| Video script | `poster/video_script.md` | Presentation script |

## 84 total result JSONs, 27 main figures, 20 paper figures, 10 poster figures
## See results/ and figures/ directories for the full listing

## Reviews (added 2026-07-24)

| Source | Path | Notes |
|--------|------|-------|
| NeurIPS 2026 reviews | `raw/reviews/neurips2026_reviews.md` | Verbatim: AC y2yo meta + fXvf(4) PYCT(4) yGKw(3) |
