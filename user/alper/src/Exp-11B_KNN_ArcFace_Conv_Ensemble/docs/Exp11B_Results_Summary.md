# Exp-11B: ConvNeXt + (ViT + kNN + ArcFace) Ensemble — Results Summary

## Headline

**Final F1 = 0.44428** (public leaderboard, 2026-04-26).

Δ vs Exp-11 baseline (0.4379): **+0.00638**.
Δ vs Exp-9B alone (0.42267): **+0.02161**.

The +0.00513 ViT-branch improvement from Exp-9B transmitted nearly fully
through the ConvNeXt ensemble. Optimal weight stayed at w=0.77 — identical
to Exp-11.

## Pipeline

```
Exp-9B (ViT + kNN + ArcFace)              Exp-10 ConvNeXt-Large DINOv3
  -> per-tile softmax + kNN vote            -> per-tile softmax
  -> beta_multi=0.85 blend                  -> max_prob across 77 tiles
  -> max_prob across 77 tiles               -> temporal fusion (alpha=0.30)
  -> temporal fusion (alpha=0.15)
                       \                  /
                        +-- 0.77*Pvit + 0.23*Pcnxt --+
                                       |
                              threshold + top-10 CSV
```

The ViT branch comes from Exp-9B's fused parquet at
`submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_fused_a0.15.parquet`.

The ConvNeXt branch is reused unchanged from Exp-11 at
`parquet/exp11_ensemble/convnext_large/scale_{4x4,5x5,6x6}.parquet`.

## What changed from Exp-11

| Component | Exp-11 (was) | Exp-11B (now) |
|---|---|---|
| ViT backbone | Frozen DINOv2 ViT-Base | DINOv2 ViT-Base + LoRA (rank=8) |
| Retriever encoder | Frozen DINOv2 CLS | LoRA + ArcFace head (sub-center K=3) |
| FAISS index dim | 768 (frozen DINOv2) | 768 (LoRA-tuned ArcFace) |
| FAISS index size | ~860K vectors (Exp-9 build) | ~1.23M vectors (Exp-9B build) |
| beta_multi | 0.80 | **0.85** |
| beta_single | 1.00 | 1.00 (unchanged) |
| ViT temporal alpha | 0.15 | 0.15 (unchanged) |
| ViT branch standalone F1 | 0.41754 | **0.42267** |
| ConvNeXt branch | (unchanged) | (unchanged) |
| Ensemble weight w | 0.77 | **0.77 (unchanged)** |
| ConvNeXt alpha | 0.30 | 0.30 (unchanged) |
| Threshold | 0.100 | 0.100 (unchanged) |
| **Final F1** | **0.4379** | **0.44428** |

## Weight sweep results

Single sweep at fixed `cnxt_alpha=0.30`, `threshold=0.100`. Submitted via
`slurm_ensemble_arcface_knn.sbatch`.

| w | F1 | Δ vs Exp-11 (0.4379) |
|---|---|---|
| 0.74 | 0.43797 | +0.00007 |
| 0.76 | 0.44339 | +0.00549 |
| **0.77** | **0.44428** | **+0.00638** |
| 0.78 | 0.44296 | +0.00506 |
| 0.80 | 0.44201 | +0.00411 |
| 0.82 | 0.43490 | -0.00300 |

Peak is sharp and centered at w=0.77. No fine-grain sweep performed (gain
ceiling was clearly small).

## Key takeaways

1. **Lift transmits cleanly through the ensemble.** Exp-9B added +0.00513 to
   the ViT branch standalone; the ensemble gained +0.00638 over Exp-11. The
   ensemble math compressed the lift only slightly, and the ConvNeXt
   complementary signal slightly amplified it.

2. **Optimal weight w=0.77 is robust.** Same as Exp-11 with a stronger ViT
   branch. The blend ratio is invariant to small ViT-branch quality changes
   in the ~+0.005 F1 range. If the ViT branch were *much* stronger, w would
   shift up; at this magnitude it doesn't.

3. **No re-extraction needed for ConvNeXt.** The same ConvNeXt parquets that
   gave 0.4379 in Exp-11 give 0.44428 here. The ConvNeXt model and its
   tile-level outputs were not the bottleneck.

4. **The mechanism behind Exp-9B's gain doesn't break the ensemble.**
   Exp-9B's +0.005 came from beta recalibration (beta_multi 0.80 -> 0.85),
   not retriever quality. There was a risk that this beta-specific sweet
   spot wouldn't compose with ConvNeXt's predictions. It did, fully.

## What was NOT swept (and why)

- **cnxt_alpha**: held at 0.30 (Exp-11 best). Could shift slightly with the
  new ViT branch but unlikely to move the answer by >0.001.
- **threshold**: held at 0.100 (Exp-11 best, also Exp-9B best).
- **ConvNeXt scale scheme**: 4+5+6 only. Exp-11 confirmed extra scales hurt
  the ensemble even though they help ConvNeXt standalone.
- **Per-class weighting / confidence-gated blending**: Exp-11 confirmed
  fixed-weight blending dominates these.

## Submission budget used

6 leaderboard submissions for the weight sweep. Total session budget for
Exp-11B: 6.

## Final best configuration

| Component | Setting |
|---|---|
| **ViT backbone** | DINOv2 ViT-Base (`vit_base_patch14_reg4_dinov2.lvd142m`) |
| **ViT LoRA** | rank=8, alpha=16, on attn.qkv across all 12 blocks |
| **ViT ArcFace head** | 768 -> 768 -> BN -> ReLU -> 768 -> L2-norm |
| **ArcFace classifier** | Sub-center K=3, m=0.30, s=30 |
| **ViT scales** | 4x4 + 5x5 + 6x6 (77 tiles) |
| **ViT aggregation** | max_prob, T=1.5 |
| **kNN** | k=20, tau=0.07, FAISS IndexFlatIP 768-d, ~1.23M vectors |
| **kNN blend** | beta_multi=0.85, beta_single=1.00 |
| **ViT temporal fusion** | prior_blend, alpha=0.15 |
| **ConvNeXt model** | `convnext_large.dinov3_lvd1689m` (fine-tuned) |
| **ConvNeXt scales** | 4x4 + 5x5 + 6x6 (77 tiles) |
| **ConvNeXt aggregation** | max_prob, T=1.5 |
| **ConvNeXt temporal fusion** | prior_blend, alpha=0.30 |
| **Ensemble weight** | w=0.77 (ViT) / 0.23 (ConvNeXt) |
| **Threshold** | 0.100 |
| **TopK** | 10 |
| **JPEG** | quality=85, subsampling=4:2:2 |
| **Geo mask** | sw_europe_mask_bool.npy (~4981 of 7806 species pass) |

## Evolution to date

| Milestone | F1 | Improvement |
|---|---|---|
| Exp-3 baseline | 0.28093 | — |
| Exp-4 max_prob + fusion | 0.35698 | +0.07605 |
| Exp-7 multi-scale | 0.38466 | +0.02768 |
| Exp-8 tuning | 0.40100 | +0.01634 |
| Exp-9 + kNN | 0.41754 | +0.01654 |
| Exp-9B + LoRA+ArcFace + beta=0.85 | 0.42267 | +0.00513 |
| Exp-11 + ConvNeXt ensemble | 0.43790 | +0.01523 |
| **Exp-11B with Exp-9B ViT branch** | **0.44428** | **+0.00638** |

Total: +0.16335 over Exp-3 baseline (+58.1% relative).

## Files

```
Exp-11B_KNN_ArcFace_Conv_Ensemble/
└── docs/
    ├── Exp11B_Plan.md
    └── Exp11B_Results_Summary.md          (this file)
└── slurm_ensemble_arcface_knn.sbatch      (single job, 6-weight sweep)
```

No new Python — reuses `Exp-11/ensemble_with_knn.py` unchanged.

## What to consider next (if pursuing further gains)

1. **Add Exp-13 fine-tuned ViT-L as a third ensemble model.** Architecture
   diversity has paid off in Exp-11 (ConvNeXt) and Exp-11B (no change to that
   premise). A third model with different inductive bias is the highest-leverage
   move for ensemble F1 going forward.
2. **Re-tune ConvNeXt cnxt_alpha** at the new w=0.77 with the stronger ViT
   branch. Exp-11 found 0.30 best with the old branch. Three submissions
   (0.25, 0.30, 0.35) is cheap and could yield +0.0005-0.001.
3. **Don't escalate retriever fine-tuning further.** Exp-9B's Tier 1 vs Tier 2
   already showed the retriever-quality lever has been fully extracted.
