# Exp-11B: ConvNeXt + (ViT + kNN-ArcFace) Ensemble

## What this is

Re-run Exp-11's winning ensemble pattern (Approach A — ViT+kNN first, then
blend with ConvNeXt) but with the upgraded **Exp-9B Tier 2 ViT+kNN+ArcFace**
output as the ViT branch.

## Why

Exp-11 final F1 = 0.4379 used Exp-9's frozen-DINOv2 ViT+kNN at F1=0.41754 as
its ViT input. Exp-9B improved that to F1=0.42267 (+0.00513) by swapping in a
LoRA-adapted DINOv2 + ArcFace head retriever and re-tuning beta_multi to 0.85.
If the +0.005 gain on the ViT branch translates through the ensemble blend,
Exp-11B should land around 0.443-0.446.

## Pipeline

```
ViT+kNN+ArcFace (Exp-9B output)              ConvNeXt-Large DINOv3
  per tile: softmax(logits/T)                  per tile: softmax(logits/T)
  + kNN vote from ArcFace FAISS index          + max_prob across tiles
  blend at beta_multi=0.85, b_single=1.00      + temporal fusion at alpha=0.30
  max_prob across tiles
  temporal fusion at alpha=0.15
                          \                   /
                           +--- w * P_vit + (1-w) * P_cnxt ---+
                                          |
                                  threshold + top-10 CSV
```

The ViT branch is the **Exp-9B fused parquet** at
`submissions/exp9b_knn_diff_beta_phase2a/knn_bM0.85_bS1.00_T1.50_fused_a0.15.parquet`.

The ConvNeXt branch is **reused unchanged** from Exp-11 at
`parquet/exp11_ensemble/convnext_large/scale_{4x4,5x5,6x6}.parquet`.

## What's reused

| Component | Source | Status |
|---|---|---|
| ConvNeXt-Large DINOv3 backbone | Exp-10 fine-tune | Already trained, no change |
| ConvNeXt test tile parquets (4x4, 5x5, 6x6) | Exp-11 extraction | Already extracted, no change |
| Ensemble blending script | `Exp-11/ensemble_with_knn.py` | Reused as-is |
| ConvNeXt aggregation params (T=1.5, alpha=0.30) | Exp-11 best | Reused as-is |
| ViT+kNN fused parquet | Exp-9B Tier 2 output | **New** — replaces Exp-9's |

## What's swept

Exp-11 found w=0.77 optimal with the F1=0.41754 ViT branch. Now the ViT branch
is at F1=0.42267 — stronger by +0.005. Stronger branch usually wants slightly
more weight, so optimum w should shift up toward 0.78-0.81.

Initial sweep: **w in {0.74, 0.76, 0.77, 0.78, 0.80, 0.82}** at fixed
cnxt_alpha=0.30, threshold=0.100.

If the optimum shifts noticeably, do a fine-grained sweep around the new peak.

## Submission budget

6 leaderboard submissions for the initial weight sweep. Budget for ~4 more
fine-grain points if the peak isn't at one of the sampled w values. Total
~10 submissions for this experiment.

## Success criteria

- **Primary**: best F1 >= Exp-11 baseline (0.4379) + 0.003 = 0.4409.
  Anything that beats Exp-11 is a real win since it confirms the +0.005
  retriever lift composes through the ensemble.
- **Secondary**: best F1 >= 0.4429 (Exp-11 + full +0.005 lift). Less likely
  due to ensemble math compression but possible if w shift is favorable.

## What this experiment does NOT change

- ConvNeXt backbone, weights, training data
- ConvNeXt scale scheme (4+5+6 — confirmed best in Exp-11)
- ConvNeXt temporal fusion alpha (0.30 — confirmed best in Exp-11)
- Threshold (0.100 — confirmed best in Exp-11)
- Top-K (10)
- Geo mask (sw_europe)
- ViT+kNN configuration (already locked in by Exp-9B)

The only free variable is the ensemble weight w.

## Files

```
Exp-11B_KNN_ArcFace_Conv_Ensemble/
└── docs/
    └── Exp11B_Plan.md                          (this file)
└── slurm_ensemble_arcface_knn.sbatch           (single job, sweeps 6 weights)
```

No new Python — reuses Exp-11/ensemble_with_knn.py.
