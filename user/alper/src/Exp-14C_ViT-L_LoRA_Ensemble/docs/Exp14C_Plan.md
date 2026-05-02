# Exp-14C — ViT-L LoRA-ArcFace + ConvNeXt ensemble (no Exp-16 mechanisms)

## Goal
Replace the Exp-11B production ViT-B+kNN branch with the new ViT-L (phase2v3)
classifier + LoRA+ArcFace 1024-d kNN branch in the Exp-14 ensemble. First
integration intentionally **excludes** Exp-16 post-hoc mechanisms (habitat-fit,
N3_r0.5 admission, biogeo mask) to isolate the new-branch contribution.

## Baselines to beat
| Reference | F1 | Notes |
|---|---|---|
| Exp-14 raw best (pre-Exp-16) | ~0.45611 | Per-source sim_w + alpine biogeo, no habitat-fit |
| Exp-11B production ensemble | 0.44428 | ViT-B+LoRA+kNN + ConvNeXt, w=0.77, beta=0.85 |
| **Current locked best** | **0.49842** | All Exp-16 mechanisms on top |

For the no-Exp-16 stage (this experiment): **target = beat 0.45611** to demonstrate
the new ViT-L branch is a real improvement before stacking Exp-16 back on top.

## Why parameter re-tuning is required
The new branch has a different probability shape than the ViT-B branch it replaces:
- LoRA+ArcFace retriever: inter-cosine = −0.006 (vs ViT-B's +0.011) → much
  sharper class separation
- ViT-L classifier (phase2v3): +0.011 F1 standalone over phase2v2 at 4×4
- Optimal classifier threshold shifted from 0.100 (ViT-B) → 0.075 (ViT-L K)

All previously-tuned hyperparameters were tuned for ViT-B. Likely shifts:

| Parameter | ViT-B optimum | Sweep range for ViT-L |
|---|---|---|
| `beta_multi` (kNN weight, multi-visit) | 0.85 | {0.65, 0.75, 0.85, 0.95} |
| `beta_single` (kNN weight, single-visit) | 1.00 | hold at 1.00 in stage 1; stage 2 try 0.90, 0.95 if winning |
| `w` (ViT vs ConvNeXt blend) | 0.77 | {0.70, 0.75, 0.80, 0.85} |
| `threshold` | 0.100 | {0.075, 0.090, 0.100} |
| `alpha` (ViT temporal fusion) | 0.15 | hold at 0.15 in stage 1; revisit if temporal mechanism dominates |
| `kNN k`, `tau` | 20, 0.07 | hold |

## Pipeline overview (one-shot per beta)

```
ViT-L K test parquets (4,5,6,7) ─┐
                                  │
1024-d FAISS index ──────────────►│  make_vitL_prefusion.py
                                  │  - per tile: classifier softmax + LoRA-kNN
                                  │  - blend with beta_multi (or beta_single)
                                  │  - aggregate per image (max_prob, T=1.5, image_topk=64)
                                  └──► vit_prefusion_b{beta}.parquet
                                                       │
                                                       ▼
ConvNeXt K parquets (4,5,6,7) ────► ensemble_v2.py (Exp-14)
                                    - re-fuses ViT branch via controller
                                    - re-fuses CNXT branch via controller
                                    - blend with weight w
                                    - emit CSV per (w, threshold)
```

## Sweep budget (LB-conscious)

LB has a **100 submissions/day** cap. Plan splits into two stages.

### Stage 1: coarse sweep (~24 CSVs)
- `beta_multi ∈ {0.75, 0.85, 0.95}` → 3 prefusion parquets
- `w ∈ {0.75, 0.80, 0.85}` × `threshold ∈ {0.075, 0.090}` → 6 combinations per beta

Expected outcome: identify the winning beta band and approximate w / threshold optima.

### Stage 2: fine sweep (~30 CSVs around winning region)
- Narrow beta around stage-1 winner ±0.05 → 3 betas
- Narrow w around stage-1 winner ±0.025 → 5 weights
- 2 thresholds at ±0.005 around stage-1 winner

Expected total: ~50 LB submissions across 2 days.

## Decision rules

After stage 1:
- **Best CSV ≥ 0.456 (beats Exp-14 raw)** → proceed to stage 2 (fine sweep)
- **Best CSV in [0.444, 0.456)** → marginal. Run stage 2 anyway but lower
  expectations for stacking Exp-16 on top
- **Best CSV < 0.444** → regression. Do NOT proceed to stage 2 with same
  approach. Diagnose: re-check kNN integration, classifier-from-LoRA-vs-phase2v3
  comparison, or whether Mixup degraded the classifier (see "comparison gate"
  in Exp-11C earlier discussion)

After stage 2:
- Lock the best (beta_multi, w, threshold) tuple
- Move to Exp-14D / Exp-16 integration: stack habitat-fit / N3_r0.5 / biogeo on
  top of the new ensemble, re-tune those mechanisms' params if needed

## What's NOT in this experiment
- No habitat-fit / habitat-predicted (Exp-16)
- No N-floor admission (Exp-16)
- No biogeo mask (Exp-16)
- No genus-dedup, no veg-fraction filter, no photometric TTA
- No Exp-14 controller features beyond baseline (lambda=0, default_alpha=0.15)
  — those get layered back in Exp-14D after the new branch is validated

## Hard prerequisites (verified 2026-05-02)
- ✓ phase2v3 K test parquets at scales 4/5/6/7
- ✓ ViT-L LoRA-ArcFace test parquets (with `teacher_logits_f16` + `emb_arcface_f16`)
- ✓ 1024-d FAISS index `faiss_index_arcface_geo.bin` + labels
- ✓ ConvNeXt K test parquets (existing, unchanged)
