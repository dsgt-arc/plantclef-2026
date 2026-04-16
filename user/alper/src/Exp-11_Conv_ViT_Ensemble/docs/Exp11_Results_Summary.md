# Exp-11: ConvNeXt + ViT Ensemble — Results Summary

## Objective

Test whether ensembling a fine-tuned ConvNeXt-Large DINOv3 model with the existing ViT-Base DINOv2 pipeline improves PlantCLEF F1 score beyond either model alone.

## Models

| Model | Pretraining | Params | Top-1 Acc | Training |
|-------|------------|--------|-----------|----------|
| ViT-Base DINOv2 | DINOv2 (ViT-B/14) | 86M | 76.0% | Phase 1 (100ep) + Phase 2 (100ep) |
| ConvNeXt-Large DINOv3 | DINOv3 (distilled) | 196M | 74.5% | Phase 1 (100ep) + Phase 2 (100ep, 4xH200) |

## Pipeline Evolution

```
Exp-8:  ViT + multiscale tiling + temporal fusion           → F1 = 0.4010
Exp-9:  ViT + multiscale tiling + kNN + temporal fusion     → F1 = 0.4175
Exp-11: ViT+kNN + ConvNeXt ensemble + temporal fusion       → F1 = 0.4379
                                                              (+0.020 over Exp-9)
```

---

## Phase 1: Quick Ensemble Validation (4x4 tiling only)

Tested whether ensembling helps at all before investing in full pipeline integration.

### Standalone Baselines (4x4)

| Model | Best F1 | Best Threshold |
|-------|---------|---------------|
| ViT 4x4 | 0.3407 | 0.050 |
| ConvNeXt 4x4 | 0.3165 | 0.040 |

### Fixed-Weight Ensemble (4x4)

| Weight (ViT) | Best F1 | Threshold |
|-------------|---------|-----------|
| **0.6** | **0.3606** | 0.040 |
| 0.7 | 0.3584 | 0.040 |
| 0.8 | 0.3565 | 0.040 |

**Finding**: +0.02 F1 gain from ensembling at 4x4 scale. Optimal weight: 60% ViT / 40% ConvNeXt.

### Confidence-Gated and Per-Class Max (4x4)

| Method | Best F1 |
|--------|---------|
| Fixed blend (w=0.6) | **0.3606** |
| Confidence-gated (best) | 0.3600 |
| Per-class max | 0.3432 |

**Finding**: Simple fixed-weight blending beats adaptive methods at this stage.

---

## Phase 2: Multi-Scale Ensemble

### ConvNeXt Standalone — Scale Optimization

| Scheme | Tiles | F1 (thr=0.100) |
|--------|-------|---------------|
| 4+5+6 | 77 | 0.3620 |
| 3+4+5+6 | 86 | 0.3677 |
| 4+5+6+7 | 126 | 0.3783 |
| 3+4+5+6+7 | 135 | 0.3823 |
| 2+3+4+5+6+7 | 139 | 0.3825 |
| 3+4+5+6+7+8 | 199 | 0.3713 |

**Finding**: ConvNeXt benefits from scales 3-7 (unlike ViT which only benefits from 4-6). Best standalone: 3+4+5+6+7 at thr=0.110 → F1 = 0.3853.

### ConvNeXt Parameter Sweep

| Parameter | Optimal | Notes |
|-----------|---------|-------|
| Temperature | 1.5 | Same as ViT |
| Alpha (temporal fusion) | 0.05-0.15 | Similar to ViT |
| Threshold | 0.100-0.110 | Slightly higher than ViT for more scales |

**Finding**: ConvNeXt's optimal parameters match ViT's closely. No significant gains from retuning.

### Multi-Scale Ensemble Results

| ConvNeXt Scheme | Weight | F1 |
|----------------|--------|-----|
| 4+5+6 | w=0.78 | **0.4224** |
| 4+5+6 | w=0.80 | 0.4224 |
| 5+6 | w=0.80 | 0.4199 |
| 3+4+5+6+7 | w=0.78 | 0.4167 |
| 6 only | w=0.78 | 0.4159 |

**Key finding**: Matching ViT's scale scheme (4+5+6) gives the best ensemble. Extra scales that help ConvNeXt standalone (3x3, 7x7) hurt the ensemble because they introduce predictions that disagree with ViT.

### JPEG Compression Test

| ConvNeXt 4x4 | F1 |
|-------------|-----|
| With JPEG85 | 0.3165 |
| Without JPEG | 0.3148 |

**Finding**: JPEG85 compression helps ConvNeXt slightly (+0.002), consistent with ViT behavior.

---

## Phase 3: kNN Integration

### Approach A: ViT+kNN first, then blend with ConvNeXt (WINNER)

```
ViT tiles → kNN (k=20, tau=0.07, beta=0.80) → aggregate → temporal fusion → P_vit_knn
ConvNeXt tiles → aggregate → temporal fusion → P_cnxt
P_final = w * P_vit_knn + (1-w) * P_cnxt
```

| Weight | F1 |
|--------|-----|
| w=0.75 | 0.4287 |
| **w=0.78** | **0.4354** |
| w=0.80 | 0.4352 |
| w=0.82 | 0.4279 |
| w=0.85 | 0.4223 |
| ViT+kNN only | 0.4175 |

**Best: F1 = 0.4354** (+0.018 over Exp-9 baseline)

### Approach B: Blend at tile level, then kNN

```
Per tile: blended_logits = w * vit_logits + (1-w) * cnxt_logits
blended_softmax → kNN blend → aggregate → temporal fusion
```

| Weight | Beta | F1 |
|--------|------|-----|
| w=0.78 | b=0.80 | 0.4140 |
| w=0.78 | b=0.85 | 0.4138 |
| w=0.80 | b=0.80 | 0.4129 |

**Finding**: Tile-level blending before kNN hurts performance (0.414 vs 0.435). The kNN index was built from ViT embeddings, so blending ConvNeXt logits before kNN creates a mismatch between the softmax probabilities and the kNN neighbor space.

### Approach C: Tile-level blend, retune kNN

Swept k={5,10,20,50}, tau={0.05,0.07,0.10}, beta={0.80,0.85,0.90,0.95} on pre-merged tiles.

**Best: F1 = 0.4145** (k=20, tau=0.10, b=0.85)

**Finding**: Even with full kNN retuning, tile-level blending cannot match Approach A. The fundamental issue is the ViT-embedding ↔ blended-logit mismatch.

### ConvNeXt Scale Scheme with Approach A

| ConvNeXt Scheme | Best F1 | Best Weight |
|----------------|---------|-------------|
| **4+5+6** | **0.4354** | w=0.78 |
| 4+5+6+7 | 0.4286 | w=0.75 |
| 3+4+5+6+7 | 0.4287 | w=0.75 |

**Finding**: ConvNeXt 4+5+6 remains optimal for ensembling even with kNN, confirming that matching ViT's scale scheme gives the cleanest complementary signal.

---

## Phase 4: Final Tuning

### Threshold Sweep (w=0.78, alpha=0.05)

| Threshold | F1 |
|-----------|-----|
| 0.085 | 0.4201 |
| 0.090 | 0.4190 |
| 0.095 | 0.4267 |
| **0.100** | **0.4354** |
| 0.105 | 0.4316 |
| 0.110 | 0.4255 |

**Finding**: thr=0.100 remains optimal.

### Fine Weight Sweep (alpha=0.05, thr=0.100)

| Weight | F1 |
|--------|-----|
| 0.76 | 0.4358 |
| **0.77** | **0.4366** |
| 0.78 | 0.4354 |
| 0.79 | 0.4350 |

**Finding**: w=0.77 slightly better than w=0.78.

### ConvNeXt Alpha Sweep (w=0.77, thr=0.100)

| Alpha | F1 |
|-------|-----|
| 0.05 | 0.4366 |
| 0.10 | 0.4356 |
| 0.15 | 0.4375 |
| 0.20 | 0.4374 |
| 0.25 | 0.4378 |
| **0.30** | **0.4379** |
| 0.35 | 0.4360 |

**Finding**: ConvNeXt benefits from less temporal fusion correction (alpha=0.30) since ViT+kNN already provides location-aware predictions.

---

## Final Best Configuration

| Component | Setting |
|-----------|---------|
| **ViT model** | vit_base_patch14_reg4_dinov2.lvd142m |
| **ViT scales** | 4x4 + 5x5 + 6x6 (77 tiles) |
| **ViT aggregation** | max_prob, T=1.5 |
| **ViT kNN** | k=20, tau=0.07, beta_multi=0.80, beta_single=1.00 |
| **ViT temporal fusion** | prior_blend, alpha=0.15 |
| **ConvNeXt model** | convnext_large.dinov3_lvd1689m |
| **ConvNeXt scales** | 4x4 + 5x5 + 6x6 (77 tiles) |
| **ConvNeXt aggregation** | max_prob, T=1.5 |
| **ConvNeXt temporal fusion** | prior_blend, alpha=0.30 |
| **Ensemble weight** | w=0.77 (ViT+kNN) / 0.23 (ConvNeXt) |
| **Threshold** | 0.100 |
| **TopK** | 10 |
| **JPEG** | quality=85, subsampling=4:2:2 |

## Final F1 = 0.4379

| Milestone | F1 | Improvement |
|-----------|-----|-------------|
| Exp-8: ViT + multiscale | 0.4010 | baseline |
| Exp-9: + kNN | 0.4175 | +0.0165 |
| Exp-11: + ConvNeXt ensemble | 0.4354 | +0.0179 |
| Exp-11: + fine tuning | **0.4379** | +0.0204 |

---

## Key Takeaways

1. **Ensembling works**: ConvNeXt adds +0.02 F1 on top of the fully optimized ViT+kNN pipeline, despite being individually weaker (74.5% vs 76.0% top-1).

2. **Order matters**: kNN must be applied to ViT alone first, then ConvNeXt blended after. Blending before kNN disrupts the ViT-embedding ↔ logit alignment that kNN relies on.

3. **Match scale schemes**: Using the same tiling (4+5+6) for both models gives the best ensemble. Extra scales that help ConvNeXt standalone hurt the ensemble.

4. **Simple blending wins**: Fixed-weight probability averaging outperforms confidence-gated, per-class max, and other adaptive methods.

5. **Different temporal fusion alphas**: ViT+kNN uses alpha=0.15 (trusts location priors more), ConvNeXt uses alpha=0.30 (trusts its own predictions more since ViT+kNN already handles location correction).

6. **Architecture diversity is the value**: The 1.5% top-1 accuracy gap between models doesn't prevent meaningful ensemble gains because ConvNeXt (convolutions) and ViT (attention) make different errors.
