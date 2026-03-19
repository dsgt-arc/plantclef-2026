# Exp-4: Enhanced Tile Aggregation — Research Plan

## Motivation

In Exp-3 (Noisy Student Learning), we built a full semi-supervised pipeline that trained a CrossRegionTransformer student on pseudo-labels from a DINOv2-based inference pipeline. After exhaustive ablation (26 training runs, ensemble blending, sharpened pseudo-labels), the student conclusively could not exceed the inference pipeline:

- **Best inference pipeline**: F1=0.28093 (T=2.0, k=6, thr=0.03)
- **Best student**: F1=0.22145 (R-Drop lambda=0.001)
- **Ensemble**: always worse than the pipeline alone (student adds noise)

Rather than continuing to train a model to beat the pipeline, Exp-4 pivots to improving the pipeline directly. Two major untapped opportunities were identified:

1. **Multi-view TTA** — The embedding extraction step already stores 3-view CLS token embeddings (original, horizontal flip, 90-degree rotation) for every test tile, but only the original-view logits are used during aggregation. The stored embeddings can produce logits via a single linear layer — no re-extraction needed.

2. **Alternative aggregation methods** — The only tile-to-image aggregation method ever tested was Noisy-OR. The submission topk was also fixed at 10 without testing alternatives.

## Background

### Current Pipeline (Exp-3 Phases 1-2)

```
Image → Tiling (4x3 grid) → 16 tiles per image
                                  ↓
                         DINOv2 inference per tile
                         (3 views: orig, flip, rot90)
                                  ↓
                         Per-tile outputs stored:
                           - teacher_logits_f16 [7806]     ← original view logits (USED)
                           - embedding_orig_f16 [768]      ← original CLS token
                           - embedding_flip_f16 [768]      ← flip CLS token (UNUSED)
                           - embedding_rot90_f16 [768]     ← rot90 CLS token (UNUSED)
                           - max_prob, passed_aug_consistency
                                  ↓
                         Tile filtering (top-k by confidence)
                                  ↓
                         Noisy-OR aggregation → image probs [7806]
                                  ↓
                         Top-10 species → submission CSV
```

### Key Insight

The classifier head (`model.head`) is a single linear layer: `Linear(768, 7806)`. Computing logits from a stored embedding is just one matrix multiply:

```
logits = embedding @ W.T + b    # [768] @ [768, 7806] + [7806] → [7806]
```

This means flip and rot90 logits can be computed from stored embeddings in milliseconds per tile, without any GPU inference. Averaging logits across views reduces per-tile variance — a standard TTA technique that typically yields 1-3% improvement.

## Experimental Design

### Step 1: Multi-View Strategies (highest priority)

Test how to combine the 3 views (original, flip, rot90) before or during tile-to-image aggregation.

| Strategy | Description | Tile Count |
|----------|-------------|------------|
| `avg_logits` | Average 3 logit vectors per tile, then aggregate | k (unchanged) |
| `all_tiles` | Treat each view as a separate tile in the aggregator | 3 x k |
| `per_view_or` | Aggregate each view independently, average the 3 image-level prob vectors | k per view |
| `orig_only` | Baseline — only original logits (current behavior) | k |

**Sweep parameters**:
- View strategy: all 4 above
- Temperature: 1.5, 2.0, 2.5 (2.0 is current best)
- tile_filter_k: 4, 6, 8 for avg_logits/per_view_or; 2-16 for all_tiles (effective count triples)

**Why it could work**: TTA is a proven variance-reduction technique. The embeddings exist but are thrown away — using them is free information.

### Step 2: Submission topk Sweep (quick win)

The submission topk (number of species predicted per image) has been fixed at 10 without testing alternatives. Since the competition metric (macro F1) is sensitive to both precision and recall, the optimal topk depends on the confidence distribution.

**Sweep**: topk in {5, 8, 10, 12, 15, 20} with threshold in {0.0, 0.03}

**Why it could work**: If the model is generally confident (few species per plot), lower topk improves precision. If many low-confidence species are correct, higher topk improves recall.

### Step 3: Alternative Aggregation Methods (medium priority)

Noisy-OR is the only tile-to-image aggregation ever tested. It has a known weakness: multiplicative sensitivity to noise. Every tile with non-zero probability inflates the aggregated score, which is why k=6 beats k=16 (more tiles = more noise inflation).

| Method | Formula | Properties |
|--------|---------|------------|
| `noisy_or` | P(c) = 1 - prod_t(1 - p_{t,c}) | Current baseline. Multiplicative — inflates with more tiles. |
| `weighted_noisy_or` | P(c) = 1 - prod_t(1 - w_t * p_{t,c}), w_t = max_prob_t / sum(max_prob) | Suppresses low-confidence tiles. May handle k>6 better. |
| `mean_prob` | P(c) = mean_t(p_{t,c}) | Simple average. Linear — no noise inflation. |
| `max_prob` | P(c) = max_t(p_{t,c}) | Conservative. One confident tile is sufficient. |

**Sweep**: all 4 methods with avg_logits and orig_only strategies, k in {6, 10, 16}

**Why it could work**: Alternative methods may tolerate more tiles (k>6) better than Noisy-OR, and mean/max may produce cleaner probability distributions for the topk selection.

### Step 4: Combined Best Config

After Steps 1-3, combine the winners:
- Best view strategy x best aggregation method x best topk x best temperature/k

This is the final submission config.

## Expected Outcomes

| Step | Hypothesis | Success Criterion |
|------|-----------|-------------------|
| 1 | Multi-view TTA reduces per-tile variance | F1 > 0.28093 for at least one strategy |
| 2 | Optimal topk differs from 10 | F1 improves at some topk != 10 |
| 3 | Alternative aggregation handles more tiles better | F1 with k>6 + alt method > F1 with k=6 + noisy_or |
| 4 | Combined config is best overall | F1 > any single-step winner |

## Data Dependencies

All data comes from Exp-3 Phase 1 — no new extraction needed.

- **Test tile parquet**: `$PLANTCLEF_ROOT/aerten3/parquet/plantclef_2026_test/phase1_tiles_test/`
  - Columns used: `image_name`, `teacher_logits_f16`, `embedding_flip_f16`, `embedding_rot90_f16`, `max_prob`, `passed_aug_consistency`
- **DINOv2 checkpoint** (for head extraction only): `$PLANTCLEF_ROOT/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth`
- **Species map**: `$PLANTCLEF_ROOT/data/raw/class_id_to_species_mapping/class_to_species_id.json`
- **Geo mask**: `$PLANTCLEF_ROOT/aerten3/species_masking/sw_europe_mask_bool.npy`

## Compute Budget

| Step | Resource | Estimated Time |
|------|----------|---------------|
| 1 | CPU (48 GB RAM) | ~2 hours |
| 2 | CPU (minimal) | ~1 minute |
| 3 | CPU (32 GB RAM) | ~1 hour |
| 4 | CPU (minimal) | ~1 minute |
| **Total** | | **~3-4 hours** |

No GPU is needed — the classifier head is extracted as a numpy weight matrix and logits are computed via matrix multiply on CPU.
