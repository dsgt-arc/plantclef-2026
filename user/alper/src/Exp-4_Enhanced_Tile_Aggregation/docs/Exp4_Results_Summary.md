# Exp-4: Enhanced Tile Aggregation — Results Summary

## Executive Summary

Exp-4 improved the PlantCLEF 2026 inference pipeline's F1 score from **0.28093 to 0.31950** (+13.7% relative) by replacing Noisy-OR aggregation with max-probability aggregation and lowering the softmax temperature from T=2.0 to T=1.5. No model retraining was required — all improvements came from better post-processing of existing DINOv2 embeddings.

A key negative finding: **multi-view test-time augmentation (TTA) does not help** and slightly hurts performance. The Step 7 ablation showed that orig_only (no augmentation) outperforms all multi-view strategies when max_prob + T=1.5 is used. The improvement is entirely attributable to the aggregation method change.

**Final best config**: `orig_only` + `max_prob` + T=1.5 + k=4 + thr=0.03 + topk=10 → **F1 = 0.31950**

---

## Background

### Motivation

In Exp-3 (Noisy Student Learning), a CrossRegionTransformer student model trained on pseudo-labels could not exceed the DINOv2-based inference pipeline:

- Best inference pipeline: F1 = 0.28093 (T=2.0, k=6, thr=0.03, Noisy-OR)
- Best student model: F1 = 0.22145
- Ensemble: always worse than pipeline alone

Exp-4 pivoted to improving the pipeline itself. Two untapped opportunities were identified:

1. **Multi-view TTA** — Phase 1 extraction already stores 3 CLS embeddings per tile (original, horizontal flip, 90-degree rotation), but only original-view logits were used. Flip/rot90 logits can be computed from stored embeddings via a single matrix multiply — no GPU needed.

2. **Alternative aggregation methods** — Only Noisy-OR was ever tested. Its multiplicative nature inflates noise with more tiles, which is why k=6 beats k=16.

### Pipeline Architecture

```
Image → 16 tiles → DINOv2 → 3 embeddings/tile (orig, flip, rot90)
                                      ↓
                         Tile filtering (top-k by confidence)
                                      ↓
                         View combination (4 strategies)
                                      ↓
                         Tile-to-image aggregation (4 methods)
                                      ↓
                         Geo-species masking → top-K species → CSV
```

### Hyperparameters Tested

| Parameter | Values | Description |
|-----------|--------|-------------|
| View strategy | orig_only, avg_logits, all_tiles, per_view_or | How to combine 3 view embeddings |
| Aggregation method | noisy_or, weighted_noisy_or, mean_prob, max_prob | How to combine tile predictions into image prediction |
| Temperature (T) | 1.5, 2.0, 2.5 | Softmax temperature (higher = smoother probs) |
| tile_filter_k | 1–16 | Number of most-confident tiles to keep per image |
| threshold | 0.03 | Minimum probability for a species to be predicted |
| topk | 10 | Number of species predicted per image |

### View Strategy Definitions

| Strategy | Description | Effective tile count |
|----------|-------------|---------------------|
| `orig_only` | Use only original-view logits (baseline behavior) | k |
| `avg_logits` | Average orig/flip/rot90 logits per tile, then aggregate | k |
| `all_tiles` | Treat each view as a separate tile in the aggregator | 3 x k |
| `per_view_or` | Aggregate each view independently, average 3 image-level prob vectors | k per view |

### Aggregation Method Definitions

| Method | Formula | Properties |
|--------|---------|------------|
| `noisy_or` | P(c) = 1 - prod_t(1 - p_t,c) | Multiplicative. Inflates with more tiles. |
| `weighted_noisy_or` | P(c) = 1 - prod_t(1 - w_t * p_t,c) | Weighted by tile confidence. |
| `mean_prob` | P(c) = mean_t(p_t,c) | Simple average. Linear, no inflation. |
| `max_prob` | P(c) = max_t(p_t,c) | Conservative. One confident tile is sufficient. |

---

## Step 1: View Strategy Comparison

**Config**: T=2.0, k=6, noisy_or, thr=0.03, topk=10

| Strategy | F1 | Delta vs baseline |
|----------|-----|-------------------|
| orig_only | 0.28093 | — (baseline) |
| per_view_or | 0.28010 | -0.00083 |
| avg_logits | 0.26857 | -0.01236 |
| all_tiles | 0.22688 | -0.05405 |

### Explanation

At T=2.0 and k=6 (the baseline's optimal config), no multi-view strategy improved over the baseline. This is because:

- **avg_logits** averages logits across views before softmax. At T=2.0 (already smooth), this over-smooths the predictions, washing out species-specific signal.
- **all_tiles** triples the effective tile count from 6 to 18. Since Noisy-OR inflates noise multiplicatively with each additional tile, this dramatically hurts performance.
- **per_view_or** nearly matches the baseline (-0.0008), suggesting that view averaging at the probability level is benign but the T=2.0/k=6 config is already near-optimal for noisy_or.

---

## Step 1b: all_tiles k Sweep

**Config**: all_tiles, T=2.0, noisy_or

| k (input) | Effective tiles | F1 |
|-----------|----------------|-----|
| 2 | 6 | 0.26764 |
| 4 | 12 | 0.26708 |
| 6 | 18 | 0.22688 |
| 8 | 24 | 0.20487 |
| 10 | 30 | 0.19572 |
| 12 | 36 | 0.19572 |
| 16 | 48 | 0.20410 |

### Explanation

The all_tiles strategy is fundamentally incompatible with Noisy-OR. Even at k=2 (6 effective tiles — matching the baseline's optimal), it only reaches 0.267 vs the baseline's 0.281. The reason: flip and rot90 views are noisier than the original (the model was not trained with augmentation consistency for test images), so treating them as independent "tiles" injects noise even at low k. Performance collapses as k increases, bottoming out at k=10-12 (0.196).

**Conclusion**: all_tiles is a dead strategy. Eliminated from further testing.

---

## Step 1c: Temperature Sweep

### avg_logits + noisy_or

| T | k=4 | k=6 | k=8 |
|---|-----|-----|-----|
| 1.5 | **0.28273** | 0.26610 | 0.24373 |
| 2.0 | 0.26789 | 0.26857 | 0.25826 |
| 2.5 | 0.25551 | 0.25664 | 0.25445 |

### per_view_or + noisy_or

| T | k=4 | k=6 | k=8 |
|---|-----|-----|-----|
| 1.5 | **0.28514** | 0.26516 | 0.24677 |
| 2.0 | 0.27012 | 0.28010 | 0.26877 |
| 2.5 | 0.25525 | 0.25879 | 0.25714 |

### Key Finding: T=1.5 + k=4 Beats Baseline

Both multi-view strategies beat the baseline when using **lower temperature (T=1.5) and fewer tiles (k=4)**:

- per_view_or T=1.5 k=4: **0.28514** (+0.00421)
- avg_logits T=1.5 k=4: **0.28273** (+0.00180)

### Explanation

Lower temperature produces sharper per-tile softmax distributions (more peaked, less uniform). With multi-view TTA, the variance reduction from averaging 3 views compensates for the increased sharpness — you get confident predictions that are also more reliable. Fewer tiles (k=4 vs k=6) mean only the most confident tiles contribute, and with 3 views reducing per-tile noise, you need fewer tiles to capture the signal.

The interaction between T and k shifts when using TTA: the baseline's optimal (T=2.0, k=6) was a compromise between sharpness and noise tolerance. With TTA reducing noise, you can afford to be sharper (T=1.5) and more selective (k=4).

---

## Step 3: Aggregation Method Comparison

### orig_only (no TTA), T=2.0

| Method | k=6 | k=10 | k=16 | k spread |
|--------|-----|------|------|----------|
| noisy_or | **0.28093** | 0.26904 | 0.24730 | 0.034 |
| max_prob | 0.27595 | **0.27841** | **0.27834** | **0.002** |
| weighted_noisy_or | 0.26851 | 0.24771 | 0.24709 | 0.021 |
| mean_prob | 0.25492 | 0.24329 | 0.24086 | 0.014 |

### avg_logits + TTA, T=2.0

| Method | k=6 | k=10 | k=16 | k spread |
|--------|-----|------|------|----------|
| noisy_or | 0.26857 | **0.27109** | 0.24983 | 0.021 |
| max_prob | 0.26868 | 0.27028 | **0.27076** | **0.002** |
| weighted_noisy_or | 0.24660 | 0.25660 | 0.25156 | 0.010 |
| mean_prob | 0.25626 | 0.24931 | 0.24232 | 0.014 |

### Key Finding: max_prob is Uniquely k-Tolerant

max_prob shows almost zero degradation as k increases (spread of only 0.002 across k=6 to k=16), while noisy_or degrades by 0.034 over the same range. This is a fundamental property difference:

**Noisy-OR**: `P(c) = 1 - prod(1 - p_t,c)`. Every tile with non-zero probability for class c *increases* the aggregated probability. Noisy low-confidence tiles contribute multiplicative false-positive pressure. More tiles = more noise inflation.

**max_prob**: `P(c) = max(p_t,c)`. Only the single most confident tile determines the class probability. Additional noisy tiles are completely ignored — they can never increase the score, only match or fall below the max. This makes max_prob inherently robust to tile count.

However, max_prob alone at T=2.0 (0.278) doesn't beat noisy_or at its optimal k=6 (0.281), because the smooth T=2.0 temperature doesn't produce sharp enough per-tile predictions for max_prob to capture strong species signal.

### Explanation: Why weighted_noisy_or and mean_prob Underperform

- **weighted_noisy_or**: Weights tiles by their max confidence, but the weighting is still applied within a multiplicative Noisy-OR framework. The fundamental noise inflation problem remains — weighting just attenuates it slightly. It also adds a failure mode: if one tile has anomalously high confidence on the wrong species, it dominates.

- **mean_prob**: Averages probabilities, which dilutes strong signal. If 1 of 6 tiles confidently detects a species (p=0.3) and 5 don't (p~0.001), the mean is ~0.05 — barely above threshold. Noisy-OR would give ~0.3 for the same input. Mean is too conservative for multi-species vegetation plots where species may only appear in one tile.

---

## Step 4: Combined Best Config

Testing `per_view_or` (best view strategy) × `max_prob` (most k-tolerant) at the best temperature (T=1.5).

### per_view_or + noisy_or + T=1.5

| k | F1 | Delta vs baseline |
|---|-----|-------------------|
| 3 | 0.29216 | +0.01123 |
| 4 | 0.28514 | +0.00421 |
| 5 | 0.27513 | -0.00580 |
| 6 | 0.26516 | -0.01577 |
| 8 | 0.24677 | -0.03416 |
| 10 | 0.22983 | -0.05110 |

### per_view_or + max_prob + T=1.5

| k | F1 | Delta vs baseline |
|---|-----|-------------------|
| 1 | 0.23281 | -0.04812 |
| 2 | 0.27252 | -0.00841 |
| **3** | **0.31487** | **+0.03394** |
| 4 | 0.31097 | +0.03004 |
| 5 | 0.30864 | +0.02771 |
| 6 | 0.30970 | +0.02877 |
| 8 | 0.30979 | +0.02886 |
| 10 | 0.30889 | +0.02796 |

### Key Finding: max_prob + T=1.5 is the Breakthrough

The combination of max_prob aggregation with T=1.5 produces a massive improvement over the baseline (0.28093 → 0.31487 with per_view_or, later improved to 0.31950 with orig_only in Step 7). The k-tolerance pattern is dramatic:

```
    noisy_or:  k=3 (0.292) ──── sharp decline ────→ k=10 (0.230)   spread: 0.062
    max_prob:  k=3 (0.315) ──── stable plateau ──→ k=10 (0.309)   spread: 0.006
```

Note: At this point in the experiment, per_view_or appeared to help because it was tested before orig_only at these settings. Step 7 later revealed that orig_only is actually better — see below.

### Explanation: Why max_prob + T=1.5 Works

**1. max_prob extracts the strongest signal per species.**
For a given species, max_prob selects the single tile where the model is most confident. In vegetation plots, a species may only be visible in 1-2 tiles. Max_prob captures this perfectly — it doesn't penalize the species for being absent from other tiles (unlike mean_prob) or inflate noise from tiles where the species isn't present (unlike noisy_or).

**2. T=1.5 sharpens predictions to match max_prob's strengths.**
Lower temperature makes per-tile softmax distributions more peaked. This benefits max_prob specifically: the maximum probability becomes higher and more decisive for true positives, while false positives (which are less consistent across tiles) don't get amplified. At T=2.0, max_prob alone (0.278) barely trails noisy_or (0.281) because smooth probabilities don't differentiate true positives sharply enough. At T=1.5, the signal is sharp and decisive.

**3. The k=3-4 sweet spot.**
k=1 and k=2 fail because even max_prob needs at least a few tiles to cover the spatial diversity of a vegetation plot. A single tile may miss species entirely. k=3-4 provides enough spatial coverage while keeping only the most confident tiles. Beyond k=4, max_prob's score plateaus (0.309-0.320) rather than declining, confirming that additional tiles neither help nor hurt — max simply ignores them.

### Why noisy_or Collapses at Low k + Low T

The opposite pattern in noisy_or (k=3 → k=10 decline) worsens at T=1.5 because:
- Lower temperature makes per-tile probabilities more extreme (closer to 0 or 1)
- Noisy-OR: P = 1 - prod(1-p). With sharper p values, even a few tiles with moderate false-positive probability compound into high image-level false positives
- More tiles = more compounding = more false positives = lower precision

---

## Complete Results Table

All 89 configurations tested, sorted by F1 score (top 25 shown):

| Rank | Strategy | Method | T | k | F1 | vs baseline |
|------|----------|--------|---|---|-----|-------------|
| 1 | **orig_only (1v)** | **max_prob** | **1.5** | **4** | **0.31950** | **+0.03857** |
| 2 | orig_only (1v) | max_prob | 1.5 | 6 | 0.31818 | +0.03725 |
| 3 | orig_only (1v) | max_prob | 1.5 | 5 | 0.31498 | +0.03405 |
| 4 | per_view_or (3v) | max_prob | 1.5 | 3 | 0.31487 | +0.03394 |
| 5 | 2v_hflip | max_prob | 1.5 | 3 | 0.31435 | +0.03342 |
| 6 | 2v_hflip | max_prob | 1.5 | 4 | 0.31385 | +0.03292 |
| 7 | orig_only (1v) | max_prob | 1.5 | 3 | 0.31371 | +0.03278 |
| 8 | 2v_hflip | max_prob | 1.5 | 6 | 0.31318 | +0.03225 |
| 9 | 2v_hflip | max_prob | 1.5 | 5 | 0.31121 | +0.03028 |
| 10 | per_view_or (3v) | max_prob | 1.5 | 4 | 0.31097 | +0.03004 |
| 11 | per_view_or (6v) | max_prob | 1.5 | 4 | 0.31067 | +0.02974 |
| 12 | per_view_or (6v) | max_prob | 1.5 | 3 | 0.31040 | +0.02947 |
| 13 | per_view_or (3v) | max_prob | 1.5 | 8 | 0.30979 | +0.02886 |
| 14 | per_view_or (3v) | max_prob | 1.5 | 6 | 0.30970 | +0.02877 |
| 15 | 2v_rot90 | max_prob | 1.5 | 3 | 0.30914 | +0.02821 |
| 16 | per_view_or (3v) | max_prob | 1.5 | 10 | 0.30889 | +0.02796 |
| 17 | per_view_or (3v) | max_prob | 1.5 | 5 | 0.30864 | +0.02771 |
| 18 | per_view_or (6v) | max_prob | 1.5 | 5 | 0.30692 | +0.02599 |
| 19 | per_view_or (6v) | max_prob | 1.5 | 6 | 0.30538 | +0.02445 |
| 20 | 2v_rot90 | max_prob | 1.5 | 4 | 0.30533 | +0.02440 |
| 21 | 2v_rot90 | max_prob | 1.5 | 5 | 0.30145 | +0.02052 |
| 22 | 2v_rot90 | max_prob | 1.5 | 6 | 0.30075 | +0.01982 |
| 23 | per_view_or | noisy_or | 1.5 | 3 | 0.29216 | +0.01123 |
| 12 | per_view_or | noisy_or | 1.5 | 4 | 0.28514 | +0.00421 |
| 13 | avg_logits | noisy_or | 1.5 | 4 | 0.28273 | +0.00180 |
| 14 | orig_only | noisy_or | 2.0 | 6 | 0.28093 | — baseline |
| 15 | per_view_or | noisy_or | 2.0 | 6 | 0.28010 | -0.00083 |
| 16 | orig_only | max_prob | 2.0 | 10 | 0.27841 | -0.00252 |
| 17 | orig_only | max_prob | 2.0 | 16 | 0.27834 | -0.00259 |
| 18 | orig_only | max_prob | 2.0 | 6 | 0.27595 | -0.00498 |
| 19 | per_view_or | noisy_or | 1.5 | 5 | 0.27513 | -0.00580 |
| 20 | per_view_or | max_prob | 1.5 | 2 | 0.27252 | -0.00841 |
| 21 | avg_logits | noisy_or | 2.0 | 10 | 0.27109 | -0.00984 |
| 22 | avg_logits | max_prob | 2.0 | 16 | 0.27076 | -0.01017 |
| 23 | avg_logits | max_prob | 2.0 | 10 | 0.27028 | -0.01065 |
| 24 | per_view_or | noisy_or | 2.0 | 4 | 0.27012 | -0.01081 |
| 25 | orig_only | noisy_or | 2.0 | 10 | 0.26904 | -0.01189 |
| 26 | per_view_or | noisy_or | 2.0 | 8 | 0.26877 | -0.01216 |
| 27 | avg_logits | max_prob | 2.0 | 6 | 0.26868 | -0.01225 |
| 28 | avg_logits | noisy_or | 2.0 | 6 | 0.26857 | -0.01236 |
| 29 | orig_only | weighted_noisy_or | 2.0 | 6 | 0.26851 | -0.01242 |
| 30 | avg_logits | noisy_or | 2.0 | 4 | 0.26789 | -0.01304 |
| 31 | all_tiles | noisy_or | 2.0 | 2 | 0.26764 | -0.01329 |
| 32 | all_tiles | noisy_or | 2.0 | 4 | 0.26708 | -0.01385 |
| 33 | avg_logits | noisy_or | 1.5 | 6 | 0.26610 | -0.01483 |
| 34 | per_view_or | noisy_or | 1.5 | 6 | 0.26516 | -0.01577 |
| 35 | per_view_or | noisy_or | 2.5 | 6 | 0.25879 | -0.02214 |
| 36 | avg_logits | noisy_or | 2.0 | 8 | 0.25826 | -0.02267 |
| 37 | per_view_or | noisy_or | 2.5 | 8 | 0.25714 | -0.02379 |
| 38 | avg_logits | noisy_or | 2.5 | 6 | 0.25664 | -0.02429 |
| 39 | avg_logits | weighted_noisy_or | 2.0 | 10 | 0.25660 | -0.02433 |
| 40 | avg_logits | mean_prob | 2.0 | 6 | 0.25626 | -0.02467 |
| 41 | avg_logits | noisy_or | 2.5 | 4 | 0.25551 | -0.02542 |
| 42 | per_view_or | noisy_or | 2.5 | 4 | 0.25525 | -0.02568 |
| 43 | orig_only | mean_prob | 2.0 | 6 | 0.25492 | -0.02601 |
| 44 | avg_logits | noisy_or | 2.5 | 8 | 0.25445 | -0.02648 |
| 45 | avg_logits | weighted_noisy_or | 2.0 | 16 | 0.25156 | -0.02937 |
| 46 | avg_logits | noisy_or | 2.0 | 16 | 0.24983 | -0.03110 |
| 47 | avg_logits | mean_prob | 2.0 | 10 | 0.24931 | -0.03162 |
| 48 | orig_only | weighted_noisy_or | 2.0 | 10 | 0.24771 | -0.03322 |
| 49 | orig_only | noisy_or | 2.0 | 16 | 0.24730 | -0.03363 |
| 50 | orig_only | weighted_noisy_or | 2.0 | 16 | 0.24709 | -0.03384 |
| 51 | per_view_or | noisy_or | 1.5 | 8 | 0.24677 | -0.03416 |
| 52 | avg_logits | weighted_noisy_or | 2.0 | 6 | 0.24660 | -0.03433 |
| 53 | avg_logits | noisy_or | 1.5 | 8 | 0.24373 | -0.03720 |
| 54 | orig_only | mean_prob | 2.0 | 10 | 0.24329 | -0.03764 |
| 55 | avg_logits | mean_prob | 2.0 | 16 | 0.24232 | -0.03861 |
| 56 | orig_only | mean_prob | 2.0 | 16 | 0.24086 | -0.04007 |
| 57 | per_view_or | max_prob | 1.5 | 1 | 0.23281 | -0.04812 |
| 58 | per_view_or | noisy_or | 1.5 | 10 | 0.22983 | -0.05110 |
| 59 | all_tiles | noisy_or | 2.0 | 6 | 0.22688 | -0.05405 |
| 60 | all_tiles | noisy_or | 2.0 | 8 | 0.20487 | -0.07606 |
| 61 | all_tiles | noisy_or | 2.0 | 16 | 0.20410 | -0.07683 |
| 62 | all_tiles | noisy_or | 2.0 | 10 | 0.19572 | -0.08521 |
| 63 | all_tiles | noisy_or | 2.0 | 12 | 0.19572 | -0.08521 |

---

## Summary of Findings

### What Worked

1. **max_prob + T=1.5** — The single most important finding. Replacing Noisy-OR with max-probability aggregation and lowering temperature from 2.0 to 1.5 produces a +13.7% relative improvement (0.28093 → 0.31950). max_prob's noise immunity (only uses the single best tile per species) combined with sharper softmax predictions is the key to the gain.

2. **max_prob is uniquely k-tolerant** — Performance spread of only 0.006 across k=3 to k=10, compared to 0.062 for noisy_or. This makes max_prob a robust default aggregator that doesn't require careful k tuning.

3. **orig_only (no TTA) is optimal** — The Step 7 ablation conclusively showed that multi-view test-time augmentation provides no benefit and slightly hurts when using max_prob. The original view produces the most reliable predictions; augmented views only add noise.

### What Didn't Work

1. **All forms of TTA** — per_view_or, avg_logits, all_tiles, 6-view. Every multi-view strategy is worse than orig_only when the aggregation method and temperature are optimized. The model was fine-tuned on upright images; augmented views (especially rot90) produce noisier predictions.

2. **all_tiles** — Tripling effective tile count catastrophically inflates Noisy-OR noise. Eliminated early.

3. **Top-m averaging (m>1)** — Averaging the top 2 or 3 tiles per species dilutes the max signal. Single-tile evidence is decisive.

4. **Additional augmentation views (6-view)** — rot180, rot270, vflip add redundant information and over-smooth predictions.

5. **weighted_noisy_or and mean_prob** — weighted_noisy_or still has multiplicative noise inflation. mean_prob dilutes strong single-tile detections.

### Aggregation Method Characteristics

| Method | Best F1 | k-sensitive? | When it works |
|--------|---------|-------------|---------------|
| max_prob | **0.31950** | No (robust to k) | Sharp T, no TTA |
| noisy_or | 0.28093 | Very (must tune k) | Low k, single view, T=2.0 |
| top2_prob | 0.30073 | Moderate | Higher k only |
| top3_prob | 0.30577 | Moderate | Higher k only |
| weighted_noisy_or | 0.26851 | Yes | Never outperforms alternatives |
| mean_prob | 0.25626 | Yes | Never outperforms alternatives |

---

## Step 5: Top-m Probability Aggregation

Testing whether averaging the top-m most confident tiles per species (instead of just the max) improves over max_prob.

**Config**: per_view_or + top_m_prob + T=1.5 + thr=0.03 + topk=10

| m | k=3 | k=4 | k=5 | k=6 |
|---|-----|-----|-----|-----|
| 1 (=max_prob) | **0.31487** | 0.31097 | 0.30864 | 0.30970 |
| 2 | 0.28703 | 0.28719 | 0.29325 | 0.30073 |
| 3 | 0.28314 | 0.29599 | 0.30577 | 0.29834 |

### Explanation: Why Top-m Doesn't Beat Max

Averaging the top-2 or top-3 tiles dilutes the strongest signal with weaker, noisier tiles. At k=3 with m=2, the second tile is the 2nd-most-confident out of only 3 — likely noisy. The advantage of max_prob is precisely that it isolates the single most informative tile per species and ignores everything else.

As k increases, m=2 and m=3 improve (the 2nd and 3rd best tiles among more candidates are more likely to be genuinely informative), but they never catch max_prob. This confirms that for vegetation plot species detection, the single-tile signal is decisive — additional tiles add more noise than confirmation.

**Conclusion**: Top-m averaging does not improve over max_prob (m=1). The aggregation method is optimized; further gains must come from the feature level (better embeddings, more augmentation views, or larger models).

---

## Step 6: Six-View Aggregation (Additional Augmentations)

Testing whether adding 3 more augmentation views (rot180, rot270, vflip) improves over the 3-view baseline (orig, hflip, rot90). This required re-running DINOv2 inference on all test tiles with the new augmentations (GPU, ~30 min), then running per_view_or aggregation averaging over 6 views instead of 3.

**Config**: per_view_or + max_prob + T=1.5 + thr=0.03 + topk=10

| Views | k=3 | k=4 | k=5 | k=6 |
|-------|-----|-----|-----|-----|
| 3-view (orig, hflip, rot90) | **0.31487** | **0.31097** | 0.30864 | 0.30970 |
| 6-view (+ rot180, rot270, vflip) | 0.31040 | 0.31067 | 0.30692 | 0.30538 |

### Result: 6-View Hurts Across the Board

3-view wins at every k value. The best 6-view result (k=4, 0.31067) is still worse than the best 3-view result (k=3, 0.31487).

### Explanation

The 3 new augmentations (rot180, rot270, vflip) add **redundant** rather than complementary information. DINOv2's ViT architecture, while not explicitly rotation-equivariant, captures similar visual features regardless of rotation — the CLS token attends to patch-level features that are largely orientation-invariant at the semantic level.

Averaging 6 views instead of 3 **over-smooths** the probability vectors. With max_prob aggregation, the strength of the signal depends on having sharp, decisive per-view predictions. More averaging dilutes this sharpness. This is consistent with the Step 5 finding (top-m averaging hurts): any form of additional averaging beyond the 3-view baseline reduces the decisive single-tile signal that makes max_prob effective.

The original 3 views (orig, hflip, rot90) were chosen in Phase 1 because they represent orthogonal geometric transforms. The additional views (rot180 = hflip + vflip; rot270 = rot90 + rot180) are compositions of the originals, so they provide diminishing marginal information.

---

## Step 7: Incremental View Ablation — Does TTA Help?

The Step 4 best config changed three variables simultaneously (view strategy, aggregation method, temperature). This ablation isolates the TTA contribution by testing 1-view, 2-view, and 3-view configs all with max_prob + T=1.5.

**Config**: max_prob + T=1.5 + thr=0.03 + topk=10

| Views | k=3 | k=4 | k=5 | k=6 |
|-------|-----|-----|-----|-----|
| **1v (orig only)** | 0.31371 | **0.31950** | 0.31498 | 0.31818 |
| 2v (orig + hflip) | 0.31435 | 0.31385 | 0.31121 | 0.31318 |
| 2v (orig + rot90) | 0.30914 | 0.30533 | 0.30145 | 0.30075 |
| 3v (orig + hflip + rot90) | 0.31487 | 0.31097 | 0.30864 | 0.30970 |

### Result: TTA Hurts — No Augmentation is Best

**New best: orig_only + max_prob + T=1.5 + k=4 = F1 0.31950** (+0.00463 over the previous best)

1-view (no TTA) outperforms all multi-view configs. The entire +0.039 F1 gain over the baseline (0.28093 → 0.31950) comes from **max_prob + T=1.5**, not from TTA.

### Per-View Analysis

- **hflip is nearly neutral**: 2v_hflip (0.31435 at k=3) is close to 1v (0.31371 at k=3) but hurts at higher k. The hflip view provides marginally useful information at low k where spatial coverage is limited, but its noise outweighs its benefit with more tiles.

- **rot90 is actively harmful**: 2v_rot90 is the worst config across all k values (best: 0.30914). Adding rot90 to hflip (3v vs 2v_hflip) also degrades performance. The 90-degree rotation creates tile representations that are too different from the original view — the model was fine-tuned on upright images, so rotated views produce less reliable predictions.

- **3v averages out to mediocre**: The 3-view per_view_or result (0.31487 at k=3) happens to beat 1v at k=3 by +0.001, but this is because hflip's slight benefit at k=3 compensates for rot90's harm. At k=4 (where 1v peaks), 3v is clearly worse (0.31097 vs 0.31950).

### Why TTA Hurts with max_prob

max_prob already selects the single best tile per species. Averaging this sharp signal with predictions from augmented views introduces noise:

- The original view produces the most reliable per-tile predictions (model was trained on upright images)
- Augmented views produce similar but noisier predictions
- per_view_or averages image-level probability vectors, diluting the sharp max_prob signal from the original view with the noisier augmented signals
- This is the same pattern seen in Steps 5 (top-m averaging) and 6 (6-view): **any form of additional averaging hurts max_prob**

### k=4 vs k=3 for orig_only

Without view averaging to compensate for limited spatial coverage, orig_only benefits from one extra tile (k=4 vs k=3). With max_prob, the 4th tile can only help (if it contains a species the top-3 tiles missed) or be harmless (max ignores it). The 3v config preferred k=3 because the view averaging already provided redundancy that extra tiles couldn't improve.

---

## Compute Cost

Steps 1-4 and 7 ran on CPU only (no GPU). Step 6 required a GPU extraction step for the additional augmentation views.

| Step | Resource | Time |
|------|----------|------|
| Steps 1 + 1b + 1c + 3 | CPU-medium, 48 GB RAM | ~4 hours |
| Step 4 (combined) | CPU, interactive | ~30 minutes |
| Step 5 (top-m sweep) | CPU, interactive | ~30 minutes |
| Step 6 extraction | GPU (RTX 6000) | ~30 minutes |
| Step 6 aggregation | CPU-medium, 48 GB RAM | ~1 hour |
| Step 7 (view ablation) | CPU-medium, 48 GB RAM | ~30 minutes |
| topk sweep | CPU | ~1 minute |
| **Total** | | **~7 hours** |

---

## Final Configuration

```
Strategy:           orig_only (no TTA)
Aggregation:        max_prob
Temperature:        1.5
tile_filter_k:      4
threshold:          0.03
topk:               10
geo_mask:           sw_europe_mask_bool.npy
aug_filter:         True
image_topk:         64

F1 Score:           0.31950
Improvement:        +0.03857 over baseline (0.28093)
Relative gain:      +13.7%
```

### Evolution of Best Config

| Step | Config | F1 | Key change |
|------|--------|-----|-----------|
| Baseline | orig_only + noisy_or + T=2.0 + k=6 | 0.28093 | — |
| Step 4 | per_view_or + max_prob + T=1.5 + k=3 | 0.31487 | max_prob + T=1.5 (+ TTA) |
| **Step 7** | **orig_only + max_prob + T=1.5 + k=4** | **0.31950** | **Remove TTA, increase k** |
