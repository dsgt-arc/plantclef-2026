# Exp-4: Enhanced Tile Aggregation — Results Summary

## Executive Summary

Exp-4 improved the PlantCLEF 2026 inference pipeline's F1 score from **0.28093 to 0.31487** (+12.1% relative) by combining two changes: (1) multi-view test-time augmentation via per-view independent aggregation, and (2) replacing Noisy-OR with max-probability aggregation. No model retraining was required — all improvements came from better post-processing of existing DINOv2 embeddings.

**Final best config**: `per_view_or` + `max_prob` + T=1.5 + k=3 + thr=0.03 + topk=10 → **F1 = 0.31487**

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

### Key Finding: per_view_or + max_prob is a Breakthrough

The combination produces a **+0.034 F1 improvement** over the baseline (0.28093 → 0.31487), a 12.1% relative gain. This is the single largest improvement found across all PlantCLEF experiments.

The k-tolerance pattern is dramatic:

```
    noisy_or:  k=3 (0.292) ──── sharp decline ────→ k=10 (0.230)   spread: 0.062
    max_prob:  k=3 (0.315) ──── stable plateau ──→ k=10 (0.309)   spread: 0.006
```

### Explanation: Why This Combination Works

The synergy between per_view_or and max_prob has three components:

**1. per_view_or reduces per-tile variance without inflating tile count.**
Unlike all_tiles (which triples tile count), per_view_or aggregates each view independently and averages the three image-level probability vectors. This means: (a) each view's aggregation operates on the same k tiles, preserving the optimal tile count; (b) the averaging happens at the image level where probabilities are already well-calibrated; (c) species that appear consistently across views get reinforced, while view-specific noise gets suppressed.

**2. max_prob extracts the strongest signal per species.**
For a given species, max_prob selects the single tile where the model is most confident. In vegetation plots, a species may only be visible in 1-2 tiles. Max_prob captures this perfectly — it doesn't penalize the species for being absent from other tiles (unlike mean_prob) or inflate noise from tiles where the species isn't present (unlike noisy_or).

**3. T=1.5 sharpens predictions to match max_prob's strengths.**
Lower temperature makes per-tile softmax distributions more peaked. This benefits max_prob specifically: the maximum probability becomes higher and more decisive for true positives, while false positives (which are less consistent across tiles) don't get amplified. At T=2.0, max_prob alone (0.278) barely trails noisy_or (0.281) because smooth probabilities don't differentiate true positives sharply enough. At T=1.5 with TTA, the signal is both sharp and reliable.

**4. The k=3 sweet spot.**
k=1 and k=2 fail because even max_prob needs at least a few tiles to cover the spatial diversity of a vegetation plot. A single tile may miss species entirely. k=3 provides just enough spatial coverage while keeping only the most confident tiles. Beyond k=3, max_prob's score plateaus (0.309-0.315) rather than declining, confirming that additional tiles neither help nor hurt — max simply ignores them.

### Why noisy_or Collapses at Low k + Low T

The opposite pattern in noisy_or (k=3 → k=10 decline) worsens at T=1.5 because:
- Lower temperature makes per-tile probabilities more extreme (closer to 0 or 1)
- Noisy-OR: P = 1 - prod(1-p). With sharper p values, even a few tiles with moderate false-positive probability compound into high image-level false positives
- More tiles = more compounding = more false positives = lower precision

---

## Complete Results Table

All 65 configurations tested, sorted by F1 score:

| Rank | Strategy | Method | T | k | F1 | vs baseline |
|------|----------|--------|---|---|-----|-------------|
| 1 | per_view_or | max_prob | 1.5 | 3 | **0.31487** | +0.03394 |
| 2 | per_view_or | max_prob | 1.5 | 4 | 0.31097 | +0.03004 |
| 3 | per_view_or | max_prob | 1.5 | 8 | 0.30979 | +0.02886 |
| 4 | per_view_or | max_prob | 1.5 | 6 | 0.30970 | +0.02877 |
| 5 | per_view_or | max_prob | 1.5 | 10 | 0.30889 | +0.02796 |
| 6 | per_view_or | max_prob | 1.5 | 5 | 0.30864 | +0.02771 |
| 7 | per_view_or | noisy_or | 1.5 | 3 | 0.29216 | +0.01123 |
| 8 | per_view_or | noisy_or | 1.5 | 4 | 0.28514 | +0.00421 |
| 9 | avg_logits | noisy_or | 1.5 | 4 | 0.28273 | +0.00180 |
| 10 | orig_only | noisy_or | 2.0 | 6 | 0.28093 | — baseline |
| 11 | per_view_or | noisy_or | 2.0 | 6 | 0.28010 | -0.00083 |
| 12 | orig_only | max_prob | 2.0 | 10 | 0.27841 | -0.00252 |
| 13 | orig_only | max_prob | 2.0 | 16 | 0.27834 | -0.00259 |
| 14 | orig_only | max_prob | 2.0 | 6 | 0.27595 | -0.00498 |
| 15 | per_view_or | noisy_or | 1.5 | 5 | 0.27513 | -0.00580 |
| 16 | per_view_or | max_prob | 1.5 | 2 | 0.27252 | -0.00841 |
| 17 | avg_logits | noisy_or | 2.0 | 10 | 0.27109 | -0.00984 |
| 18 | avg_logits | max_prob | 2.0 | 16 | 0.27076 | -0.01017 |
| 19 | per_view_or | noisy_or | 2.0 | 4 | 0.27012 | -0.01081 |
| 20 | avg_logits | max_prob | 2.0 | 10 | 0.27028 | -0.01065 |
| 21 | orig_only | noisy_or | 2.0 | 10 | 0.26904 | -0.01189 |
| 22 | per_view_or | noisy_or | 2.0 | 8 | 0.26877 | -0.01216 |
| 23 | avg_logits | max_prob | 2.0 | 6 | 0.26868 | -0.01225 |
| 24 | avg_logits | noisy_or | 2.0 | 6 | 0.26857 | -0.01236 |
| 25 | orig_only | weighted_noisy_or | 2.0 | 6 | 0.26851 | -0.01242 |
| 26 | avg_logits | noisy_or | 2.0 | 4 | 0.26789 | -0.01304 |
| 27 | all_tiles | noisy_or | 2.0 | 2 | 0.26764 | -0.01329 |
| 28 | all_tiles | noisy_or | 2.0 | 4 | 0.26708 | -0.01385 |
| 29 | avg_logits | noisy_or | 1.5 | 6 | 0.26610 | -0.01483 |
| 30 | per_view_or | noisy_or | 1.5 | 6 | 0.26516 | -0.01577 |
| 31 | per_view_or | noisy_or | 2.5 | 6 | 0.25879 | -0.02214 |
| 32 | avg_logits | noisy_or | 2.0 | 8 | 0.25826 | -0.02267 |
| 33 | per_view_or | noisy_or | 2.5 | 8 | 0.25714 | -0.02379 |
| 34 | avg_logits | noisy_or | 2.5 | 6 | 0.25664 | -0.02429 |
| 35 | avg_logits | weighted_noisy_or | 2.0 | 10 | 0.25660 | -0.02433 |
| 36 | avg_logits | mean_prob | 2.0 | 6 | 0.25626 | -0.02467 |
| 37 | avg_logits | noisy_or | 2.5 | 4 | 0.25551 | -0.02542 |
| 38 | per_view_or | noisy_or | 2.5 | 4 | 0.25525 | -0.02568 |
| 39 | orig_only | mean_prob | 2.0 | 6 | 0.25492 | -0.02601 |
| 40 | avg_logits | noisy_or | 2.5 | 8 | 0.25445 | -0.02648 |
| 41 | avg_logits | weighted_noisy_or | 2.0 | 16 | 0.25156 | -0.02937 |
| 42 | avg_logits | noisy_or | 2.0 | 16 | 0.24983 | -0.03110 |
| 43 | avg_logits | mean_prob | 2.0 | 10 | 0.24931 | -0.03162 |
| 44 | orig_only | weighted_noisy_or | 2.0 | 10 | 0.24771 | -0.03322 |
| 45 | orig_only | noisy_or | 2.0 | 16 | 0.24730 | -0.03363 |
| 46 | orig_only | weighted_noisy_or | 2.0 | 16 | 0.24709 | -0.03384 |
| 47 | per_view_or | noisy_or | 1.5 | 8 | 0.24677 | -0.03416 |
| 48 | avg_logits | weighted_noisy_or | 2.0 | 6 | 0.24660 | -0.03433 |
| 49 | avg_logits | noisy_or | 1.5 | 8 | 0.24373 | -0.03720 |
| 50 | orig_only | mean_prob | 2.0 | 10 | 0.24329 | -0.03764 |
| 51 | avg_logits | mean_prob | 2.0 | 16 | 0.24232 | -0.03861 |
| 52 | orig_only | mean_prob | 2.0 | 16 | 0.24086 | -0.04007 |
| 53 | per_view_or | max_prob | 1.5 | 1 | 0.23281 | -0.04812 |
| 54 | per_view_or | noisy_or | 1.5 | 10 | 0.22983 | -0.05110 |
| 55 | all_tiles | noisy_or | 2.0 | 6 | 0.22688 | -0.05405 |
| 56 | all_tiles | noisy_or | 2.0 | 8 | 0.20487 | -0.07606 |
| 57 | all_tiles | noisy_or | 2.0 | 16 | 0.20410 | -0.07683 |
| 58 | all_tiles | noisy_or | 2.0 | 10 | 0.19572 | -0.08521 |
| 59 | all_tiles | noisy_or | 2.0 | 12 | 0.19572 | -0.08521 |

---

## Summary of Findings

### What Worked

1. **per_view_or + max_prob + T=1.5 + k=3** — F1 = 0.31487, the best config by a wide margin. The synergy between independent view aggregation (per_view_or) and noise-immune tile combination (max_prob) at sharper temperature produced a +12.1% relative improvement over the baseline.

2. **max_prob is uniquely k-tolerant** — Performance spread of only 0.002-0.006 across k=3 to k=16, compared to 0.034-0.062 for noisy_or. This makes max_prob a robust default aggregator that doesn't require careful k tuning.

3. **Lower temperature + fewer tiles works better with TTA** — T=1.5 + k=4 beats T=2.0 + k=6 when multi-view averaging reduces per-tile variance. The old optimal (T=2.0, k=6) was a noise-tolerance compromise that TTA renders unnecessary.

### What Didn't Work

1. **all_tiles** — Tripling effective tile count by treating views as independent tiles catastrophically inflates Noisy-OR noise. Even at k=2 (6 effective tiles), it underperforms the baseline by 0.013. Eliminated early.

2. **avg_logits at T=2.0** — Averaging logits before softmax at high temperature over-smooths predictions. Only works at T=1.5 + k=4 where sharper base predictions survive the averaging.

3. **weighted_noisy_or** — Still multiplicative, so the fundamental noise inflation problem persists. Weighting by tile confidence doesn't help because the weighting scale is too narrow (tiles passing the k filter already have similar confidence).

4. **mean_prob** — Too conservative. Dilutes strong single-tile species detections by averaging with near-zero probabilities from tiles where the species isn't visible.

### Aggregation Method Characteristics

| Method | Best F1 | k-sensitive? | When it works |
|--------|---------|-------------|---------------|
| noisy_or | 0.28093 | Very (must tune k) | Low k, single view |
| max_prob (=top1) | **0.31487** | No (robust to k) | With TTA, sharp T |
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

## Compute Cost

All experiments ran on CPU only (no GPU). The DINOv2 classifier head was extracted as numpy arrays and logits computed via matrix multiply.

| Step | Resource | Time |
|------|----------|------|
| Steps 1 + 1b + 1c + 3 | CPU-medium, 48 GB RAM | ~4 hours |
| Step 4 (combined) | CPU, interactive | ~30 minutes |
| Step 5 (top-m sweep) | CPU, interactive | ~30 minutes |
| topk sweep | CPU | ~1 minute |
| **Total** | | **~5 hours** |

---

## Final Configuration

```
Strategy:           per_view_or
Aggregation:        max_prob
Temperature:        1.5
tile_filter_k:      3
threshold:          0.03
topk:               10
geo_mask:           sw_europe_mask_bool.npy
aug_filter:         True
image_topk:         64

F1 Score:           0.31487
Improvement:        +0.03394 over baseline (0.28093)
Relative gain:      +12.1%
```
